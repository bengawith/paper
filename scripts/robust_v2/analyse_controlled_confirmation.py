"""Compare sealed before-label predictions with every genuine confirmation outcome.

This analysis does not fit the aerodynamic predictor, risk model, screen,
residual scales, cutoffs or geometry selection. It reports unsuccessful cases
and conditional-on-observation statistics without imputing their missing truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from robust_airfoil.config import load_yaml
from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.freeze import verify_frozen_artifacts
from robust_airfoil.provenance import utc_now, write_json_atomic
from robust_airfoil.research_lineage import resolve_lineage
from robust_airfoil.research_metrics import geometry_bootstrap_metrics, prediction_metrics
from robust_airfoil.residual_risk import label_residual_risk, predict_residual_risk

ROOT=Path(__file__).resolve().parents[2]


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> dict:
    if len(np.unique(labels))!=2:
        return {'auroc':None,'average_precision':None,'reason':'Both outcome classes are required'}
    return {'auroc':float(roc_auc_score(labels,scores,sample_weight=weights)),
            'average_precision':float(average_precision_score(labels,scores,sample_weight=weights))}


def main() -> None:
    paths=resolve_lineage(ROOT,'controlled')
    lineage=paths['results']
    output=paths['reports']/'CONFIRMATION'
    confirmation=lineage/'confirmation'
    target=output/'CONFIRMATION_RESULTS.json'
    if target.exists():
        raise FileExistsError('Completed prospective confirmation is immutable; use a new analysis revision')
    complete_path=confirmation/'SOLVER_LABELS_COMPLETE.json'
    complete=json.loads(complete_path.read_text(encoding='utf-8'))
    selection_path=confirmation/'GEOMETRY_SELECTION_FROZEN.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    before_path=confirmation/'PREDICTIONS_BEFORE_LABELS_RECEIPT.json'
    before=json.loads(before_path.read_text(encoding='utf-8'))
    frozen_path=confirmation/'FROZEN_CONFIRMATION_PREDICTOR.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    frozen_artifacts={name:Path(path) for name,path in frozen['artifact_paths'].items()}
    verify_frozen_artifacts(frozen_path,frozen_artifacts)
    risk_path=confirmation/'CALIBRATION_ONLY_RISK_MODEL.json'
    risk=json.loads(risk_path.read_text(encoding='utf-8'))
    if risk['risk_implementation_sha256']!=sha256_file(ROOT/'src/robust_airfoil/residual_risk.py'):
        raise ValueError('Frozen residual-risk implementation changed before confirmation analysis')
    predicted_path=confirmation/'predictions_before_labels.parquet'
    observed_path=confirmation/'observed_confirmation_points.parquet'
    if sha256_file(predicted_path)!=before['prediction_file_sha256'] or sha256_file(observed_path)!=complete['observed_points_sha256']:
        raise ValueError('Confirmation prediction/observation bytes changed')
    for relative,expected in selection['files_sha256'].items():
        if sha256_file(confirmation/relative)!=expected:
            raise ValueError('A sealed confirmation geometry/parameter file changed')
    predicted=pd.read_parquet(predicted_path)
    observed=pd.read_parquet(observed_path)
    if predicted.duplicated(['airfoil_id','alpha_deg']).any() or observed.duplicated(['airfoil_id','alpha_deg']).any():
        raise ValueError('Confirmation join keys are not one-to-one')
    selected_names={case['confirmation_id'] for case in selection['candidates']}
    if set(predicted.airfoil_id)!=selected_names or not set(observed.airfoil_id).issubset(selected_names):
        raise ValueError('Prediction/solver cohorts differ from sealed geometry selection')
    observed_columns=['airfoil_id','alpha_deg','cl','cd','log_cd','cm','mask_cl','mask_log_cd','mask_cm','geometry_sha256']
    combined=predicted.merge(observed[observed_columns],on=['airfoil_id','alpha_deg'],how='left',
                             suffixes=('_prediction','_observed'),validate='one_to_one',indicator=True)
    accounted=combined['_merge'].eq('both')
    if int(accounted.sum())!=len(observed):
        raise ValueError('Some actual solver observations cannot be matched to before-label predictions')
    if not combined.loc[accounted,'geometry_sha256_prediction'].eq(combined.loc[accounted,'geometry_sha256_observed']).all():
        raise ValueError('A confirmation coefficient was predicted for different coordinates than its solver label')
    valid=combined.loc[accounted].drop(columns=['_merge']).copy()
    for name in ['mask_cl','mask_log_cd','mask_cm']:
        valid[name]=valid[name].astype(bool)
    if not valid[['mask_cl','mask_log_cd','mask_cm']].all().all():
        raise ValueError('Confirmation admitted observations contain incomplete target rows')
    # Verify risk values using only their frozen observable features; query
    # labels cannot alter this calculation or the already stored predictions.
    risk_predictions=predict_residual_risk(valid,risk)
    for name,scores in risk_predictions.items():
        np.testing.assert_allclose(scores,valid['risk_'+name].to_numpy(float),rtol=0,atol=1e-12)
    label=label_residual_risk(valid,risk)
    binary=label['high_error'].astype(int)
    rules_path=lineage/'calibration/trust_calibration.json'
    rule=json.loads(rules_path.read_text(encoding='utf-8'))
    study=load_yaml(ROOT/'configs/robust_v2/controlled_study.yaml')
    repetitions=int(study['uncertainty_reporting']['geometry_bootstrap_replicates'])
    seed=int(study['uncertainty_reporting']['seed'])
    rows=prediction_metrics(valid,rule)
    bootstrap=geometry_bootstrap_metrics(valid,rule,replicates=repetitions,seed=seed)
    unique,index=np.unique(valid.airfoil_id.astype(str).to_numpy(),return_inverse=True)
    counts=np.bincount(index)
    equal_geometry_weights=1./counts[index]
    ranking={name:{**ranking_metrics(binary,scores),
                   'geometry_balanced':ranking_metrics(binary,scores,equal_geometry_weights)}
             for name,scores in risk_predictions.items()}
    rng=np.random.default_rng(seed)
    differences=[]
    for _ in range(repetitions):
        weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))[index]
        if weights[binary==0].sum()==0 or weights[binary==1].sum()==0:
            continue
        delta=roc_auc_score(binary,risk_predictions['support_plus_ensemble'],sample_weight=weights)
        delta-=roc_auc_score(binary,risk_predictions['support_only'],sample_weight=weights)
        differences.append(float(delta))
    retention=[]
    for name,scores in risk_predictions.items():
        for percentage,threshold in risk['models'][name]['calibration_retention_thresholds'].items():
            keep=scores<=float(threshold)
            selected=valid.loc[keep].copy()
            selected['scaled_error']=label['scaled_error'][keep]
            retention.append({'model':name,'calibration_quantile_percent':int(percentage),
                              'frozen_threshold':threshold,'retained_confirmation_states':int(keep.sum()),
                              'actual_confirmation_retained_fraction':float(keep.mean()),
                              'retained_geometries':int(selected.airfoil_id.nunique()),
                              'conditional_geometry_macro_scaled_error':float(selected.groupby('airfoil_id').scaled_error.mean().mean()) if len(selected) else None})
    missing=(~accounted).to_numpy(int)
    failure_ranking={name:ranking_metrics(missing,predicted['risk_'+name].to_numpy(float)) for name in risk_predictions}
    delta_point=(ranking['support_plus_ensemble']['auroc']-ranking['support_only']['auroc']
                 if ranking['support_plus_ensemble']['auroc'] is not None and ranking['support_only']['auroc'] is not None else None)
    report={'schema':'sealed-before-training-geometry-before-label-predictor-confirmation-v1',
            'completed_utc':utc_now(),'requested_geometries':len(selected_names),
            'geometries_with_observations':int(valid.airfoil_id.nunique()),
            'requested_states':len(predicted),'observed_states':len(valid),
            'missing_solver_observations':int((~accounted).sum()),
            'accepted_observed_states':int(valid.trusted_domain.sum()),
            'model_metrics':rows,'fixed_model_geometry_bootstrap':bootstrap,
            'frozen_calibration_defined_high_error_fraction':float(binary.mean()),
            'risk_models':ranking,'incremental_combined_minus_support_auroc':delta_point,
            'fixed_model_geometry_bootstrap_delta_interval':np.quantile(differences,[.025,.975]).tolist() if differences else None,
            'risk_bootstrap_replicates_requested':repetitions,'risk_bootstrap_replicates_usable':len(differences),
            'risk_vs_missing_solver_observation_diagnostic':failure_ranking,
            'solver_accounting':complete,
            'input_hashes':{'selection':sha256_file(selection_path),'before_labels_receipt':sha256_file(before_path),
                            'frozen_predictor':sha256_file(frozen_path),'risk_model':sha256_file(risk_path),
                            'predictions':sha256_file(predicted_path),'observed_labels':sha256_file(observed_path)},
            'claims':{'geometry_selection_preceded_controlled_training':True,'predictions_and_risk_preceded_confirmation_labels':True,
                      'confirmation_used_to_fit_or_select_model':False,'operational_trust_policy_changed':False,
                      'independent_physical_validation':False},
            'limits':['The cohort samples explicit mixtures of development geometry; it is not an unseen-family or manufacturing-population sample.',
                      'Local XFOIL shares the training numerical method; fresh executions are not independent RANS CFD or experiments.',
                      'Error metrics condition on clean-process observed states. Missing solver truth is never imputed.',
                      'Whole-observed-curve coverage is not a simultaneous coverage guarantee.',
                      'Geometry bootstrap holds predictions fixed and does not include full refitting variability; shared parents and selection can add dependence.',
                      'A favourable ranking diagnostic does not automatically validate or deploy a new operational threshold.']}
    valid.to_parquet(confirmation/'joined_observed_predictions.parquet',index=False)
    pd.DataFrame(rows).to_csv(output/'CONFIRMATION_METRICS.csv',index=False)
    pd.DataFrame(retention).to_csv(output/'FROZEN_THRESHOLD_RETENTION.csv',index=False)
    write_json_atomic(target,report)
    write_json_atomic(confirmation/'CONFIRMATION_RESULTS.json',report)
    print(json.dumps({key:value for key,value in report.items() if key not in ['model_metrics','fixed_model_geometry_bootstrap','solver_accounting','limits']},indent=2),flush=True)
    print('SEALED_CONFIRMATION_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
