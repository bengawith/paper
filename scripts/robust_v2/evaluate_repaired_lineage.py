"""Calibrate, freeze, then evaluate the genuinely retrained geometry lineage.

The historical held-out partition is protected from fitting but has been
inspected in earlier work. This is not relabelled as a pristine prospective test.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from robust_airfoil.data.splits import assert_locked_test_access
from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.calibrate import apply_trust_model, calibrate_trust_model
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS, TARGET_COLUMNS, fit_scaling
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.modelling.freeze import freeze_artifacts
from robust_airfoil.modelling.support_index import GroupedSupportIndex
from robust_airfoil.provenance import utc_now
from robust_airfoil.research_lineage import resolve_lineage

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT/'results/robust_v2/lineages/robust-v2-rebuilt-20261014'
OUT = ROOT/'reports/robust_v2/external_academic_review_20261014/REBUILT_MODEL'


def metrics(frame: pd.DataFrame, rule: dict) -> list[dict]:
    records = []
    for label, subset in [('all',frame),('accepted',frame.loc[frame.trusted_domain]),('rejected',frame.loc[~frame.trusted_domain])]:
        for target in [*TARGET_COLUMNS,'cd']:
            base = 'log_cd' if target == 'cd' else target
            mask = subset['mask_'+base].astype(bool)
            chosen = subset.loc[mask].copy()
            actual = chosen[base].to_numpy(float)
            predicted = chosen['prediction_'+base].to_numpy(float)
            if target == 'cd':
                actual,predicted = np.exp(actual),np.exp(predicted)
            finite = np.isfinite(actual) & np.isfinite(predicted)
            chosen,actual,predicted = chosen.loc[finite],actual[finite],predicted[finite]
            if not len(chosen):
                records.append({'subset':label,'target':target,'states':0,'airfoils':0})
                continue
            error = abs(actual-predicted)
            squared = error**2
            band = float(rule['residual_absolute_quantiles'][base]['0.95'])
            lower,upper = chosen['prediction_'+base].to_numpy(float)-band,chosen['prediction_'+base].to_numpy(float)+band
            if target == 'cd':
                lower,upper = np.exp(lower),np.exp(upper)
            coverage = (actual>=lower)&(actual<=upper)
            grouping = pd.DataFrame({'airfoil_id':chosen.airfoil_id.to_numpy(),'abs_error':error,'squared_error':squared,'covered':coverage})
            records.append({'subset':label,'target':target,'states':len(chosen),'airfoils':int(chosen.airfoil_id.nunique()),'geometry_clusters':int(chosen.cluster_id.nunique()),'mae':float(error.mean()),'mse':float(squared.mean()),'r2':float(r2_score(actual,predicted)) if len(actual)>1 else None,'geometry_macro_mae':float(grouping.groupby('airfoil_id').abs_error.mean().mean()),'geometry_macro_mse':float(grouping.groupby('airfoil_id').squared_error.mean().mean()),'state_interval_coverage':float(coverage.mean()),'geometry_macro_interval_coverage':float(grouping.groupby('airfoil_id').covered.mean().mean()),'whole_observed_curve_coverage':float(grouping.groupby('airfoil_id').covered.all().mean())})
    return records


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--lineage', choices=['rebuilt','controlled'], default='rebuilt', help='Explicit immutable scientific revision; never overwrites the other lineage')
    args=parser.parse_args()
    paths=resolve_lineage(ROOT,args.lineage)
    LINEAGE=paths['results']
    OUT=paths['reports']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if (OUT/'REPAIRED_TEST_METRICS.json').exists():
        raise FileExistsError('Completed test evidence exists; do not silently rerun a frozen evaluation')
    (LINEAGE/'calibration').mkdir(exist_ok=True)
    (LINEAGE/'locked_test').mkdir(exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    partitions = json.loads((LINEAGE/'partition_inputs/PARTITIONS.json').read_text(encoding='utf-8'))
    for name,expected in partitions['partition_sha256'].items():
        if sha256_file(LINEAGE/'partition_inputs'/(name+'.parquet')) != expected:
            raise ValueError('Partition file changed: '+name)
    development = pd.read_parquet(LINEAGE/'partition_inputs/development.parquet')
    calibration = pd.read_parquet(LINEAGE/'partition_inputs/calibration.parquet')
    ensemble = LoadedEnsemble(LINEAGE/'full_ensemble')
    calibrated = ensemble.predict(calibration)
    rule = calibrate_trust_model(development,calibrated,ensemble.first_scaling,LINEAGE/'calibration/trust_calibration.json')
    checked = apply_trust_model(development,calibrated,ensemble.first_scaling,rule)
    if not np.array_equal(checked.trusted_domain,calibrated.trusted_domain):
        raise AssertionError('Saved calibration decisions and runtime decisions differ')
    calibrated.to_parquet(LINEAGE/'calibration/calibration_predictions.parquet',index=False)
    # Independently verify the accelerated support algorithm against the original
    # full 13-D nearest-neighbour search, before any optimisation uses it.
    scaler = ensemble.first_scaling
    training_features = (development[FEATURE_COLUMNS].to_numpy(float)-scaler.feature_center)/scaler.feature_scale
    query_features = (calibration[FEATURE_COLUMNS].to_numpy(float)-scaler.feature_center)/scaler.feature_scale
    old_start = time.monotonic()
    old = cKDTree(training_features).query(query_features,k=5)[0].mean(axis=1)
    old_time = time.monotonic()-old_start
    new_start = time.monotonic()
    new = GroupedSupportIndex(training_features).query(query_features,k=5)[0].mean(axis=1)
    new_time = time.monotonic()-new_start
    np.testing.assert_allclose(new,old,atol=1e-11,rtol=1e-11)
    support_audit = {'queries':len(query_features),'reference_states':len(training_features),'max_absolute_distance_difference':float(np.max(abs(new-old))),'full_state_tree_seconds':old_time,'exact_grouped_tree_seconds':new_time,'grouped_index_source_sha256':sha256_file(ROOT/'src/robust_airfoil/modelling/support_index.py')}
    (OUT/'EXACT_SUPPORT_RECOMPUTATION.json').write_text(json.dumps(support_audit,indent=2),encoding='utf-8')
    source_hashes = {p.relative_to(ROOT).as_posix():sha256_file(p) for p in (ROOT/'src/robust_airfoil').rglob('*.py')}
    source_record = LINEAGE/'SOURCE_AT_MODEL_FREEZE.json'
    source_record.write_text(json.dumps({'utc':utc_now(),'sources':source_hashes},indent=2),encoding='utf-8')
    artifacts = {
        'source_record':source_record,'data_manifest':LINEAGE/'DATA_MANIFEST.json',
        'splits':LINEAGE/'splits/full.json','partition_manifest':LINEAGE/'partition_inputs/PARTITIONS.json',
        'development':LINEAGE/'partition_inputs/development.parquet','calibration':LINEAGE/'partition_inputs/calibration.parquet',
        'locked_test':LINEAGE/'partition_inputs/locked_test.parquet',
        'ensemble_manifest':LINEAGE/'full_ensemble/ensemble_manifest.json',
        'trust_calibration':LINEAGE/'calibration/trust_calibration.json',
        'calibration_predictions':LINEAGE/'calibration/calibration_predictions.parquet',
    }
    for i in range(5):
        for label,filename in [('checkpoint','checkpoint.pt'),('scaling','scaling.joblib')]:
            artifacts[f'member_{i:03d}_{label}'] = LINEAGE/f'full_ensemble/member_{i:03d}'/filename
    frozen_path = LINEAGE/'FROZEN_MODEL_MANIFEST.json'
    frozen = freeze_artifacts(frozen_path,artifacts,{'lineage':LINEAGE.name,'label_source':'actual local XFOIL on exact represented CST geometry' if args.lineage=='controlled' else 'archived source polars','geometry_policy':'corrected order, nonnegative fitted TE,0.005 chord cap,2001-point reconstruction screen','test_status':'historical test revisited; protected from training and calibration, not newly pristine','operational_policy':'existing empirical support/disagreement rule recalibrated; no combined risk policy release'})
    (OUT/'FROZEN_MODEL_MANIFEST.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    assert_locked_test_access(frozen_path,'locked_test',artifacts)
    test = pd.read_parquet(LINEAGE/'partition_inputs/locked_test.parquet')
    prediction = apply_trust_model(development,ensemble.predict(test),ensemble.first_scaling,rule)
    prediction.to_parquet(LINEAGE/'locked_test/predictions.parquet',index=False)
    records = metrics(prediction,rule)
    report = {'schema':'repaired-five-member-actual-heldout-evaluation-v1','utc':utc_now(),'states':len(test),'airfoils':int(test.airfoil_id.nunique()),'geometry_clusters':int(test.cluster_id.nunique()),'accepted_states':int(prediction.trusted_domain.sum()),'rejected_states':int((~prediction.trusted_domain).sum()),'model_frozen_before_test':True,'frozen_manifest_sha256':sha256_file(frozen_path),'predictions_sha256':sha256_file(LINEAGE/'locked_test/predictions.parquet'),'metrics':records,'limitations':['Geometry identities were previously inspected; the protected cohort is not pristine prospective sampling. No new test outcomes were used to fit or stop the models.','Representation-admitted corpus differs from the1560-shape historical corpus; metrics are not a like-for-like improvement percentage.','XFOIL source labels are numerical, not independent CFD or experiments.','State bands do not guarantee polar-wise, accepted-subset or optimiser-shift coverage.']}
    (OUT/'REPAIRED_TEST_METRICS.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT/'REPAIRED_TEST_METRICS.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    # Fixed simple baselines on exactly the same repaired development/test
    # partitions. No model/threshold selection is based on these test values.
    baseline_rows=[]
    baseline_scaler = fit_scaling(development)
    x_train = (development[FEATURE_COLUMNS].to_numpy(float)-baseline_scaler.feature_center)/baseline_scaler.feature_scale
    x_test = (test[FEATURE_COLUMNS].to_numpy(float)-baseline_scaler.feature_center)/baseline_scaler.feature_scale
    for target in TARGET_COLUMNS:
        eligible = development['mask_'+target].astype(bool).to_numpy()
        test_mask = test['mask_'+target].astype(bool).to_numpy()
        truth = test.loc[test_mask,target].to_numpy(float)
        train_y = development.loc[eligible,target].to_numpy(float)
        weights = development.loc[eligible,'airfoil_weight'].to_numpy(float)
        for name,model in [('ridge',Ridge(alpha=1.0)),('hist_gradient_boosting',HistGradientBoostingRegressor(max_iter=200,learning_rate=.1,max_leaf_nodes=31,early_stopping=False,random_state=20261014))]:
            start=time.monotonic()
            model.fit(x_train[eligible],train_y,sample_weight=weights)
            values=model.predict(x_test[test_mask])
            differences=pd.DataFrame({'airfoil_id':test.loc[test_mask,'airfoil_id'].to_numpy(),'error':abs(values-truth)})
            baseline_rows.append({'model':name,'target':target,'r2':float(r2_score(truth,values)),'mae':float(mean_absolute_error(truth,values)),'geometry_macro_mae':float(differences.groupby('airfoil_id').error.mean().mean()),'seconds_train_predict':time.monotonic()-start,'states':len(truth),'sample_weighting':'inverse states per development aerofoil','hyperparameters':model.get_params()})
        values=np.full(len(truth),np.average(train_y,weights=weights))
        baseline_rows.append({'model':'weighted_mean','target':target,'r2':float(r2_score(truth,values)),'mae':float(mean_absolute_error(truth,values)),'states':len(truth)})
    (OUT/'REPAIRED_BASELINES.json').write_text(json.dumps({'schema':'same-repaired-split-fixed-baselines-v1','rows':baseline_rows,'selection':'predeclared baseline configurations, not selected on test'},indent=2,allow_nan=False),encoding='utf-8')
    print('REPAIRED_CALIBRATION_FREEZE_TEST_AND_BASELINES_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
