"""Freeze calibration-only risk models and precompute predictions before new labels.

Owned by the review/confirmation session. This is an actual consumer of the
sealed 200-shape cohort; no confirmation solver outcome is read or fitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.config import load_yaml
from robust_airfoil.constants import FEATURE_COLUMNS, ROOT
from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.freeze import freeze_artifacts, verify_frozen_artifacts
from robust_airfoil.provenance import utc_now, write_json_atomic
from robust_airfoil.research_lineage import resolve_lineage
from robust_airfoil.residual_risk import fit_residual_risk, predict_residual_risk


def main() -> None:
    paths = resolve_lineage(ROOT, 'controlled')
    lineage, reports = paths['results'], paths['reports']
    confirmation = lineage/'confirmation'
    seal_path = confirmation/'GEOMETRY_SELECTION_FROZEN.json'
    model_manifest_path = lineage/'FROZEN_MODEL_MANIFEST.json'
    final_manifest_path = confirmation/'FROZEN_CONFIRMATION_PREDICTOR.json'
    risk_path = confirmation/'CALIBRATION_ONLY_RISK_MODEL.json'
    predictions_path = confirmation/'predictions_before_labels.parquet'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    if seal['model_training_started_at_seal'] or seal['confirmation_solver_labels_generated_at_seal']:
        raise ValueError('Confirmation geometry selection was not sealed before model/label execution')
    if final_manifest_path.exists() or risk_path.exists() or predictions_path.exists():
        raise FileExistsError('Confirmation predictor already exists or is partial; preserve this freeze')
    if (confirmation/'solver_labels').exists() or (confirmation/'CONFIRMATION_RESULTS.json').exists():
        raise RuntimeError('Confirmation outcomes already exist; refuse retrospective predictor fitting')
    config_path = ROOT/'configs/robust_v2/controlled_study.yaml'
    if sha256_file(config_path) != seal['declared_study_config_sha256']:
        raise ValueError('The declared confirmation protocol changed after geometry selection')
    for relative, expected in seal['files_sha256'].items():
        if sha256_file(confirmation/relative) != expected:
            raise ValueError('Sealed confirmation geometry changed: '+relative)
    model_manifest = json.loads(model_manifest_path.read_text(encoding='utf-8'))
    model_artifacts = {name:Path(value) for name,value in model_manifest['artifact_paths'].items()}
    verify_frozen_artifacts(model_manifest_path, model_artifacts)
    source_record = json.loads((lineage/'SOURCE_AT_MODEL_FREEZE.json').read_text(encoding='utf-8'))
    inference_files = ['constants.py','modelling/dataset.py','modelling/models.py','modelling/train.py',
                       'modelling/ensemble.py','modelling/calibrate.py','modelling/support_index.py']
    for relative in inference_files:
        path = ROOT/'src/robust_airfoil'/relative
        if source_record['sources'].get(path.relative_to(ROOT).as_posix()) != sha256_file(path):
            raise ValueError('Inference implementation changed after model freeze: '+relative)
    calibration_path = lineage/'calibration/calibration_predictions.parquet'
    calibration = pd.read_parquet(calibration_path)
    joint = calibration[['mask_cl','mask_log_cd','mask_cm']].astype(bool).all(axis=1)
    risk = fit_residual_risk(calibration.loc[joint].copy(),ridge_alpha=10.)
    risk.update({'created_utc':utc_now(), 'calibration_predictions_sha256':sha256_file(calibration_path),
                 'model_manifest_sha256':sha256_file(model_manifest_path),
                 'geometry_selection_sha256':sha256_file(seal_path),
                 'excluded_incomplete_calibration_rows':int((~joint).sum()),
                 'risk_implementation_sha256':sha256_file(ROOT/'src/robust_airfoil/residual_risk.py')})
    write_json_atomic(risk_path,risk)
    artifacts = {**model_artifacts, 'model_manifest':model_manifest_path, 'risk_model':risk_path,
                 'confirmation_geometry_selection':seal_path, 'study_protocol':config_path,
                 'confirmation_parameters':confirmation/'geometries.parquet'}
    manifest = freeze_artifacts(final_manifest_path,artifacts,{
        'purpose':'Prospective numerical confirmation of fixed prediction and risk ranking on sealed new geometries',
        'geometry_count':len(seal['candidates']), 'confirmation_labels_generated':False,
        'selection_status':'Geometry-only source/parents/weights sealed before controlled model training',
        'operational_screen_policy_changed':False,
    })
    # Delay GPU framework loading until all prerequisites and the risk freeze
    # have passed. No solver command or aerodynamic outcome is accessed here.
    import torch

    from robust_airfoil.modelling.calibrate import apply_trust_model
    from robust_airfoil.modelling.ensemble import LoadedEnsemble

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    geometries = pd.read_parquet(confirmation/'geometries.parquet')
    alphas = np.asarray(load_yaml(config_path)['service_alpha_deg'],float)
    # The confirmation assesses the complete declared data-incidence domain,
    # not only service points or the subset accepted by the empirical screen.
    alpha_grid = np.arange(-12.,16.0001,.25)
    rows = []
    for _, geometry in geometries.iterrows():
        frame = pd.DataFrame(np.tile(geometry[FEATURE_COLUMNS[:-1]].to_numpy(float),(len(alpha_grid),1)),columns=FEATURE_COLUMNS[:-1])
        frame['alpha_deg'] = alpha_grid
        frame['airfoil_id'] = str(geometry.airfoil_id)
        frame['confirmation_index'] = int(geometry.confirmation_index)
        frame['geometry_sha256'] = str(geometry.geometry_sha256)
        rows.append(frame)
    query = pd.concat(rows,ignore_index=True)
    development = pd.read_parquet(lineage/'partition_inputs/development.parquet')
    rule = json.loads((lineage/'calibration/trust_calibration.json').read_text(encoding='utf-8'))
    ensemble = LoadedEnsemble(lineage/'full_ensemble')
    predictions = apply_trust_model(development,ensemble.predict(query),ensemble.first_scaling,rule)
    scores = predict_residual_risk(predictions,risk)
    for name,values in scores.items():
        predictions['risk_'+name] = values
    predictions.to_parquet(predictions_path,index=False)
    verify_frozen_artifacts(final_manifest_path,artifacts)
    receipt = {'status':'PREDICTIONS_AND_RISK_FROZEN_BEFORE_CONFIRMATION_LABELS',
               'created_utc':utc_now(), 'geometries':len(geometries), 'requested_states':len(predictions),
               'alpha_grid':alpha_grid.tolist(), 'service_grid_preserved_separately':alphas.tolist(),
               'predictor_manifest_sha256':sha256_file(final_manifest_path),
               'prediction_file_sha256':sha256_file(predictions_path),
               'calibration_risk_file_sha256':sha256_file(risk_path),
               'geometry_selection_sha256':sha256_file(seal_path),
               'confirmation_labels_generated':False,
               'scope':'Outcomes may now be generated on these exact files, without retraining or tuning the frozen predictor/risk model.'}
    write_json_atomic(confirmation/'PREDICTIONS_BEFORE_LABELS_RECEIPT.json',receipt)
    write_json_atomic(reports/'CONFIRMATION_PREDICTOR_FROZEN.json',{'manifest':manifest,'receipt':receipt})
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
