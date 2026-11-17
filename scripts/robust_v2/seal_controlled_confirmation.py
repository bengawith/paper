"""Seal unseen numerical geometries before controlled-model training or labels.

Owned by the active review/confirmation session; see REVIEW_COORDINATION.md.
This does not run a solver, fit a model, inspect aerodynamic labels or choose
cases based on prediction quality. All original experiment artifacts survive.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.config import load_yaml
from robust_airfoil.confirmation_geometry import build_confirmation_geometry
from robust_airfoil.constants import FEATURE_COLUMNS
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.hashing import sha256_file
from robust_airfoil.provenance import utc_now, write_json_atomic
from robust_airfoil.research_lineage import resolve_lineage

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    paths = resolve_lineage(ROOT, 'controlled')
    lineage, reports = paths['results'], paths['reports']
    destination = lineage/'confirmation'
    manifest_path = destination/'GEOMETRY_SELECTION_FROZEN.json'
    if manifest_path.exists() or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError('Confirmation selection already exists or is partial; preserve it rather than reselecting')
    training = lineage/'full_ensemble'
    if training.exists() and any(training.rglob('*.pt')):
        raise RuntimeError('Controlled training already started; cannot retrospectively call a selection pre-training')
    config_path = ROOT/'configs/robust_v2/controlled_study.yaml'
    config = load_yaml(config_path)['confirmation']
    shapes_path = lineage/'shapes_and_partitions.parquet'
    shapes = pd.read_parquet(shapes_path)
    if shapes.airfoil_id.duplicated().any() or not shapes['split'].isin(['development','calibration','locked_test']).all():
        raise ValueError('The sealed source geometry identity/partition contract is invalid')
    parameters = shapes[FEATURE_COLUMNS[:-1]].to_numpy(float)
    development = np.flatnonzero(shapes['split'].eq('development').to_numpy())
    result = build_confirmation_geometry(
        parameters, development, shapes.cluster_id.to_numpy(),
        count=int(config['geometries']), seed=int(config['seed']),
        minimum_rms_separation=float(config['minimum_rms_separation']),
        mixing_limits=tuple(float(x) for x in config['mixing_weights']),
    )
    destination.mkdir(parents=True)
    reports.mkdir(parents=True, exist_ok=True)
    x = (1-np.cos(np.linspace(0,np.pi,201)))/2
    table = []
    files = {}
    for candidate in result['candidates']:
        name = candidate['confirmation_id']
        parent_indices = candidate['parent_indices']
        candidate['parent_airfoil_ids'] = [str(shapes.iloc[index].airfoil_id) for index in parent_indices]
        candidate['parent_geometry_cluster_ids'] = [int(shapes.iloc[index].cluster_id) for index in parent_indices]
        p = np.asarray(candidate['parameters'], float)
        upper, lower = reconstruct_cst(p, x)
        coordinates = np.vstack([np.c_[x, upper][::-1], np.c_[x, lower][1:]])
        geometry = destination/'inputs'/name/'geometry.dat'
        geometry.parent.mkdir(parents=True)
        geometry.write_text('_'+name+'\n'+'\n'.join(f'{a:.12f} {b:.12f}' for a,b in coordinates)+'\n', encoding='utf-8')
        np.testing.assert_allclose(parse_coordinate_file(geometry), coordinates, rtol=0, atol=5.1e-13)
        candidate['geometry_sha256'] = sha256_file(geometry)
        candidate['geometry_path'] = geometry.relative_to(ROOT).as_posix()
        files[geometry.relative_to(destination).as_posix()] = candidate['geometry_sha256']
        table.append({'airfoil_id':name, 'confirmation_index':len(table), 'source_tier':'sealed_development_geometry_mixture',
                      'geometry_sha256':candidate['geometry_sha256'],
                      **dict(zip(FEATURE_COLUMNS[:-1], p, strict=True))})
    cohort_path = destination/'geometries.parquet'
    pd.DataFrame(table).to_parquet(cohort_path,index=False)
    files[cohort_path.relative_to(destination).as_posix()] = sha256_file(cohort_path)
    manifest = {
        **result, 'created_utc':utc_now(),
        'geometry_source_sha256':sha256_file(shapes_path),
        'source_split_sha256':sha256_file(lineage/'splits/full.json'),
        'declared_study_config_sha256':sha256_file(config_path),
        'selection_implementation_sha256':sha256_file(ROOT/'src/robust_airfoil/confirmation_geometry.py'),
        'sealing_script_sha256':sha256_file(Path(__file__)),
        'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'files_sha256':files,
        'model_training_started_at_seal':False,
        'confirmation_solver_labels_generated_at_seal':False,
        'scope':'Unseen represented geometry mixtures with an explicitly chosen sampling law; no model or aerodynamic outcome selected them. Not a natural manufacturing population or an unseen-family benchmark.',
    }
    if training.exists() and any(training.rglob('*.pt')):
        raise RuntimeError('Controlled training began concurrently during sealing; preserve these files and reconcile before any prospective claim')
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(reports/'CONFIRMATION_SELECTION_FROZEN.json', manifest)
    print(json.dumps({'status':'GEOMETRY_SELECTION_SEALED_BEFORE_CONTROLLED_TRAINING',
                      'count':len(table), 'attempts':result['attempts'],
                      'rejected_attempts':result['rejected_attempts'],
                      'manifest_sha256':sha256_file(manifest_path),
                      'minimum_observed_rms_separation':min(c['minimum_rms_to_prior_shapes'] for c in result['candidates'])},indent=2),flush=True)


if __name__ == '__main__':
    main()
