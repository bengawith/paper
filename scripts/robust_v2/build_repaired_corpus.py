"""Rebuild genuine archived source rows under the predeclared repaired geometry policy."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.data.builder import build_long_form_dataset
from robust_airfoil.data.splits import protect_rebuilt_splits, write_split_manifest
from robust_airfoil.geometry.clustering import cluster_coordinate_vectors
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.geometry.normalise import normalise_geometry
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'data/robust_v2/processed/full_exact'
DATA = ROOT/'data/robust_v2/processed/rebuilt_20261014'
LINEAGE = ROOT/'results/robust_v2/lineages/robust-v2-rebuilt-20261014'
REVIEW = ROOT/'reports/robust_v2/external_academic_review_20261014/REBUILT_MODEL'
OLD_SPLIT = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908/splits/full.json'


def main() -> None:
    if (LINEAGE/'DATA_MANIFEST.json').exists():
        raise FileExistsError('Rebuilt corpus is already frozen; do not overwrite it')
    if DATA.exists() and any(DATA.iterdir()):
        raise FileExistsError('Nonempty rebuilt data directory requires inspection, not silent overwrite')
    old_shapes = pd.read_parquet(ORIGINAL/'airfoils.parquet')
    old_points = pd.read_parquet(ORIGINAL/'model_points.parquet')
    polar_paths = [Path(p) for p in sorted(old_points.source_path.unique())]
    geometry_paths = [Path(p) for p in sorted(old_shapes.geometry_path.unique())]
    source_hashes = {str(p.relative_to(ROOT)).replace('\\','/'):sha256_file(p) for p in polar_paths+geometry_paths}
    initial = build_long_form_dataset(polar_paths,geometry_paths,DATA,'archived_airfoiltools_xfoil',cosine_points=201,near_duplicate_threshold=1e-4)
    shapes = pd.read_parquet(DATA/'airfoils.parquet')
    points = pd.read_parquet(DATA/'model_points.parquet')
    admitted, vectors, audit = [], [], []
    dense_x = (1-np.cos(np.linspace(0,np.pi,2001)))/2
    for _, row in shapes.iterrows():
        parameters = row[FEATURE_COLUMNS[:-1]].to_numpy(float)
        upper, lower = reconstruct_cst(parameters,dense_x)
        validity = validate_geometry(np.column_stack([dense_x,upper]),np.column_stack([dense_x,lower]))
        normal = normalise_geometry(parse_coordinate_file(Path(row.geometry_path)),201)
        dense_error = max(float(np.max(abs(upper-np.interp(dense_x,normal.upper[:,0],normal.upper[:,1])))),float(np.max(abs(lower-np.interp(dense_x,normal.lower[:,0],normal.lower[:,1])))))
        reasons = list(validity.reasons)
        if max(float(row.cst_max_error),dense_error) > .005:
            reasons.append('maximum_cst_error_above_0p005_chord')
        audit.append({'airfoil_id':str(row.airfoil_id),'admitted':not reasons,'reasons':';'.join(reasons),'fit_max_error':float(row.cst_max_error),'dense_max_error':dense_error,'te_thickness':float(parameters[-1]),'geometry_sha256':str(row.geometry_sha256)})
        if not reasons:
            admitted.append(str(row.airfoil_id))
            vectors.append(np.r_[normal.upper[:,1],normal.lower[:,1]])
    shapes = shapes.loc[shapes.airfoil_id.isin(admitted)].copy().reset_index(drop=True)
    assert shapes.airfoil_id.tolist() == admitted
    shapes['cluster_id'] = cluster_coordinate_vectors(np.asarray(vectors),rms_threshold=1e-4)
    points = points.loc[points.airfoil_id.isin(admitted)].copy()
    counts = points.groupby('airfoil_id').size()
    points['airfoil_weight'] = points.airfoil_id.map(1.0/counts)
    points['cluster_id'] = points.airfoil_id.map(shapes.set_index('airfoil_id').cluster_id)
    splits = protect_rebuilt_splits(shapes,pd.DataFrame(json.loads(OLD_SPLIT.read_text(encoding='utf-8'))['records']))
    points['split'] = points.airfoil_id.map(splits.set_index('airfoil_id')['split'])
    shapes.to_parquet(DATA/'airfoils.parquet',index=False)
    points.to_parquet(DATA/'model_points.parquet',index=False)
    LINEAGE.mkdir(parents=True,exist_ok=True)
    REVIEW.mkdir(parents=True,exist_ok=True)
    write_split_manifest(splits,LINEAGE/'splits/full.json',seed=2026101400)
    pd.DataFrame(audit).to_csv(REVIEW/'GEOMETRY_ADMISSION.csv',index=False)
    summary = {'schema':'repaired-physical-geometry-corpus-v1','source_row_policy':'actual archived polar files parsed again; no synthetic aerodynamic labels or repaired-vector/old-checkpoint substitution','starting_source_airfoils':len(old_shapes),'starting_source_states':len(old_points),'initial_builder':initial,'admitted_airfoils':len(shapes),'admitted_states':len(points),'geometry_clusters':int(shapes.cluster_id.nunique()),'split_counts':[{ 'split':name,'airfoils':int(group.airfoil_id.nunique()),'clusters':int(group.cluster_id.nunique()),'states':int(points.split.eq(name).sum())} for name,group in splits.groupby('split')],'geometry_policy':{'max_fit_error_chord':.005,'dense_verification_points':2001,'nonnegative_fitted_te':True,'postfit_surface_validity':True,'near_duplicate_rms_threshold':1e-4,'partition_protection':'new physical cluster max historical rank; test > calibration > development'},'data_files':{p.name:sha256_file(p) for p in DATA.iterdir() if p.is_file()},'split_sha256':sha256_file(LINEAGE/'splits/full.json'),'geometry_source_sha256':{name:sha256_file(ROOT/'src/robust_airfoil/geometry'/name) for name in ['parser.py','normalise.py','cst.py','validity.py','clustering.py']},'original_inputs_preserved':True,'test_limitation':'historical observations have been inspected previously; evaluation is a repaired-pipeline audit, not a claim of pristine prospective population confirmation'}
    (LINEAGE/'DATA_MANIFEST.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (REVIEW/'DATA_MANIFEST.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (REVIEW/'RAW_SOURCE_SHA256.json').write_text(json.dumps(source_hashes,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
