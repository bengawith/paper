"""Audit all frozen geometry encodings against repaired coordinate-order handling.

Writes a separate repair evidence table. It never overwrites the frozen dataset,
changes the train/test assignment, trains a model, or re-labels solver data.
"""
from pathlib import Path
import hashlib
import json
import re
import time
import numpy as np
import pandas as pd
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.geometry.normalise import normalise_geometry
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent/'GEOMETRY_REPAIR'
OUT.mkdir(exist_ok=True)
DATA = ROOT/'data/robust_v2/processed/full_exact'
airfoils = pd.read_parquet(DATA/'airfoils.parquet')
PARAM = FEATURE_COLUMNS[:-1]
rows = []
repaired = []
start = time.monotonic()
for _,old in airfoils.iterrows():
    path = Path(old.geometry_path)
    raw = parse_coordinate_file(path)
    minimum = int(np.argmin(raw[:,0]))
    span = float(np.ptp(raw[:,0]))
    jumps = np.flatnonzero(np.diff(raw[:,0]) < -.5*span)
    order = 'two_LE_to_TE_blocks' if len(jumps)==1 and 2<=jumps[0]<=len(raw)-4 else 'TE_LE_TE_contour'
    row = {'airfoil_id':str(old.airfoil_id),'raw_path':path.relative_to(ROOT).as_posix(),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_points':len(raw),'detected_order':order,'old_endpoint_median_fallback':minimum in (0,len(raw)-1),'old_cst_rms_error':float(old.cst_rms_error),'old_cst_max_error':float(old.cst_max_error)}
    try:
        geometry = normalise_geometry(raw)
        validity = validate_geometry(geometry.upper,geometry.lower)
        fit = fit_cst(geometry.upper,geometry.lower)
        upper_fit,lower_fit = reconstruct_cst(fit.parameters,geometry.upper[:,0])
        fit_validity = validate_geometry(np.column_stack([geometry.upper[:,0],upper_fit]),np.column_stack([geometry.lower[:,0],lower_fit]))
        difference = float(np.max(np.abs(fit.parameters-old[PARAM].to_numpy(float))))
        row.update(status='recomputed',maximum_parameter_change=difference,changed=bool(difference>1e-10),new_cst_rms_error=float(fit.rms_error),new_cst_max_error=float(fit.max_error),physical_geometry_valid=validity.valid,physical_geometry_reasons=';'.join(validity.reasons),refitted_geometry_valid=fit_validity.valid,refitted_geometry_reasons=';'.join(fit_validity.reasons))
        repaired.append({'airfoil_id':str(old.airfoil_id),**dict(zip(PARAM,fit.parameters,strict=True)),'cst_rms_error':fit.rms_error,'cst_max_error':fit.max_error,'geometry_valid':validity.valid,'cst_geometry_valid':fit_validity.valid,'original_cluster_id':int(old.cluster_id)})
    except (ValueError,RuntimeError,IndexError) as exc:
        row.update(status='rejected_ambiguous_or_invalid_order',error=str(exc),changed=True)
    rows.append(row)
frame = pd.DataFrame(rows)
frame.to_csv(OUT/'all_airfoils.csv',index=False)
pd.DataFrame(repaired).to_parquet(OUT/'repaired_features_RETRAINING_INPUT_NOT_FROZEN_MODEL.parquet',index=False)
valid = frame.loc[frame.status=='recomputed']
changed = valid.loc[valid.changed]
summary = {'frozen_airfoils':len(frame),'raw_coordinate_order_counts':frame.detected_order.value_counts().to_dict(),'old_median_fallback_count':int(frame.old_endpoint_median_fallback.sum()),'recomputed':len(valid),'rejected':len(frame)-len(valid),'changed_features':int(valid.changed.sum()),'changed_by_order':changed.detected_order.value_counts().to_dict(),'old_max_error_above_0p5pct':int((valid.old_cst_max_error>.005).sum()),'corrected_max_error_above_0p5pct':int((valid.new_cst_max_error>.005).sum()),'corrected_physical_invalid':int((~valid.physical_geometry_valid.astype(bool)).sum()),'corrected_cst_invalid':int((~valid.refitted_geometry_valid.astype(bool)).sum()),'old_fit_quantiles':valid.old_cst_max_error.quantile([.5,.9,.95,1]).to_dict(),'corrected_fit_quantiles':valid.new_cst_max_error.quantile([.5,.9,.95,1]).to_dict(),'largest_changes':json.loads(changed.nlargest(10,'maximum_parameter_change').to_json(orient='records')),'naca2412':json.loads(valid.loc[valid.airfoil_id=='naca2412'].to_json(orient='records')),'seconds':time.monotonic()-start,'scientific_consequence':'Frozen prediction metrics reproduce the historical encoding, not correctness of that encoding. A repaired geometry training/split/calibration/model lineage is needed before claiming a physically faithful trained surrogate or launching publication-grade optimisation. Existing direct-XFOIL results remain numerical evidence for their exact stored coordinates, not validation of all training encodings.','frozen_inputs_rewritten':False}
(OUT/'GEOMETRY_AUDIT.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['largest_changes','naca2412']},indent=2),flush=True)
