"""Read-only re-evaluation of every stored selected-design solver case and seed.

Writes only this review's derived evidence. Solver failures are not imputed;
raw-coordinate identity, physical admissibility and solver usability are distinct.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

from robust_airfoil.config import UncertaintyConfig, load_yaml
from robust_airfoil.front_agreement import front_agreement
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.modelling.evaluate import cvar
from robust_airfoil.optimisation.objectives import weighted_required_lift_drag
from robust_airfoil.uncertainty.fields import smooth_normal_perturbation
from robust_airfoil.uncertainty.sampling import sobol_normal_samples

ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path(__file__).resolve().parent
OUT = REVIEW/'RAW_XFOIL_RECOMPUTATION'
OUT.mkdir(exist_ok=True)
LIN = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908'
BASE = ROOT/'results/robust_v2/submission_review_20261014'
OLD = ROOT/'reports/robust_v2/submission_review_20261014'
UNC = UncertaintyConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/uncertainty.yaml'))
SELECTED = {r['design_id']: np.asarray(r['parameters'], float) for r in json.loads((LIN/'optimisation/selected_designs.json').read_text())}
DESIGNS = {'reference_naca2412': SELECTED['verified_reference'], 'within_trust_knee': SELECTED['trust_constrained_pareto_knee'], 'outside_trust_deterministic': SELECTED['deterministic_optimum']}
X = (1-np.cos(np.linspace(0,np.pi,UNC.cosine_points_per_surface)))/2
REQUIRED = np.asarray([.4,.7,1.])
WEIGHTS = np.asarray([.25,.5,.25])
files = {}
def sha(path):
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    files[path.relative_to(ROOT).as_posix()] = value
    return value

def case(path, expected_coordinates):
    row = {'case': path.relative_to(ROOT).as_posix(), 'exists': path.is_dir(), 'solver_usable': False, 'physical_admissible': False, 'coordinate_identity': False, 'weighted_cd': None}
    geometry = path/'geometry.dat'
    if geometry.is_file():
        raw = parse_coordinate_file(geometry)
        row['geometry_sha256'] = sha(geometry)
        row['coordinate_identity'] = raw.shape == expected_coordinates.shape and bool(np.allclose(raw,expected_coordinates,atol=1e-11,rtol=0))
        row['maximum_coordinate_difference'] = float(np.max(np.abs(raw-expected_coordinates))) if raw.shape == expected_coordinates.shape else None
        mid = len(X)-1
        physical = validate_geometry(raw[:mid+1][::-1],raw[mid:])
        row['physical_admissible'] = physical.valid
        row['geometry_reasons'] = list(physical.reasons)
    statusfile = path/'run_status.json'
    pointsfile = path/'combined_points.csv'
    commandfile = path/'command.txt'
    if not statusfile.is_file():
        row['reason'] = 'no_stored_solver_status'
        return row
    status = json.loads(statusfile.read_text())
    row['status_sha256'] = sha(statusfile)
    row['stored_status'] = status.get('status')
    row['stored_failure'] = status.get('failure_reason')
    clean = status.get('status') == 'ok' and status.get('return_code') == 0 and not status.get('timed_out',False) and status.get('clean_exit',False)
    row['clean_exit'] = bool(clean)
    if not pointsfile.is_file() or not commandfile.is_file():
        row['reason'] = 'missing_polar_or_command'
        return row
    row['polar_sha256'] = sha(pointsfile)
    row['command_sha256'] = sha(commandfile)
    command = commandfile.read_text()
    row['conditions_match'] = all(s in command for s in ['VISC 1000000','MACH 0','N 9'])
    polar = pd.read_csv(pointsfile)
    matrix = polar[['alpha_deg','cl','cd']].to_numpy(float)
    ordered = matrix[np.argsort(matrix[:,0])]
    row['actual_rows'] = len(ordered)
    row['actual_alpha_complete'] = len(ordered)==12 and bool(np.allclose(ordered[:,0],np.arange(12),rtol=0,atol=1e-10))
    row['finite_positive_drag'] = bool(np.isfinite(matrix).all() and np.all(matrix[:,2]>0))
    row['stored_and_actual_coverage_agree'] = bool(status.get('span_complete')) == row['actual_alpha_complete']
    if clean and row['actual_alpha_complete'] and row['finite_positive_drag'] and row['conditions_match']:
        value = weighted_required_lift_drag(ordered[:,0],ordered[:,1],ordered[:,2],REQUIRED,WEIGHTS)
        if np.isfinite(value):
            row['weighted_cd'] = float(value)
            row['solver_usable'] = True
            row['reason'] = 'complete_and_required_lift_feasible'
        else:
            row['reason'] = 'required_lift_not_unique_on_initial_rising_branch'
    else:
        row['reason'] = 'solver_exit_coverage_or_numeric_failure'
    return row

campaigns = []
all_cases = []
for name in ['xfoil_strict','xfoil_eps005_n128','xfoil_eps005_n128_b']:
    directory = BASE/name
    summary_path = directory/'design_verification_summary.json'
    original = json.loads(summary_path.read_text())
    sha(summary_path)
    n = original['shared_perturbations']
    epsilon = original['manufacturing_amplitude_fraction_chord']
    offset = original.get('sobol_scramble_seed_offset',50000)
    latents = sobol_normal_samples(n,2*UNC.basis_control_points_per_surface,UNC.seed+offset,antithetic=True)
    by_design = {}
    for design,parameters in DESIGNS.items():
        uy,ly = reconstruct_cst(parameters,X)
        upper,lower = np.column_stack([X,uy]),np.column_stack([X,ly])
        nominal = case(directory/design/'nominal',np.vstack([upper[::-1],lower[1:]]))
        nominal.update(campaign=name,design=design,scenario='nominal')
        all_cases.append(nominal)
        rows = []
        for i,latent in enumerate(latents):
            pert = smooth_normal_perturbation(upper,lower,latent[:UNC.basis_control_points_per_surface],latent[UNC.basis_control_points_per_surface:],epsilon,UNC.correlation_length_chord,UNC.leading_edge_taper,UNC.trailing_edge_zero_displacement)
            row = case(directory/design/f'shared_{i:02d}',np.vstack([pert.upper[::-1],pert.lower[1:]]))
            row.update(campaign=name,design=design,scenario=i)
            rows.append(row)
            all_cases.append(row)
        values = np.asarray([r['weighted_cd'] for r in rows if r['solver_usable']],float)
        by_design[design] = {'nominal':nominal,'requested':n,'usable':len(values),'physical_admissible_usable':sum(r['solver_usable'] and r['physical_admissible'] for r in rows),'coordinate_identity_failures':sum(r['exists'] and not r['coordinate_identity'] for r in rows),'mean':float(values.mean()) if len(values) else None,'cvar95':cvar(values,.95) if len(values) else None,'rows':rows}
    ref = by_design['reference_naca2412']
    cand = by_design['within_trust_knee']
    pair = [(a['weighted_cd'],b['weighted_cd']) for a,b in zip(ref['rows'],cand['rows'],strict=True) if a['solver_usable'] and b['solver_usable']]
    rp,cp = np.asarray(pair,float).T
    campaign = {'campaign':name,'requested':n,'epsilon':epsilon,'scramble_seed_offset':offset,'nominal_reduction_fraction':1-cand['nominal']['weighted_cd']/ref['nominal']['weighted_cd'],'nominal_outside_increase_fraction':by_design['outside_trust_deterministic']['nominal']['weighted_cd']/ref['nominal']['weighted_cd']-1,'designs':{k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in by_design.items()},'paired':{'n':len(pair),'reference_mean':float(rp.mean()),'candidate_mean':float(cp.mean()),'difference':float((rp-cp).mean()),'reduction_fraction':float((rp-cp).mean()/rp.mean()),'reference_cvar95':cvar(rp,.95),'candidate_cvar95':cvar(cp,.95),'cvar_change_fraction':cvar(cp,.95)/cvar(rp,.95)-1,'physical_admissible_pairs':sum(a['solver_usable'] and b['solver_usable'] and a['physical_admissible'] and b['physical_admissible'] for a,b in zip(ref['rows'],cand['rows'],strict=True))}}
    campaigns.append(campaign)
    print(json.dumps({'campaign':name,'pairs':campaign['paired'],'usable':{k:v['usable'] for k,v in by_design.items()}},indent=2),flush=True)
pd.DataFrame(all_cases).to_csv(OUT/'all_825_cases.csv',index=False)
(OUT/'XFOIL_RECALCULATED.json').write_text(json.dumps({'campaigns':campaigns,'cases':len(all_cases),'scope':'Same-solver, fixed stored coordinates; no independent CFD, unconditional population CVaR or fresh solver campaign.'},indent=2,allow_nan=False),encoding='utf-8')

cfd_rows = []
for manifest_path in [OLD/'CFD_HANDOFF/CFD_CASE_MANIFEST.json',OLD/'CFD_HANDOFF/OPTIONAL_0P5_PERCENT/OPTIONAL_CFD_MANIFEST.json']:
    payload = json.loads(manifest_path.read_text())
    for row in payload.get('cases',[]):
        if manifest_path.name.startswith('OPTIONAL'):
            print('OPTIONAL_MANIFEST_SCHEMA',json.dumps(row)[:500],flush=True)
            continue
        for filename,expected in row['files'].items():
            path = manifest_path.parent/'inputs'/row['design']/row['scenario']/filename
            cfd_rows.append({'path':path.relative_to(ROOT).as_posix(),'expected_sha256':expected,'actual_sha256':sha(path),'matches':sha(path)==expected})
(OUT/'CFD_MAIN_INPUT_HASH_AUDIT.json').write_text(json.dumps(cfd_rows,indent=2),encoding='utf-8')

front_root = OLD/'CORRECTED_OPTIMISATION'
front_manifest = json.loads((front_root/'CORRECTED_SMOKE_FRONT.json').read_text())
fronts = []
seeds = []
for path in sorted((front_root/'smoke_seeds').glob('*.json')):
    d = json.loads(path.read_text())
    objectives = np.asarray(d['objectives'],float)
    constraints = np.asarray(d['constraint_values'],float)
    feasible = np.all(constraints<=0,axis=1)
    fronts.append(objectives[feasible])
    seeds.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'seed':d['seed'],'feasible':int(feasible.sum()),'schema':d['schema_version'],'history_generations':len(d['history'])})
scales = np.asarray(front_manifest['agreement']['objective_agreement']['objective_scales'])
recalculated = front_agreement(fronts,scales,.2,.8)
recalculated.update(seeds=seeds,scale_context='POST-HOC pooled objective range, not a prospective fixed scale',historical_scope='Schema-4 smoke only; not current schema/source acceptance',original_distance=front_manifest['agreement']['pairwise_objective_distance_mean'])
(OUT/'HISTORICAL_FRONT_RECALCULATED.json').write_text(json.dumps(recalculated,indent=2),encoding='utf-8')
(OUT/'READ_SOURCE_SHA256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print('RAW_AUDIT_FINISHED',len(all_cases),'cases;',sum(r['matches'] for r in cfd_rows),'main CFD files;',recalculated['mean_symmetric_distance'],'historical front distance',flush=True)
