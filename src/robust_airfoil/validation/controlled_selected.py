"""Strict, explicit local-XFOIL verification of fixed selected geometries.

The input file is the actual displaced geometry, not a surrogate CST refit.
A numerical warm-up is not counted as recorded verification evidence. All45
requested incidences, clean exit and all fixed-domain service roots are required.
This same-solver verification is not independent RANS CFD or physical truth.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_airfoil.data.polar_parser import parse_polar_file
from robust_airfoil.hashing import hash_object, sha256_file
from robust_airfoil.optimisation.objectives import required_lift_drag
from robust_airfoil.provenance import utc_now, write_json_atomic

ALPHA_GRID = np.arange(0.,11.0001,.25)
SERVICE_LIFTS = np.asarray([.4,.7,1.])
SERVICE_WEIGHTS = np.asarray([.25,.50,.25])


def selected_xfoil_commands(panel_count: int = 160) -> str:
    """Fixed recorded protocol, with optional declared panel-refinement cases."""
    if not isinstance(panel_count,int) or not 80 <= panel_count <= 300:
        raise ValueError('Selected-study panel count must be an explicit integer within80..300')
    return '\n'.join([
        'LOAD geometry.dat', 'PPAR', f'N {panel_count}', '', '', 'PANE',
        'OPER', 'VISC 1000000', 'MACH 0', 'VPAR', 'N 9', '', 'ITER 200',
        'ASEQ -2 0 0.5',
        'PACC', 'polar.txt', '',
        'ASEQ 0 11 0.25', 'PACC', '', 'QUIT', '',
    ])


def assess_selected_polar(points: pd.DataFrame, *, return_code: int | None, timed_out: bool) -> dict[str, Any]:
    """Do not turn incomplete, conflicting or off-mission polars into success."""
    columns = ['alpha_deg','cl','cd','cm']
    result: dict[str, Any] = {
        'expected_points':len(ALPHA_GRID),'printed_points':len(points),'observed_valid_points':0,
        'invalid_printed_rows':0,
        'return_code':return_code,'timed_out':bool(timed_out),
        'missing_alpha_deg':ALPHA_GRID.tolist(),'unexpected_alpha_deg':[],
        'conflicting_alpha_deg':[],'consistent_duplicate_rows_removed':0,
        'solver_complete':False,'required_lift_feasible':False,'usable':False,
        'weighted_cd':None,'weighted_cm_at_service_lift':None,'mean_cm_over_service_alpha':None,
        'required_lift_roots':[], 'service_root_domain_deg':[0.,8.],
    }
    if not set(columns).issubset(points.columns):
        result['failure_reason'] = 'missing_coefficient_columns'
        return result
    matrix = points[columns].to_numpy(float)
    finite = np.isfinite(matrix).all(axis=1) & (matrix[:,2] > 0)
    result['invalid_printed_rows'] = int((~finite).sum())
    usable_rows = points.loc[finite,columns].copy()
    kept=[]
    conflicts=[]
    for _alpha,group in usable_rows.groupby('alpha_deg',sort=True):
        alpha=float(group.alpha_deg.iloc[0])
        values=group[['cl','cd','cm']].to_numpy(float)
        if np.any(np.ptp(values,axis=0)>1e-10):
            conflicts.append(alpha)
        else:
            kept.append(group.iloc[:1])
            result['consistent_duplicate_rows_removed'] += len(group)-1
    valid = pd.concat(kept,ignore_index=True) if kept else usable_rows.iloc[:0].copy()
    alphas = valid.alpha_deg.to_numpy(float)
    tolerance = 5.01e-4
    missing = [float(a) for a in ALPHA_GRID if not np.any(abs(alphas-a)<=tolerance)]
    unexpected = [float(a) for a in alphas if not np.any(abs(ALPHA_GRID-a)<=tolerance)]
    result.update(observed_valid_points=len(valid),missing_alpha_deg=missing,
                  unexpected_alpha_deg=unexpected,conflicting_alpha_deg=conflicts)
    complete = (return_code==0 and not timed_out and not missing and not unexpected
                and not conflicts and result['invalid_printed_rows']==0 and len(valid)==len(ALPHA_GRID))
    result['solver_complete']=bool(complete)
    if not complete:
        result['failure_reason']='abnormal_exit_timeout_incomplete_invalid_or_conflicting_sweep'
        return result
    service=valid.loc[valid.alpha_deg.between(0.,8.)].sort_values('alpha_deg')
    alpha=service.alpha_deg.to_numpy(float)
    cl,cd,cm=(service[target].to_numpy(float) for target in ['cl','cd','cm'])
    non_rising=np.flatnonzero(np.diff(cl)<=1e-12)
    initial_end=int(non_rising[0]) if len(non_rising) else len(alpha)-1
    drags=[]
    moments=[]
    roots=[]
    for required in SERVICE_LIFTS:
        drag=required_lift_drag(alpha,cl,cd,float(required))
        if not np.isfinite(drag):
            result['failure_reason']='missing_ambiguous_or_poststall_service_lift_root'
            return result
        incidence=float(np.interp(required,cl[:initial_end+1],alpha[:initial_end+1]))
        moment=float(np.interp(incidence,alpha,cm))
        roots.append({'required_cl':float(required),'alpha_deg':incidence,'cd':float(drag),'cm':moment})
        drags.append(drag)
        moments.append(moment)
    result.update(required_lift_feasible=True,usable=True,
                  weighted_cd=float(np.dot(SERVICE_WEIGHTS,drags)),
                  weighted_cm_at_service_lift=float(np.dot(SERVICE_WEIGHTS,moments)),
                  mean_cm_over_service_alpha=float(cm.mean()),
                  required_lift_roots=roots,failure_reason=None)
    return result


def run_selected_xfoil(executable: Path, geometry_path: Path, output: Path, *,
                       panel_count: int = 160, timeout_seconds: int = 30) -> dict[str, Any]:
    """Run once or verify an exact matching immutable completed case receipt."""
    if timeout_seconds < 1:
        raise ValueError('Positive bounded solver timeout is required')
    command = selected_xfoil_commands(panel_count)
    identity = {'schema':'controlled-selected-xfoil-case-v1','geometry_sha256':sha256_file(geometry_path),
                'executable_sha256':sha256_file(executable),'runner_sha256':sha256_file(Path(__file__)),
                'command':command,'timeout_seconds':timeout_seconds,'panel_count':panel_count}
    context_hash = hash_object(identity)
    receipt_path = output/'CASE_RECEIPT.json'
    if receipt_path.is_file():
        record=json.loads(receipt_path.read_text(encoding='utf-8'))
        if record['context_hash']!=context_hash:
            raise ValueError('Selected solver evidence context differs; do not overwrite')
        for relative,expected in record['files_sha256'].items():
            if sha256_file(output/relative)!=expected:
                raise ValueError('Selected solver evidence hash changed: '+relative)
        return record
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('An incomplete selected-solver attempt must be preserved in its own revision')
    output.mkdir(parents=True,exist_ok=True)
    (output/'command.txt').write_text(command,encoding='utf-8')
    shutil.copy2(geometry_path,output/'geometry.dat')
    started=time.monotonic()
    return_code: int | None=None
    timed_out=False
    points=pd.DataFrame()
    parse_error: str | None=None
    with tempfile.TemporaryDirectory(prefix='controlled-selected-xfoil-') as temporary_text:
        temporary=Path(temporary_text)
        shutil.copy2(geometry_path,temporary/'geometry.dat')
        try:
            completed=subprocess.run([str(executable.resolve())],input=command,cwd=temporary,
                                     capture_output=True,text=True,encoding='utf-8',errors='replace',
                                     timeout=timeout_seconds,check=False)
            return_code=completed.returncode
            stdout,stderr=completed.stdout,completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out=True
            def decoded(value: bytes | str | None) -> str:
                return value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value or ''
            stdout,stderr=decoded(exc.stdout),decoded(exc.stderr)
        (output/'stdout.txt').write_text(stdout,encoding='utf-8')
        (output/'stderr.txt').write_text(stderr,encoding='utf-8')
        if (temporary/'polar.txt').exists():
            shutil.copy2(temporary/'polar.txt',output/'polar.txt')
            try:
                points=parse_polar_file(output/'polar.txt').points
            except (ValueError,RuntimeError,IndexError) as exc:
                parse_error=f'{type(exc).__name__}: {exc}'
    points.to_csv(output/'parsed_points.csv',index=False)
    assessment=assess_selected_polar(points,return_code=return_code,timed_out=timed_out)
    if sha256_file(geometry_path)!=identity['geometry_sha256'] or sha256_file(output/'geometry.dat')!=identity['geometry_sha256']:
        raise ValueError('Selected geometry changed during solver execution')
    record={'context_hash':context_hash,'identity':identity,'completed_utc':utc_now(),
            'seconds':time.monotonic()-started,'parse_error':parse_error,
            'assessment':assessment,
            'scope':'Strict local-XFOIL same-method numerical check; no incomplete-sweep imputation or independent-CFD claim',
            'files_sha256':{path.relative_to(output).as_posix():sha256_file(path) for path in sorted(output.rglob('*')) if path.is_file()}}
    write_json_atomic(receipt_path,record)
    return record
