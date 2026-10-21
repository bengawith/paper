"""Generate all sealed confirmation labels only after predictor/risk freeze.

Only actual clean-process solver observations are admitted. Every requested
case, missing incidence and failure survives in a hash-bound case ledger.
No model, risk coefficient, geometry selection or threshold is fitted here.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import numpy as np
import pandas as pd
import psutil

from robust_airfoil.data.builder import admit_local_polar_rows, merge_local_polar_rows
from robust_airfoil.hashing import hash_object, sha256_file
from robust_airfoil.modelling.freeze import verify_frozen_artifacts
from robust_airfoil.optimisation.checkpointing import assert_campaign_resources
from robust_airfoil.provenance import utc_now, write_json_atomic
from robust_airfoil.research_lineage import resolve_lineage
from robust_airfoil.validation.xfoil_runner import XFoilCase, _run_xfoil_segment

ROOT=Path(__file__).resolve().parents[2]
PATHS=resolve_lineage(ROOT,'controlled')
LINEAGE=PATHS['results']
CONFIRMATION=LINEAGE/'confirmation'
REPORTS=PATHS['reports']/'CONFIRMATION'
EXE=ROOT/'tools/xfoil/xfoil.exe'
REQUESTED=np.arange(-12.,16.0001,.25)
SEGMENTS=[('positive',0.,16.,.25),('negative',0.,-12.,-.25)]


def verify_case(directory: Path, context_hash: str) -> dict[str, Any]:
    receipt=json.loads((directory/'CASE_RECEIPT.json').read_text(encoding='utf-8'))
    if receipt['context_hash']!=context_hash:
        raise ValueError('A different confirmation scientific context already occupies this case')
    for relative,expected in receipt['files_sha256'].items():
        if sha256_file(directory/relative)!=expected:
            raise ValueError('Confirmation case bytes changed: '+relative)
    return receipt


def solve(candidate: dict[str, Any], context_hash: str) -> dict[str, Any]:
    name=candidate['confirmation_id']
    output=CONFIRMATION/'solver_labels'/name
    if (output/'CASE_RECEIPT.json').exists():
        return verify_case(output,context_hash)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Partial confirmation attempt requires inspection and a separate revision: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    geometry=ROOT/candidate['geometry_path']
    if sha256_file(geometry)!=candidate['geometry_sha256']:
        raise ValueError('The sealed confirmation geometry changed before execution')
    rows=[]
    segments=[]
    started=time.monotonic()
    for label,start,end,step in SEGMENTS:
        requested=start+np.arange(round((end-start)/step)+1)*step
        case=XFoilCase(geometry,1000000,0.,9.,start,end,step,200,label)
        began=time.monotonic()
        result=_run_xfoil_segment(EXE,case,output/label,30)
        raw=result.parsed_points.copy()
        raw['source_segment']=label
        raw['source_path']=result.polar_path.relative_to(ROOT).as_posix() if result.polar_path else ''
        raw['source_sha256']=sha256_file(result.polar_path) if result.polar_path else ''
        accepted=admit_local_polar_rows(raw,requested,return_code=result.return_code,timed_out=result.timed_out)
        rows.append(accepted)
        segments.append({'segment':label,'requested_states':len(requested),'printed_states':len(raw),'admitted_states':len(accepted),
                         'full_sweep_status':result.status,'return_code':result.return_code,'timed_out':result.timed_out,
                         'failure_reason':result.failure_reason,'seconds':time.monotonic()-began})
    observed,merge_audit=merge_local_polar_rows(rows)
    observed['airfoil_id']=name
    observed['geometry_sha256']=candidate['geometry_sha256']
    observed['source_tier']='fresh_confirmation_local_xfoil_exact_cst'
    observed.to_parquet(output/'observed_points.parquet',index=False)
    receipt={'confirmation_id':name,'context_hash':context_hash,'geometry_sha256':candidate['geometry_sha256'],
             'requested_states':len(REQUESTED),'admitted_states':len(observed),'segments':segments,
             'missing_requested_alpha_deg':[float(a) for a in REQUESTED if not np.any(abs(observed.alpha_deg.to_numpy(float)-a)<=5.01e-4)],
             'complete_unique_grid':len(observed)==len(REQUESTED),'merge_audit':merge_audit,
             'seconds':time.monotonic()-started,
             'files_sha256':{path.relative_to(output).as_posix():sha256_file(path) for path in sorted(output.rglob('*')) if path.is_file()}}
    write_json_atomic(output/'CASE_RECEIPT.json',receipt)
    return receipt


def main() -> None:
    REPORTS.mkdir(parents=True,exist_ok=True)
    complete_path=CONFIRMATION/'SOLVER_LABELS_COMPLETE.json'
    if complete_path.exists():
        raise FileExistsError('Confirmation solver labels are already complete; preserve them')
    before=json.loads((CONFIRMATION/'PREDICTIONS_BEFORE_LABELS_RECEIPT.json').read_text(encoding='utf-8'))
    seal_path=CONFIRMATION/'GEOMETRY_SELECTION_FROZEN.json'
    seal=json.loads(seal_path.read_text(encoding='utf-8'))
    frozen_path=CONFIRMATION/'FROZEN_CONFIRMATION_PREDICTOR.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    verify_frozen_artifacts(frozen_path,{name:Path(path) for name,path in frozen['artifact_paths'].items()})
    if before['confirmation_labels_generated'] or before['geometry_selection_sha256']!=sha256_file(seal_path):
        raise ValueError('Predictor/geometry preparation did not precede confirmation labels')
    if before['prediction_file_sha256']!=sha256_file(CONFIRMATION/'predictions_before_labels.parquet'):
        raise ValueError('Predictions were altered after their before-label receipt')
    context={'schema':'sealed-confirmation-solver-execution-v1','geometry_selection_sha256':sha256_file(seal_path),
             'frozen_predictor_sha256':sha256_file(frozen_path),'before_labels_receipt_sha256':sha256_file(CONFIRMATION/'PREDICTIONS_BEFORE_LABELS_RECEIPT.json'),
             'exe_sha256':sha256_file(EXE),'script_sha256':sha256_file(Path(__file__)),
             'runner_sha256':sha256_file(ROOT/'src/robust_airfoil/validation/xfoil_runner.py'),
             'admission_sha256':sha256_file(ROOT/'src/robust_airfoil/data/builder.py'),
             'requested_grid':REQUESTED.tolist(),'iterations':200,'timeout_seconds':30,'workers':2,
             'conditions':{'reynolds_number':1000000,'mach':0.,'ncrit':9.}}
    context_hash=hash_object(context)
    protocol_path=CONFIRMATION/'CONFIRMATION_SOLVER_PROTOCOL.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
        if protocol['context_hash']!=context_hash:
            raise ValueError('Confirmation execution context changed; do not mix revisions')
    else:
        write_json_atomic(protocol_path,{'context':context,'context_hash':context_hash,'created_utc':utc_now()})
    if os.name=='nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    assert_campaign_resources(CONFIRMATION)
    records=[]
    remaining=[]
    for candidate in seal['candidates']:
        output=CONFIRMATION/'solver_labels'/candidate['confirmation_id']
        if (output/'CASE_RECEIPT.json').exists():
            records.append(verify_case(output,context_hash))
        else:
            remaining.append(candidate)
    iterator=iter(remaining)
    pending: set[concurrent.futures.Future] = set()
    exhausted=False
    resource_pause: str | None=None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending)<2:
                try:
                    assert_campaign_resources(CONFIRMATION)
                    candidate=next(iterator)
                    pending.add(pool.submit(solve,candidate,context_hash))
                except StopIteration:
                    exhausted=True
                except RuntimeError as exc:
                    resource_pause=str(exc)
                    exhausted=True
            if not pending:
                break
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                record=future.result()
                records.append(record)
                progress={'completed_geometries':len(records),'requested_geometries':len(seal['candidates']),
                          'observed_states_so_far':sum(r['admitted_states'] for r in records),'utc':utc_now()}
                write_json_atomic(REPORTS/'SOLVER_PROGRESS.json',progress)
                if len(records)%10==0:
                    print(json.dumps(progress),flush=True)
    if resource_pause:
        write_json_atomic(REPORTS/'RESOURCE_PAUSE.json',{'reason':resource_pause,'utc':utc_now(),'completed_geometries':len(records)})
        raise RuntimeError(resource_pause)
    if len(records)!=len(seal['candidates']):
        raise RuntimeError('Confirmation geometry accounting is incomplete')
    # Recheck before-label predictions and all model/risk artifacts after solves.
    verify_frozen_artifacts(frozen_path,{name:Path(path) for name,path in frozen['artifact_paths'].items()})
    if before['prediction_file_sha256']!=sha256_file(CONFIRMATION/'predictions_before_labels.parquet'):
        raise ValueError('Before-label predictions changed during solver execution')
    records=sorted(records,key=lambda row:row['confirmation_id'])
    frames=[pd.read_parquet(CONFIRMATION/'solver_labels'/record['confirmation_id']/'observed_points.parquet') for record in records]
    observed=pd.concat([frame for frame in frames if len(frame)],ignore_index=True)
    observed_path=CONFIRMATION/'observed_confirmation_points.parquet'
    observed.to_parquet(observed_path,index=False)
    write_json_atomic(REPORTS/'SOLVER_CASE_ACCOUNTING.json',records)
    receipt={'schema':'complete-sealed-confirmation-label-accounting-v1','completed_utc':utc_now(),
             'requested_geometries':len(records),'geometries_with_observations':int(observed.airfoil_id.nunique()),
             'requested_states':len(records)*len(REQUESTED),'observed_states':len(observed),
             'missing_requested_states':len(records)*len(REQUESTED)-len(observed),
             'complete_grid_geometries':sum(record['complete_unique_grid'] for record in records),
             'failed_full_sweep_segments':sum(segment['full_sweep_status']!='ok' for record in records for segment in record['segments']),
             'timed_out_segments':sum(segment['timed_out'] for record in records for segment in record['segments']),
             'nonzero_or_missing_exit_segments':sum(segment['return_code']!=0 for record in records for segment in record['segments']),
             'observed_points_sha256':sha256_file(observed_path),'case_accounting_sha256':sha256_file(REPORTS/'SOLVER_CASE_ACCOUNTING.json'),
             'geometry_selection_sha256':sha256_file(seal_path),'frozen_predictor_sha256':sha256_file(frozen_path),
             'solver_protocol_sha256':sha256_file(protocol_path),
             'scope':'All sealed shapes attempted; only actual clean-process finite positive-drag observations. Accuracy is conditional on solver-observed states, not the missing outcomes.'}
    write_json_atomic(complete_path,receipt)
    write_json_atomic(REPORTS/'SOLVER_LABELS_COMPLETE.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    print('ALL_SEALED_CONFIRMATION_SOLVES_COMPLETE',flush=True)


if __name__=='__main__':
    main()
