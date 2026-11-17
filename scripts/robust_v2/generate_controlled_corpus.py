"""Generate genuine local-XFOIL labels on the exact represented CST geometries.

Two bounded workers, immutable per-geometry receipts, exact-context resume, no
imputation or archived-label frame assumptions. Partial solver coverage is
reported explicitly; it is not a completed validation sweep.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import pandas as pd
import psutil

from robust_airfoil.constants import FEATURE_COLUMNS
from robust_airfoil.data.builder import admit_local_polar_rows, merge_local_polar_rows
from robust_airfoil.data.splits import protect_rebuilt_splits, write_split_manifest
from robust_airfoil.geometry.clustering import cluster_coordinate_vectors
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import hash_object, sha256_file
from robust_airfoil.optimisation.checkpointing import assert_campaign_resources
from robust_airfoil.provenance import utc_now, write_json_atomic
from robust_airfoil.validation.xfoil_runner import XFoilCase, _run_xfoil_segment

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
DATA = ROOT/'data/robust_v2/processed/controlled_20261020'
REVIEW = ROOT/'reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL'
SHAPE_SOURCE = ROOT/'data/robust_v2/processed/rebuilt_20261014/airfoils.parquet'
OLD_SPLIT = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908/splits/full.json'
EXE = ROOT/'tools/xfoil/xfoil.exe'
X = (1-np.cos(np.linspace(0,np.pi,201)))/2
REQUESTED = np.arange(-12.,16.0001,.25)
SEGMENTS = [('positive',0.,16.,.25),('negative',0.,-12.,-.25)]


def context(execution_revision: int = 1) -> dict[str, Any]:
    source_files = ['constants.py','geometry/cst.py','geometry/parser.py','geometry/validity.py','geometry/clustering.py',
                    'data/builder.py','data/polar_parser.py','data/splits.py','validation/xfoil_runner.py']
    identity = {'schema':'exact-cst-local-xfoil-corpus-v1','geometry_source_sha256':sha256_file(SHAPE_SOURCE),
            'protected_source_split_sha256':sha256_file(OLD_SPLIT),'executable_sha256':sha256_file(EXE),
            'generator_sha256':sha256_file(Path(__file__)),
            'scientific_source_sha256':{name:sha256_file(ROOT/'src/robust_airfoil'/name) for name in source_files},
            'conditions':{'reynolds_number':1000000,'mach':0.,'ncrit':9.,'iterations':200},
            'coordinate_points_per_surface':201,'requested_alpha_deg':REQUESTED.tolist(),
            'directed_segments':[{'name':name,'start':a,'end':b,'step':step} for name,a,b,step in SEGMENTS],
            'timeout_seconds_per_segment':30,'maximum_solver_workers':2,
            'admission':'Only actual finite positive-drag CL/CD/CM rows from clean zero-exit non-timeout segments; missing rows never filled',
            'geometry_domain':'All1256 previously admitted12-parameter CST reconstructions, not claimed identical to original source coordinates'}
    identity['execution_revision'] = execution_revision
    if execution_revision == 2:
        proof_path = ROOT/'reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/RUNTIME_MEMORY_REPAIR/COMPATIBILITY_PROOF.json'
        identity['import_only_compatibility_proof_sha256'] = sha256_file(proof_path)
    return identity


def verified_compatibility() -> dict[str, Any]:
    proof_path = ROOT/'reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/RUNTIME_MEMORY_REPAIR/COMPATIBILITY_PROOF.json'
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    parent_path = LINEAGE/'LABEL_PROTOCOL.json'
    if sha256_file(parent_path) != proof['parent_protocol_sha256']:
        raise ValueError('The preserved parent label protocol changed')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    if proof['parent_context_hash'] != parent['context_hash']:
        raise ValueError('Wrong parent label context')
    tree = ast.parse((ROOT/'src/robust_airfoil/validation/xfoil_runner.py').read_text(encoding='utf-8'))
    actual = {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
              for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
              and node.name in proof['pure_solver_ast_sha256']}
    if actual != proof['pure_solver_ast_sha256']:
        raise ValueError('Pure solver calculation changed; old cases cannot be adopted')
    tree = ast.parse((ROOT/'src/robust_airfoil/constants.py').read_text(encoding='utf-8'))
    features = {node.targets[0].id:hashlib.sha256(ast.dump(node.value,include_attributes=False).encode()).hexdigest()
                for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id in proof['feature_contract_ast_sha256']}
    if features != proof['feature_contract_ast_sha256']:
        raise ValueError('Input feature order changed')
    for name,expected in parent['context']['scientific_source_sha256'].items():
        if name != 'validation/xfoil_runner.py' and sha256_file(ROOT/'src/robust_airfoil'/name) != expected:
            raise ValueError('Unapproved label calculation change: '+name)
    return proof


def prepare_shapes(protocol: dict[str, Any]) -> pd.DataFrame:
    shapes = pd.read_parquet(SHAPE_SOURCE)
    if len(shapes) != 1256:
        raise ValueError('The declared geometry cohort changed')
    vectors = []
    dense = (1-np.cos(np.linspace(0,np.pi,2001)))/2
    for _, row in shapes.iterrows():
        parameters = row[FEATURE_COLUMNS[:-1]].to_numpy(float)
        top,bottom = reconstruct_cst(parameters,dense)
        if not validate_geometry(np.c_[dense,top],np.c_[dense,bottom]).valid:
            raise ValueError('Previously admitted CST geometry fails current dense validity: '+str(row.airfoil_id))
        top,bottom = reconstruct_cst(parameters,X)
        vectors.append(np.r_[top,bottom])
    shapes['cluster_id'] = cluster_coordinate_vectors(np.asarray(vectors),rms_threshold=1e-4)
    old = pd.DataFrame(json.loads(OLD_SPLIT.read_text(encoding='utf-8'))['records'])
    splits = protect_rebuilt_splits(shapes,old)
    shapes['split'] = shapes.airfoil_id.map(splits.set_index('airfoil_id')['split'])
    existing = LINEAGE/'shapes_and_partitions.parquet'
    if existing.exists():
        recorded = pd.read_parquet(existing)
        pd.testing.assert_frame_equal(shapes,recorded)
    else:
        shapes.to_parquet(existing,index=False)
        write_split_manifest(splits,LINEAGE/'splits/full.json',seed=2026101400)
        write_json_atomic(LINEAGE/'GEOMETRY_COHORT.json',{
            'context_hash':protocol['context_hash'],'airfoils':len(shapes),
            'clusters':int(shapes.cluster_id.nunique()),'shape_file_sha256':sha256_file(existing),
            'split_file_sha256':sha256_file(LINEAGE/'splits/full.json'),
            'selection':'Geometry-only cohort/cluster/protected partition fixed before new aerodynamic labels',
            'partition_counts':[{ 'split':name,'airfoils':len(part),'clusters':int(part.cluster_id.nunique())} for name,part in shapes.groupby('split')],
        })
    return shapes


def verify_receipt(directory: Path, expected_context: str, compatibility: dict[str, Any] | None = None) -> dict[str, Any]:
    record = json.loads((directory/'CASE_RECEIPT.json').read_text(encoding='utf-8'))
    if record['context_hash'] != expected_context and (
        compatibility is None or record['context_hash'] != compatibility['parent_context_hash']
        or compatibility['compatible_existing_receipts_sha256'].get(directory.name) != sha256_file(directory/'CASE_RECEIPT.json')
    ):
        raise ValueError('Stored solver case belongs to an unverified different scientific context')
    for relative,expected in record['files_sha256'].items():
        path=directory/relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError('Stored solver case changed: '+str(path))
    return record


def solve_shape(row: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    name=str(row['airfoil_id'])
    directory=LINEAGE/'solver_labels'/name
    receipt=directory/'CASE_RECEIPT.json'
    if receipt.is_file():
        return verify_receipt(directory,protocol['context_hash'],protocol.get('compatibility'))
    if directory.exists() and any(directory.iterdir()):
        # Incomplete outputs are evidence, not permission to overwrite logs.
        raise FileExistsError('Interrupted case needs an explicit new attempt directory: '+str(directory))
    directory.mkdir(parents=True,exist_ok=True)
    parameters=np.asarray([row[column] for column in FEATURE_COLUMNS[:-1]],float)
    top,bottom=reconstruct_cst(parameters,X)
    coords=np.vstack([np.c_[X,top][::-1],np.c_[X,bottom][1:]])
    geometry=directory/'geometry.dat'
    geometry.write_text('_controlled_cst_'+name+'\n'+'\n'.join(f'{x:.12f} {y:.12f}' for x,y in coords)+'\n',encoding='utf-8')
    np.testing.assert_allclose(parse_coordinate_file(geometry),coords,rtol=0,atol=5.1e-13)
    observed=[]
    segments=[]
    started=time.monotonic()
    for label,start,end,step in SEGMENTS:
        case=XFoilCase(geometry,1e6,0.,9.,start,end,step,200,label)
        output=directory/label
        began=time.monotonic()
        result=_run_xfoil_segment(EXE,case,output,30)
        requested=start+np.arange(round((end-start)/step)+1)*step
        raw=result.parsed_points.copy()
        raw['source_segment']=label
        raw['source_path']=result.polar_path.relative_to(ROOT).as_posix() if result.polar_path else ''
        raw['source_sha256']=sha256_file(result.polar_path) if result.polar_path else ''
        admitted=admit_local_polar_rows(raw,requested,return_code=result.return_code,timed_out=result.timed_out)
        observed.append(admitted)
        segments.append({'segment':label,'requested_states':len(requested),'printed_states':len(raw),'admitted_states':len(admitted),
                         'full_sweep_status':result.status,'return_code':result.return_code,'timed_out':result.timed_out,
                         'failure_reason':result.failure_reason,'seconds':time.monotonic()-began})
    merged,audit=merge_local_polar_rows(observed)
    merged['airfoil_id']=name
    merged['polar_id']=name+'_controlled_cst_v1'
    merged['source_tier']='local_xfoil_699_exact_cst'
    merged['reynolds_number']=1000000.
    merged['mach']=0.
    merged['ncrit']=9.
    merged['geometry_sha256']=sha256_file(geometry)
    for i,column in enumerate(FEATURE_COLUMNS[:-1]):
        merged[column]=parameters[i]
    merged['cluster_id']=int(row['cluster_id'])
    merged['split']=str(row['split'])
    merged['airfoil_weight']=1./len(merged) if len(merged) else 0.
    merged.to_parquet(directory/'observed_points.parquet',index=False)
    record={'airfoil_id':name,'context_hash':protocol['context_hash'],'geometry_sha256':sha256_file(geometry),
            'requested_unique_states':len(REQUESTED),'admitted_states':len(merged),
            'missing_requested_alpha_deg':[float(a) for a in REQUESTED if not np.any(abs(merged.alpha_deg.to_numpy(float)-a)<=5.01e-4)],
            'segments':segments,'merge_audit':audit,'seconds':time.monotonic()-started,
            'complete_unique_grid':len(merged)==len(REQUESTED),
            'files_sha256':{p.relative_to(directory).as_posix():sha256_file(p) for p in sorted(directory.rglob('*')) if p.is_file()},
            'interpretation':'Observed clean-process numerical labels; incomplete polars remain incomplete, no imputation'}
    write_json_atomic(receipt,record)
    return record


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--execution-revision',type=int,choices=[1,2],default=1)
    args=parser.parse_args()
    for directory in (LINEAGE,DATA,REVIEW):
        directory.mkdir(parents=True,exist_ok=True)
    if (LINEAGE/'DATA_MANIFEST.json').exists():
        raise FileExistsError('Controlled data is complete and frozen; no regeneration in place')
    assert_campaign_resources(LINEAGE)
    if os.name=='nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    compatibility=verified_compatibility() if args.execution_revision == 2 else None
    current=context(args.execution_revision)
    protocol_path=LINEAGE/('LABEL_PROTOCOL_R2.json' if args.execution_revision == 2 else 'LABEL_PROTOCOL.json')
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
        if protocol['context_hash'] != hash_object(current):
            raise ValueError('Changed label-generation context; preserve this revision')
    else:
        protocol={'context':current,'context_hash':hash_object(current),'created_utc':utc_now(),'compatibility':compatibility,
                  'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'working_source_identity':'Bound by explicit source hashes; base commit alone does not identify the uncommitted review.'}
        write_json_atomic(protocol_path,protocol)
    shapes=prepare_shapes(protocol)
    records=[]
    todo=[]
    for row in shapes.to_dict(orient='records'):
        case=LINEAGE/'solver_labels'/str(row['airfoil_id'])
        if (case/'CASE_RECEIPT.json').is_file():
            records.append(verify_receipt(case,protocol['context_hash'],protocol.get('compatibility')))
        else:
            todo.append(row)
    started=time.monotonic()
    iterator=iter(todo)
    pending: set[concurrent.futures.Future] = set()
    exhausted=False
    blocked: str | None=None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or not exhausted:
            while not exhausted and blocked is None and len(pending)<args.workers:
                try:
                    assert_campaign_resources(LINEAGE)
                    row=next(iterator)
                    pending.add(pool.submit(solve_shape,row,protocol))
                except StopIteration:
                    exhausted=True
                except RuntimeError as exc:
                    blocked=str(exc)
                    exhausted=True
            if not pending:
                break
            finished,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                record=future.result()
                records.append(record)
                progress={'status':'RUNNING' if len(records)<len(shapes) else 'ALL_GEOMETRIES_ATTEMPTED',
                          'completed_geometries':len(records),'requested_geometries':len(shapes),
                          'observed_states_so_far':sum(r['admitted_states'] for r in records),
                          'last_airfoil':record['airfoil_id'],'last_observed_states':record['admitted_states'],
                          'elapsed_invocation_seconds':time.monotonic()-started,'utc':utc_now()}
                write_json_atomic(REVIEW/'LABEL_PROGRESS.json',progress)
                if len(records)%10==0 or len(records)==len(shapes):
                    print(json.dumps(progress),flush=True)
    if blocked:
        write_json_atomic(REVIEW/'LABEL_PAUSED.json',{'reason':blocked,'completed_geometries':len(records),'utc':utc_now()})
        raise RuntimeError(blocked)
    if len(records)!=len(shapes) or hash_object(context(args.execution_revision))!=protocol['context_hash']:
        raise RuntimeError('Incomplete geometry accounting or source changed during label generation')
    records=sorted(records,key=lambda r:r['airfoil_id'])
    parts=[pd.read_parquet(LINEAGE/'solver_labels'/name/'observed_points.parquet') for name in shapes.airfoil_id]
    points=pd.concat([part for part in parts if len(part)],ignore_index=True)
    shapes.to_parquet(DATA/'airfoils.parquet',index=False)
    points.to_parquet(DATA/'model_points.parquet',index=False)
    write_json_atomic(REVIEW/'SOLVER_CASE_ACCOUNTING.json',records)
    summary={'schema':'controlled-cst-genuine-xfoil-data-v1','lineage':'robust-v2-controlled-20261020',
             'created_utc':utc_now(),'requested_geometries':len(shapes),'requested_unique_states':len(shapes)*len(REQUESTED),
             'observed_states':len(points),'geometries_with_observations':int(points.airfoil_id.nunique()),
             'complete_grid_geometries':sum(r['complete_unique_grid'] for r in records),
             'missing_requested_states':len(shapes)*len(REQUESTED)-len(points),
             'timed_out_segments':sum(s['timed_out'] for r in records for s in r['segments']),
             'nonzero_or_missing_exit_segments':sum(s['return_code']!=0 for r in records for s in r['segments']),
             'full_sweep_failed_segments':sum(s['full_sweep_status']!='ok' for r in records for s in r['segments']),
             'split_counts':[{'split':name,'airfoils':int(part.airfoil_id.nunique()),'clusters':int(part.cluster_id.nunique()),'states':len(part)} for name,part in points.groupby('split')],
             'data_files':{p.name:sha256_file(p) for p in DATA.iterdir() if p.is_file()},
             'label_protocol_file':protocol_path.name,'execution_revision':args.execution_revision,'compatible_prior_case_count':len(compatibility['compatible_existing_receipts_sha256']) if compatibility else 0,
             'label_protocol_sha256':sha256_file(protocol_path),'cohort_sha256':sha256_file(LINEAGE/'GEOMETRY_COHORT.json'),
             'split_sha256':sha256_file(LINEAGE/'splits/full.json'),'case_accounting_sha256':sha256_file(REVIEW/'SOLVER_CASE_ACCOUNTING.json'),
             'source_geometry_definition':'Exact recorded CST reconstructions, not unmodified UIUC coordinate contours',
             'labels':'Actual localXFOIL6.99 observations on exactly these coordinates; archived aerodynamic labels not used',
             'test_scope':'Protected historical geometry identities; not pristine prospective population sampling',
             'scientific_limits':['XFOIL is a numerical viscous/inviscid model, not independent RANS or experiment.',
                                  'Convergence selects available supervised observations; no prediction accuracy outside observed data is established.']}
    write_json_atomic(LINEAGE/'DATA_MANIFEST.json',summary)
    write_json_atomic(REVIEW/'DATA_MANIFEST.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    print('CONTROLLED_CORPUS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
