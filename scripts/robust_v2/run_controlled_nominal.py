"""Run the full matched-budget nominal-only control on the frozen controlled model.

Requires the completed robust production profile and uses exactly its reference,
bounds, physical model, service grid and nominal feasibility constraints. Only
nominal weighted drag is optimised; no verification-XFOIL outcome is consulted.
"""
from __future__ import annotations

import json
import os
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import numpy as np
import pandas as pd
import torch

from robust_airfoil.config import OptimisationConfig, load_yaml
from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.modelling.freeze import verify_frozen_artifacts
from robust_airfoil.optimisation.checkpointing import assert_campaign_resources
from robust_airfoil.optimisation.nominal_control import run_nominal_control_profile
from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.optimisation.study import RobustCandidateEvaluator
from robust_airfoil.provenance import utc_now, write_json_atomic
from robust_airfoil.research_lineage import resolve_lineage

ROOT=Path(__file__).resolve().parents[2]


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    paths=resolve_lineage(ROOT,'controlled')
    lineage,reports=paths['results'],paths['reports']
    output=lineage/'nominal_control'
    output.mkdir(parents=True,exist_ok=True)
    assert_campaign_resources(output)
    robust_protocol_path=lineage/'optimisation_production/PRODUCTION_PROTOCOL.json'
    robust_summary_path=lineage/'optimisation_production/PRODUCTION_SUMMARY.json'
    robust_protocol=json.loads(robust_protocol_path.read_text(encoding='utf-8'))
    robust_summary=json.loads(robust_summary_path.read_text(encoding='utf-8'))
    if robust_summary['status']!='FULL_PROFILE_COMPLETED' or len(robust_summary['per_seed'])!=10:
        raise ValueError('The complete declared robust campaign is required before selecting its matched control')
    source_root=ROOT/'src/robust_airfoil'
    sources={path.relative_to(source_root).as_posix():sha256_file(path) for path in source_root.rglob('*.py')}
    if sources!=robust_protocol['context']['source_sha256']:
        raise ValueError('Scientific source differs from the robust campaign; refuse unmatched control')
    study_path=ROOT/'configs/robust_v2/controlled_study.yaml'
    study=load_yaml(study_path)
    seal=json.loads((lineage/'confirmation/GEOMETRY_SELECTION_FROZEN.json').read_text(encoding='utf-8'))
    if sha256_file(study_path)!=seal['declared_study_config_sha256']:
        raise ValueError('Predeclared controlled study changed after confirmation selection')
    requested=study['nominal_control']
    if (requested['population'],requested['generations'],requested['independent_seeds'])!=(128,200,10):
        raise ValueError('The declared nominal-control budget has changed')
    frozen_path=lineage/'FROZEN_MODEL_MANIFEST.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    verify_frozen_artifacts(frozen_path,{name:Path(path) for name,path in frozen['artifact_paths'].items()})
    context=robust_protocol['context']
    if sha256_file(frozen_path)!=context['frozen_model_manifest_sha256']:
        raise ValueError('Nominal control would use a different model from robust production')
    opt=OptimisationConfig.model_validate(context['optimisation'])
    bounds=DesignBounds(np.asarray(context['bounds']['lower'],float),np.asarray(context['bounds']['upper'],float))
    reference=np.asarray(context['reference_parameters'],float)
    development=pd.read_parquet(lineage/'partition_inputs/development.parquet')
    if not development['split'].eq('development').all():
        raise ValueError('Only development geometry may seed the control population')
    observed=development[FEATURE_COLUMNS[:-1]].drop_duplicates().to_numpy(float)
    ensemble=LoadedEnsemble(lineage/'full_ensemble')
    calibration=json.loads((lineage/'calibration/trust_calibration.json').read_text(encoding='utf-8'))
    # This evaluator is used solely through its nominal constraints/predictions.
    # No manufacturing placeholder is included in the control's objective.
    evaluator=RobustCandidateEvaluator(reference,np.zeros((2,12)),np.zeros(2,bool),development,ensemble,calibration,opt)
    scientific_context={'robust_protocol_sha256':sha256_file(robust_protocol_path),
                        'robust_summary_sha256':sha256_file(robust_summary_path),
                        'frozen_model_sha256':sha256_file(frozen_path),
                        'controlled_study_sha256':sha256_file(study_path),
                        'entrypoint_sha256':sha256_file(Path(__file__)),
                        'source_sha256':sources,
                        'optimisation':opt.model_dump(mode='json'),
                        'versions':{name:version(name) for name in ['numpy','scipy','pandas','torch','pymoo']},
                        'algorithm':'NSGA-II with one nominal weighted-drag objective; same default evolutionary operators as robust production',
                        'control_scope':'Same reference, geometry bounds, service-lift grid and nominal constraints, with128x200x10 candidate-evaluation budget; not equal total uncertain-state compute.'}
    seeds=[int(requested['first_seed'])+index for index in range(10)]
    began=time.monotonic()
    result=run_nominal_control_profile(evaluator,bounds,reference,observed,seeds,128,200,output,scientific_context)
    after={path.relative_to(source_root).as_posix():sha256_file(path) for path in source_root.rglob('*.py')}
    if after!=sources:
        raise RuntimeError('Scientific source changed during matched nominal campaign')
    if len(result['runs'])!=10 or any(len(run['history'])!=200 for run in result['runs']):
        raise AssertionError('Nominal control has incomplete seed/generation evidence')
    summary={'schema':'controlled-full-matched-nominal-control-v1','status':'FULL_NOMINAL_PROFILE_COMPLETED',
             'completed_utc':utc_now(),'profile':{'population':128,'generations':200,'independent_seeds':10},
             'context_hash':result['context_hash'],'invocation_seconds':time.monotonic()-began,
             'per_seed':[{'seed':run['seed'],'generations':len(run['history']),
                          'feasible_entries':run['feasible_count'],'minimum_feasible_nominal_drag':run['minimum_feasible_nominal_drag'],
                          'final_recheck_max_objective_difference':run['final_recheck_max_objective_difference'],
                          'file':f"nominal_seed_{run['seed']}.json",'sha256':sha256_file(output/f"nominal_seed_{run['seed']}.json")}
                         for run in result['runs']],
             'matched_robust_protocol_sha256':sha256_file(robust_protocol_path),
             'all_seeds_retained':True,'independent_solver_results_used_for_optimisation':False,
             'scope':scientific_context['control_scope']}
    write_json_atomic(output/'NOMINAL_CONTROL_SUMMARY.json',summary)
    write_json_atomic(reports/'NOMINAL_CONTROL_SUMMARY.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    print('FULL_NOMINAL_128_200_10_COMPLETE',flush=True)


if __name__=='__main__':
    main()
