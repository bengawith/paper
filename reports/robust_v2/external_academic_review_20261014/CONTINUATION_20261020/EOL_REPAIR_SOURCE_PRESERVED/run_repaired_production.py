"""Execute the unreduced 128 x 200 x 64 x 10 rebuilt optimisation campaign.

Every generation is checkpointed under a frozen source/model context. The
original pooled-range numerical criterion and prospective reference scales are
reported separately; neither is called proof of a global optimum.
"""
from __future__ import annotationsimport argparseimport jsonimport osimport subprocessimport timefrom pathlib import Pathos.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import numpy as npimport pandas as pdimport torchfrom robust_airfoil.config import UncertaintyConfig, load_yamlfrom robust_airfoil.hashing import hash_object, sha256_filefrom robust_airfoil.modelling.dataset import FEATURE_COLUMNSfrom robust_airfoil.modelling.ensemble import LoadedEnsemblefrom robust_airfoil.modelling.freeze import verify_frozen_artifactsfrom robust_airfoil.optimisation.checkpointing import (    assert_campaign_resources,    load_algorithm_checkpoint,)from robust_airfoil.optimisation.problems import DesignBounds, development_boundsfrom robust_airfoil.optimisation.study import (    RobustCandidateEvaluator,    _agreement,    _RobustProblem,    _run_nsga_profile,)from robust_airfoil.provenance import utc_nowfrom robust_airfoil.research_lineage import load_optimisation_for_lineage, resolve_lineagefrom robust_airfoil.uncertainty.sampling import sobol_normal_samplesROOT=Path(__file__).resolve().parents[2]
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-rebuilt-20261014'
REVIEW=ROOT/'reports/robust_v2/external_academic_review_20261014/REBUILT_MODEL'
OUTPUT=LINEAGE/'optimisation_production'


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight-only',action='store_true',help='Run the separately labelled tiny real-model checkpoint integration check only')
    parser.add_argument('--lineage', choices=['rebuilt','controlled'], default='rebuilt', help='Explicit immutable scientific revision; never overwrites the other lineage')
    args=parser.parse_args()
    paths=resolve_lineage(ROOT,args.lineage)
    LINEAGE=paths['results']
    REVIEW=paths['reports']
    OUTPUT=LINEAGE/'optimisation_production'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    REVIEW.mkdir(parents=True,exist_ok=True)
    resources=assert_campaign_resources(OUTPUT)
    frozen_path=LINEAGE/'FROZEN_MODEL_MANIFEST.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    verify_frozen_artifacts(frozen_path,{key:Path(value) for key,value in frozen['artifact_paths'].items()})
    development=pd.read_parquet(LINEAGE/'partition_inputs/development.parquet')
    if not development['split'].eq('development').all():
        raise ValueError('Non-development rows cannot determine production bounds or seeds')
    shapes=pd.read_parquet(paths['data']/'airfoils.parquet')
    reference=shapes.loc[shapes.airfoil_id.eq('naca2412'),FEATURE_COLUMNS[:-1]].to_numpy(float)[0]
    opt = load_optimisation_for_lineage(ROOT, args.lineage)
    unc=UncertaintyConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/uncertainty.yaml'))
    profile=opt.full_nsga2
    if (profile.population,profile.generations,profile.uncertainty_samples,profile.independent_seeds)!=(128,200,64,10):
        raise ValueError('The declared production profile was changed; do not silently lower its budget')
    ensemble=LoadedEnsemble(LINEAGE/'full_ensemble')
    calibration=json.loads((LINEAGE/'calibration/trust_calibration.json').read_text(encoding='utf-8'))
    parameters=development[FEATURE_COLUMNS[:-1]].drop_duplicates().to_numpy(float)
    derived=development_bounds(parameters)
    lower=np.minimum(derived.lower,reference)
    upper=np.maximum(derived.upper,reference)
    lower[-1]=max(lower[-1],0.)
    bounds=DesignBounds(lower,upper)
    latent=sobol_normal_samples(64,2*unc.basis_control_points_per_surface,unc.seed+2000,antithetic=True)
    evaluator=RobustCandidateEvaluator(reference,np.zeros((64,12)),np.zeros(64,bool),development,ensemble,calibration,opt,candidate_latent=latent,uncertainty=unc)
    reference_f,reference_g,reference_d=evaluator.evaluate(reference[None,:])
    scales=reference_f[0].copy()
    if not np.isfinite(scales).all() or np.any(scales<=0) or np.any(reference_g[0]>0):
        raise ValueError('Reference cannot supply valid predeclared objective scales and feasibility')
    source_hashes={p.relative_to(ROOT/'src/robust_airfoil').as_posix():sha256_file(p) for p in (ROOT/'src/robust_airfoil').rglob('*.py')}
    context={'frozen_model_manifest_sha256':sha256_file(frozen_path),'source_sha256':source_hashes,
             'profile':profile.model_dump(mode='json'),'optimisation':opt.model_dump(mode='json'),
             'uncertainty':unc.model_dump(mode='json'),'latent_seed':unc.seed+2000,
             'reference_parameters':reference.tolist(),'reference_objective_scales':scales.tolist(),
             'bounds':{'lower':bounds.lower.tolist(),'upper':bounds.upper.tolist()},
             'seed_list':[opt.seed+3000+i for i in range(10)],
             'original_distance_threshold':.20,'minimum_feasible_fraction':.80,
             'scale_policy':'Both predeclared reference magnitudes and original post-hoc pooled-range numerical criterion are reported without substituting one for the other.'}
    context_hash=hash_object(context)
    preflight_root=REVIEW/'PRODUCTION_PREFLIGHT'/context_hash[:12]
    preflight_root.mkdir(parents=True,exist_ok=True)
    tiny=profile.model_copy(update={'population':4,'generations':2,'uncertainty_samples':4,'independent_seeds':1})
    tiny_result=_run_nsga_profile('rebuilt-integration-preflight',tiny,opt,reference,latent,np.zeros(64,bool),development,ensemble,calibration,bounds,preflight_root,8,sha256_file(LINEAGE/'full_ensemble/ensemble_manifest.json'),unc,objective_scales=scales,checkpoint_each_generation=True)
    seed=opt.seed+9000
    checkpoint_root=preflight_root/'checkpoints'/f'seed_{seed}'
    receipt=json.loads((checkpoint_root/'algorithm_checkpoint.json').read_text(encoding='utf-8'))
    test_evaluator=RobustCandidateEvaluator(reference,np.zeros((4,12)),np.zeros(4,bool),development,ensemble,calibration,opt,candidate_latent=latent[:4],uncertainty=unc)
    restored=load_algorithm_checkpoint(checkpoint_root,receipt['context'],problem=_RobustProblem(test_evaluator,bounds),evaluator=test_evaluator)
    if restored.has_next() or len(restored.callback.history)!=2 or restored.callback.evaluator is not test_evaluator:
        raise AssertionError('Real-model checkpoint integration failed')
    proof={'status':'REAL_MODEL_PREFLIGHT_PASS_NOT_PRODUCTION','source_context_hash':context_hash,'directory':preflight_root.relative_to(ROOT).as_posix(),'profile':tiny.model_dump(mode='json'),'generation_history_count':len(restored.callback.history),'checkpoint_hash_verified':True,'verified_model_rebound':True,'seed_feasible_counts':tiny_result['agreement']['feasible_solution_counts']}
    (REVIEW/'PRODUCTION_PREFLIGHT.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof,indent=2),flush=True)
    if args.preflight_only:
        return
    protocol_path=OUTPUT/'PRODUCTION_PROTOCOL.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
        if protocol['context_hash']!=context_hash:
            raise ValueError('Production context changed; preserve this campaign and create an explicit new revision')
    else:
        protocol={'schema':'full-rebuilt-production-admission-v1','frozen_before_first_production_seed_utc':utc_now(),'context':context,'context_hash':context_hash,'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'initial_resources':resources,'reference_objectives':reference_f[0].tolist(),'reference_constraints':reference_g[0].tolist(),'reference_diagnostics':reference_d[0]}
        protocol_path.write_text(json.dumps(protocol,indent=2,allow_nan=False),encoding='utf-8')
        np.save(OUTPUT/'common_latent_scenarios.npy',latent,allow_pickle=False)
    if not np.array_equal(np.load(OUTPUT/'common_latent_scenarios.npy',allow_pickle=False),latent):
        raise ValueError('Stored common random scenarios changed')
    (REVIEW/'PRODUCTION_PROTOCOL.json').write_text(json.dumps(protocol,indent=2,allow_nan=False),encoding='utf-8')
    print('FULL_PRODUCTION_BEGIN',context_hash,'seeds',context['seed_list'],flush=True)
    start=time.monotonic()
    result=_run_nsga_profile('rebuilt-full-production',profile,opt,reference,latent,np.zeros(64,bool),development,ensemble,calibration,bounds,OUTPUT,2,sha256_file(LINEAGE/'full_ensemble/ensemble_manifest.json'),unc,objective_scales=scales,checkpoint_each_generation=True)
    after={p.relative_to(ROOT/'src/robust_airfoil').as_posix():sha256_file(p) for p in (ROOT/'src/robust_airfoil').rglob('*.py')}
    if after!=source_hashes:
        raise RuntimeError('Scientific source changed during the campaign; retain evidence but do not approve it')
    if len(result['runs'])!=10 or any(len(r['history'])!=200 for r in result['runs']):
        raise AssertionError('Incomplete production seed/generation count')
    original=_agreement(result['runs'],bounds)
    original_met=bool(original['objective_agreement']['descriptive_threshold_satisfied'])
    prospective_met=bool(result['agreement']['objective_agreement']['passed'])
    summary={'schema':'full-rebuilt-production-completion-v1','status':'FULL_PROFILE_COMPLETED','utc':utc_now(),'profile':profile.model_dump(mode='json'),'source_context_hash':context_hash,'invocation_seconds':time.monotonic()-start,'reference_scaled_agreement':result['agreement'],'original_pooled_range_diagnostic':original,'original_unmodified_numerical_criterion_met':original_met,'prospective_reference_scaled_criterion_met':prospective_met,'both_numerical_checks_met':original_met and prospective_met,'per_seed':[{'seed':r['seed'],'generations':len(r['history']),'reported_entries':r['solution_count'],'feasible_entries':int(np.all(np.asarray(r['constraint_values'])<=0,axis=1).sum()),'returned_population_fallback':r['returned_population_fallback'],'final_recheck_max_objective_difference':r['final_recheck_max_objective_difference'],'file':f"nsga2_rebuilt-full-production_seed_{r['seed']}.json",'sha256':sha256_file(OUTPUT/f"nsga2_rebuilt-full-production_seed_{r['seed']}.json")} for r in result['runs']],'interpretation':'Finite-budget seeded algorithm agreement, not proof of a global optimum, manufacturing distribution or independent CFD.'}
    (OUTPUT/'PRODUCTION_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (REVIEW/'PRODUCTION_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('FULL_128_200_64_10_PRODUCTION_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
