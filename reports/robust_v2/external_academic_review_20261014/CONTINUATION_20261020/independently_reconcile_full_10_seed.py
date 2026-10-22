"""Read-only source-bound verification of every full 128x200x64 robust seed.

Recomputes final feasibility, per-seed 200 generation and 25,600 evaluation
budgets, retained SHA-256 receipts, boundary objectives and declared criteria.
Writes a distinct academic-review audit, not a replacement optimiser output.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/FULL_TEN_SEED_INDEPENDENT_RECONCILIATION.json'
BASE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020/optimisation_production'

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError('Independent scientific audit already exists')
 main_summary=load(BASE/'PRODUCTION_SUMMARY.json')
 protocol=load(BASE/'PRODUCTION_PROTOCOL.json')
 p=main_summary['profile']
 if (p['population'],p['generations'],p['uncertainty_samples'],p['independent_seeds'])!=(128,200,64,10):
  raise AssertionError('Production budget changed')
 if len(main_summary['per_seed'])!=10:raise AssertionError('Missing seed')
 seeds=[];seen=set();feasible_all=0
 for item in main_summary['per_seed']:
  seed=item['seed']
  if seed in seen:raise AssertionError('Duplicate seed')
  seen.add(seed)
  target=BASE/item['file']
  if not target.is_file() or sha(target)!=item['sha256']:raise AssertionError('Seed evidence changed')
  source=load(target)
  if source['seed']!=seed or len(source['history'])!=200:raise AssertionError('Incomplete seed evidence')
  history=source['history']
  if any(int(row['generation'])!=i for i,row in enumerate(history,1)):
   raise AssertionError('Generation sequence absent or inconsistent')
  if any(row['population_count']!=128 for row in history):
   raise AssertionError('A generation silently reduced its population')
  if int(history[-1]['evaluations'])!=25_600:raise AssertionError('Incomplete candidate evaluation budget')
  f=np.asarray(source['objectives'],float);g=np.asarray(source['constraint_values'],float)
  x=np.asarray(source['solutions'],float)
  if x.ndim!=2 or x.shape[1]!=12 or f.shape!=(len(x),2) or g.shape!=(len(x),6):
   raise AssertionError('Reported optimisation solution structure broken')
  feasible=np.isfinite(f).all(axis=1)&np.isfinite(g).all(axis=1)&(g<=0).all(axis=1)
  count=int(feasible.sum())
  if count!=item['feasible_entries']:raise AssertionError('Feasibility mismatch')
  if (source['sample_count']!=64 or source['schema_version']!=6 or
      not np.isfinite(source['final_recheck_max_objective_difference'])):
   raise AssertionError('Scientific objectives not current validated versions')
  seeds.append({'seed':seed,'generations':len(history),'candidate_evaluation_budget':int(history[-1]['evaluations']),
      'feasible_final_entries':count,
      'first_generation_feasible_fraction':float(history[0]['feasible_fraction']),
      'last_generation_feasible_fraction':float(history[-1]['feasible_fraction']),
      'max_final_objective_recheck_difference':float(source['final_recheck_max_objective_difference']),
      'minimum_final_feasible_mean':float(f[feasible,0].min()) if count else None,
      'minimum_final_feasible_cvar':float(f[feasible,1].min()) if count else None,
      'seed_result_sha256':sha(target)})
  feasible_all+=count
 expected=sum(record['feasible_final_entries']>0 for record in seeds)>=8
 objective=main_summary['reference_scaled_agreement']
 alternative=main_summary['original_pooled_range_diagnostic']
 passed_expected=expected and objective['pairwise_objective_distance_mean']<=.20
 passed_original=expected and alternative['pairwise_objective_distance_mean']<=.20
 if passed_expected!=main_summary['prospective_reference_scaled_criterion_met'] or passed_original!=main_summary['original_unmodified_numerical_criterion_met']:
  raise AssertionError('Published front-agreement predicate not reproduced')
 if passed_expected and passed_original != main_summary['both_numerical_checks_met']:
  raise AssertionError('Published joint result differs')
 data={'schema':'independent-ten-seed-robust-verification-v1','original_protocol_sha256':sha(BASE/'PRODUCTION_PROTOCOL.json'),
   'summary_sha256':sha(BASE/'PRODUCTION_SUMMARY.json'),'total_seeds':len(seeds),
   'aggregate_completed_generations':sum(r['generations'] for r in seeds),
   'aggregate_candidate_evaluations':sum(r['candidate_evaluation_budget'] for r in seeds),
   'aggregate_feasible_final_entries':feasible_all,
   'feasible_seed_count':sum(r['feasible_final_entries']>0 for r in seeds),
   'reference_scaled_mean_distance':objective['pairwise_objective_distance_mean'],
   'pooled_range_mean_distance':alternative['pairwise_objective_distance_mean'],
   'prospective_numerical_criterion_met':passed_expected,
   'retrospective_numerical_criterion_met':passed_original,
   'joint_criterion_met':passed_expected and passed_original,
   'seeds':seeds,
   'scope':'Independent recomputation of frozen finite algorithmic receipts only; not CFD, a proof of global optimality, or measured manufacturing risk.'}
 OUT.write_text(json.dumps(data,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in data.items() if k!='seeds'},indent=2),flush=True)
if __name__=='__main__':main()
