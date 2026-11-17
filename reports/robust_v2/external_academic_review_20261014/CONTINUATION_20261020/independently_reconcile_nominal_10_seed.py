"""Independent SHA-bound analysis of full matched-budget nominal control.

No rerun of the original scientific search; inspect every frozen seed,
population, generation history, candidate budget, feasibility and objective.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020/nominal_control'
OUT=Path(__file__).resolve().parent/'FULL_TEN_SEED_NOMINAL_INDEPENDENT_RECONCILIATION.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
 if OUT.exists():raise FileExistsError('Existing independent nominal analysis is immutable')
 summary_path=BASE/'NOMINAL_CONTROL_SUMMARY.json'
 summary=read(summary_path)
 if summary['status']!='FULL_NOMINAL_PROFILE_COMPLETED' or len(summary['per_seed'])!=10:raise ValueError('Incomplete nominal search')
 all_rows=[]; seen=set()
 for item in summary['per_seed']:
  seed=item['seed']
  if seed in seen:raise ValueError('Duplicated seed')
  seen.add(seed)
  casepath=BASE/item['file']
  if sha(casepath)!=item['sha256']:raise ValueError('Seed hash mismatch')
  case=read(casepath)
  history=case['history']
  if len(history)!=200 or [x['generation'] for x in history]!=list(range(1,201)):
   raise ValueError('Missing nominal generation')
  if any(x['population_count']!=128 for x in history) or history[-1]['evaluations']!=25600:
   raise ValueError('Reduced nominal search budget')
  x=np.asarray(case['solutions'],float)
  f=np.asarray(case['objectives'],float)
  g=np.asarray(case['constraint_values'],float)
  if x.ndim!=2 or x.shape[1]!=12 or f.shape!=(len(x),1) or g.shape!=(len(x),6):
   raise ValueError('Nominal geometry/objective/constraints have wrong shape')
  feasible=np.isfinite(f).all(axis=1)&np.isfinite(g).all(axis=1)&(g<=0).all(axis=1)
  if feasible.sum()!=item['feasible_entries']:
   raise ValueError('Final feasibility differs from reported result')
  actual=float(f[feasible,0].min()) if feasible.any() else None
  if actual!=item['minimum_feasible_nominal_drag']:
   raise ValueError('Final nominal objective differs from reported result')
  all_rows.append({'seed':seed,'generations':200,'candidate_evaluations':25600,
     'feasible_final_entries':int(feasible.sum()),'minimum_feasible_nominal_drag':actual,
     'final_recheck_max_objective_difference':item['final_recheck_max_objective_difference'],
     'case_sha256':sha(casepath)})
 result={'schema':'independent-nominal-full-budget-reconciliation-v1',
  'nominal_summary_sha256':sha(summary_path),
  'robust_matched_protocol_sha256':summary['matched_robust_protocol_sha256'],
  'seed_count':len(all_rows),'completed_generations':sum(x['generations'] for x in all_rows),
  'candidate_evaluations':sum(x['candidate_evaluations'] for x in all_rows),
  'feasible_seeds':sum(x['feasible_final_entries']>0 for x in all_rows),
  'total_feasible_final_entries':sum(x['feasible_final_entries'] for x in all_rows),
  'minimum_feasible_nominal_cd':min(x['minimum_feasible_nominal_drag'] for x in all_rows if x['minimum_feasible_nominal_drag'] is not None),
  'seeds':all_rows,'scope':'Independent numerical check of matched-budget nominal constrained search, not independent CFD or robust-manufacturing validation'}
 OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='seeds'},indent=2),flush=True)
if __name__=='__main__':main()
