"""Review immutable fresh 2048-scenario penalised sensitivity without cherry picking."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
SRC=LINEAGE/'selected_study/FINAL_2048_SCENARIO_EVALUATION.json'
OUT=Path(__file__).resolve().parent/'INDEPENDENT_2048_SENSITIVITY_RECONCILIATION.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError('Independent sensitivity review already exists')
 r=json.loads(SRC.read_text(encoding='utf-8'))
 if r['scenario_count']!=2048 or len(r['rows'])!=20:raise ValueError('Incomplete 2048 sensitivity')
 names={s['design'] for s in r['rows']}
 amplitudes={float(s['amplitude_fraction_chord']) for s in r['rows']}
 if names!={'reference','nominal_control','risk_knee','risk_mean','risk_tail'} or amplitudes!={.001,.0025,.005,.01}:
  raise ValueError('Study designs or amplitudes differ from protocol')
 keyed={(s['design'],float(s['amplitude_fraction_chord'])):s for s in r['rows']}
 if len(keyed)!=20:raise ValueError('Reused design/amplitude identity')
 analyses=[]
 for amplitude in sorted(amplitudes):
  ref=keyed['reference',amplitude]
  a=float(ref['mean_penalised_weighted_cd'])
  b=float(ref['empirical_cvar95_penalised_weighted_cd'])
  if not np.isfinite([a,b]).all() or min(a,b)<=0:raise ValueError('Reference objective invalid')
  comparisons=[]
  for name in sorted(names-{'reference'}):
   item=keyed[name,amplitude]
   mean=float(item['mean_penalised_weighted_cd'])
   tail=float(item['empirical_cvar95_penalised_weighted_cd'])
   invalid=int(item['manufacturing_invalid_samples'])
   trust=int(item['uncertainty_trust_violations'])
   if not np.isfinite([mean,tail]).all() or min(mean,tail)<=0 or min(invalid,trust)<0 or max(invalid,trust)>2048:
    raise ValueError('Incomplete or invalid scenario accounting')
   comparisons.append({'design':name,'reference_minus_candidate_mean':a-mean,
     'relative_mean_reduction':(a-mean)/a,
     'reference_minus_candidate_tail':b-tail,
     'relative_tail_reduction':(b-tail)/b,
     'candidate_nominal_feasible':item['nominal_feasible'],
     'candidate_invalid_refits':invalid,'candidate_trust_exits':trust,
     'penalty_objective_not_pure_physical_drag':True})
  analyses.append({'amplitude_fraction_chord':amplitude,
   'reference_mean':a,'reference_cvar95':b,
   'reference_invalid_refits':ref['manufacturing_invalid_samples'],
   'reference_trust_exits':ref['uncertainty_trust_violations'],
   'comparisons':comparisons})
 result={'schema':'independent-prescribed-2048-sensitivity-verification-v1',
   'source_sha256':sha(SRC),'selected_sha256':r['selected_sha256'],
   'scenarios':2048,'amplitudes':sorted(amplitudes),'fixed_design_count':len(names),
   'analysis':analyses,'assessment':'Prescribed-surface, penalty-inclusive fixed-design scenario means and empirical tails. Numerical improvements do not establish measured manufacturing risks or independent aerodynamic CFD.'}
 OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({'status':'VERIFIED','rows':len(r['rows']),'scenario_count':2048,
  'one_percent':analyses[-1],'source_sha256':result['source_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
