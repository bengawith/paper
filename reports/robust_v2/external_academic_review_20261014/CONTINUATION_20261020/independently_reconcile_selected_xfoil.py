"""Recompute every strict selected XFOIL outcome and pair from saved raw files.

No new XFOIL invocation, simulated result, permissive fallback or dropped
solver failures. Audits 819 fixed cases with exact file hashes and 45-point
required-lift feasibility on actual stored observed polars.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from robust_airfoil.validation.controlled_selected import assess_selected_polar
ROOT=Path(__file__).resolve().parents[4]
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
OUT=Path(__file__).resolve().parent/'INDEPENDENT_SELECTED_XFOIL_RECONCILIATION.json'
SRC=LINEAGE/'selected_study/SELECTED_XFOIL_SUMMARY.json'
PROTO=LINEAGE/'selected_study/XFOIL_CASE_SELECTION_PROTOCOL.json'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
 if OUT.exists():raise FileExistsError('Independent selected audit already exists')
 result=read(SRC);protocol=read(PROTO)
 assert result['cases_requested']==819
 assert len(result['cases'])==819
 assert protocol['requested_cases']==819
 frozen=LINEAGE/'selected_study/FROZEN_SELECTED_DESIGNS.json'
 if sha(frozen)!=result['preverification_design_selection_sha256']:
  raise ValueError('Finalist selection changed after XFOIL')
 seen=set();stats={}
 for row in result['cases']:
  key=(row['design'],row['campaign'],row['sample_index'])
  if key in seen:raise ValueError('Duplicate selected solver scenario identity')
  seen.add(key)
  group=(row['campaign'],row['design'])
  stat=stats.setdefault(group,{'requested':0,'usable':0,'failed':0})
  stat['requested']+=1
  if not row['usable']:
   stat['failed']+=1
   if row.get('weighted_cd') is not None:
    raise ValueError('Nonusable case contains imputed weighted drag')
   continue
  stat['usable']+=1
  run=ROOT/row['run_folder']
  receipt_path=run/'CASE_RECEIPT.json'
  if sha(receipt_path)!=row['receipt_sha256']:
   raise ValueError('Run receipt hash changed: '+str(run))
  receipt=read(receipt_path)
  for rel,expected in receipt['files_sha256'].items():
   if sha(run/rel)!=expected:raise ValueError('Case file changed: '+str(run/rel))
  if sha(run/'geometry.dat')!=row['geometry_sha256']:
   raise ValueError('Displaced physical geometry mismatch')
  assessment=receipt['assessment']
  if not assessment['usable'] or len(assessment['required_lift_roots'])!=3:
   raise ValueError('Reported usable but not all nominal lift targets')
  raw=pd.read_csv(run/'parsed_points.csv')
  again=assess_selected_polar(raw,return_code=assessment['return_code'],timed_out=assessment['timed_out'])
  if not again['usable'] or again['expected_points']!=45 or again['missing_alpha_deg']:
   raise ValueError('Independent strict solver feasibility check failed')
  if not np.isclose(again['weighted_cd'],row['weighted_cd'],rtol=0,atol=1e-12):
   raise ValueError('Recomputed actual weighted drag mismatch')
 paired={}
 for name,stored in result['paired_conditional_results'].items():
  campaign,design=name.rsplit('_',1)
  # 'risk_knee' / 'nominal_control' have embedded underscore; choose the
  # known declared design suffix rather than naive split.
  for suffix in ('risk_knee','nominal_control'):
   if name.endswith('_'+suffix):
    campaign=name[:-(len(suffix)+1)]
    design=suffix
    break
  ref={r['sample_index']:r['weighted_cd'] for r in result['cases'] if r['campaign']==campaign and r['design']=='reference' and r['usable']}
  alt={r['sample_index']:r['weighted_cd'] for r in result['cases'] if r['campaign']==campaign and r['design']==design and r['usable']}
  common=sorted(set(ref)&set(alt),key=lambda value:-1 if value is None else value)
  if len(common)!=stored['pairs']:
   raise ValueError('Mismatched conditional pair denominator: '+name)
  if common:
   a=np.asarray([ref[k] for k in common],float)
   b=np.asarray([alt[k] for k in common],float)
   value=float((a-b).mean()/a.mean())
   if not np.isclose(value,stored['mean_reduction'],rtol=0,atol=1e-12):
    raise ValueError('Paired mean reduction changed: '+name)
   paired[name]={'matched_usable_count':len(common),'mean_reduction':value,
      'reference_max':float(a.max()),'candidate_max':float(b.max())}
  else:paired[name]={'matched_usable_count':0,'mean_reduction':None}
 audit={'schema':'independent-controlled-selected-local-xfoil-audit-v1',
  'selection_sha256':sha(frozen),'stored_source_sha256':sha(SRC),
  'protocol_sha256':sha(PROTO),
  'total_attempts':len(seen),'total_usable':sum(s['usable'] for s in stats.values()),
  'total_failed_or_rejected':sum(s['failed'] for s in stats.values()),
  'per_campaign_design':[{'campaign':c,'design':d,**v} for (c,d),v in sorted(stats.items())],
  'conditional_pair_recomputation':paired,
  'scientific_limitation':'Complete same-XFOIL-family selected-case audit. No solver failures imputed, no unconditional manufacturing tail probability or independent CFD.'}
 if audit['total_usable']!=result['usable_cases'] or audit['total_failed_or_rejected']!=result['cases_failed_or_not_admitted']:
  raise ValueError('Stored solver acceptance counts inconsistent')
 OUT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print('INDEPENDENT_ALL_819_SELECTED_CASES_REPRODUCED',audit['total_usable'],'usable',audit['total_failed_or_rejected'],'unusable',flush=True)
if __name__=='__main__':main()
