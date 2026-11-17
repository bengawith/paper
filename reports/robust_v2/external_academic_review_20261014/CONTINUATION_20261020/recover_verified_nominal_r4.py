"""Adopt verified successful nominal checkpoint resume without erasing R3 failure.

R3 failed due one transient Windows os.replace WinError5 on generation-progress
metadata AFTER the intact generation-158 algorithm checkpoint was committed.
We preserve both receipt generations and original failed status/log. Only when
the exact scientific context resumes to all ten 200-generation seeds do we
create a separately named R4 gate receipt for existing verified stages.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
R3=HERE/'CONTROLLED_PIPELINE_STATUS_R3.json'
R4=HERE/'CONTROLLED_PIPELINE_STATUS_R4.json'
SUMMARY=LINEAGE/'nominal_control/NOMINAL_CONTROL_SUMMARY.json'
LOG=HERE/'NOMINAL_R4_SAME_CONTEXT_RESUME.log'
OUT=HERE/'NOMINAL_R4_RECOVERY_PROOF.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def safe_existing_stage(name,item):
 if item.get('status')!='PASSED' or item.get('returncode')!=0:
  raise ValueError('R3 historical accepted gate not terminal: '+name)
 path=ROOT/item['log']
 if not path.is_file() or sha(path)!=item['log_sha256']:
  raise ValueError('Historical R3 gate log hash differs: '+name)
 expected=item.get('marker_sha256')
 if expected:
  markers={'evaluation':LINEAGE.parent.parent.parent/'dummy'}
  sources={
   'evaluation':ROOT/'reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/REPAIRED_TEST_METRICS.json',
   'risk_oof':ROOT/'reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/RISK_OOF/REPAIRED_RISK_OOF.json',
   'confirmation_freeze':LINEAGE/'confirmation/PREDICTIONS_BEFORE_LABELS_RECEIPT.json',
   'confirmation_solver':LINEAGE/'confirmation/SOLVER_LABELS_COMPLETE.json',
   'confirmation_analysis':ROOT/'reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/CONFIRMATION/CONFIRMATION_RESULTS.json',
   'production':LINEAGE/'optimisation_production/PRODUCTION_SUMMARY.json',
  }
  p=sources.get(name)
  if p is None or not p.is_file() or sha(p)!=expected:
   raise ValueError('Previously accepted R3 scientific result changed: '+name)
def main():
 if OUT.exists() or R4.exists():
  raise FileExistsError('Nominal R4 recovery already exists, inspect original')
 record=load(R3)
 if record.get('status')!='STOPPED_FOR_REVIEW' or record['stages']['nominal_control']['status']!='FAILED':
  raise ValueError('R3 nominal failure receipt not in expected state')
 for name,item in record['stages'].items():
  if name!='nominal_control':safe_existing_stage(name,item)
 failure=ROOT/record['stages']['nominal_control']['log']
 if sha(failure)!=record['stages']['nominal_control']['log_sha256']:
  raise ValueError('The failed nominal run log was altered')
 if 'PermissionError: [WinError 5]' not in failure.read_text(encoding='utf-8'):
  raise ValueError('Different failure root cause; do not reinterpret it as Windows transient')
 for fn in ['NOMINAL_R3_FAILED_GEN158_UNCOMMITTED.json','NOMINAL_R3_LAST_RECORDED_GEN157.json','NOMINAL_R3_VALID_CHECKPOINT_GEN158.json']:
  if not (HERE/fn).is_file():raise FileNotFoundError('Missing pre-resume failure evidence: '+fn)
 for _ in range(180):
  if SUMMARY.is_file():break
  text=LOG.read_bytes().decode('utf-16' if LOG.read_bytes().startswith(b'\xff\xfe') else 'utf-8',errors='replace') if LOG.exists() else ''
  if 'Traceback (most recent call last)' in text:
   raise RuntimeError('R4 same-context nominal resume itself failed')
  time.sleep(30)
 if not SUMMARY.is_file():raise TimeoutError('Nominal run not terminal within reserved window')
 # Read-only evidence validation, not a replacement or re-optimisation.
 nominal=load(SUMMARY)
 if nominal.get('status')!='FULL_NOMINAL_PROFILE_COMPLETED' or len(nominal.get('per_seed',[]))!=10:
  raise ValueError('Matched nominal study did not finish all ten seeds')
 if any(x.get('generations')!=200 for x in nominal['per_seed']):
  raise ValueError('A nominal seed has an incomplete scientific budget')
 for item in nominal['per_seed']:
  case=LINEAGE/'nominal_control'/item['file']
  if not case.is_file() or sha(case)!=item['sha256']:
   raise ValueError('Completed nominal seed hash mismatch: '+str(case))
  payload=load(case)
  if len(payload.get('history',[]))!=200 or payload['seed']!=item['seed']:
   raise ValueError('Nominal seed history mismatch')
 text=LOG.read_bytes().decode('utf-16' if LOG.read_bytes().startswith(b'\xff\xfe') else 'utf-8',errors='replace')
 if 'FULL_NOMINAL_128_200_10_COMPLETE' not in text:
  raise ValueError('Nominal resumption completion log is not terminal')
 # Prove original R3 failure remains immutable.
 R3_before=sha(R3)
 provenance={
  'schema':'source-unchanged-checkpoint-nominal-recovery-v1',
  'completed_utc':stamp(),
  'r3_terminal_failed_state_sha256':R3_before,
  'failure_log_sha256':sha(failure),
  'root_cause':'Windows PermissionError [WinError 5] replacing non-scientific GENE​​RATION_PROGRESS.json.tmp; separate complete generation 158 algorithm_checkpoint is intact',
  'scientific_source_modified_in_recovery':False,
  'failed_file_snapshots':{p.name:sha(p) for p in [HERE/'NOMINAL_R3_FAILED_GEN158_UNCOMMITTED.json',HERE/'NOMINAL_R3_LAST_RECORDED_GEN157.json',HERE/'NOMINAL_R3_VALID_CHECKPOINT_GEN158.json']},
  'same_scientific_context_resume_log_sha256':sha(LOG),
  'nominal_summary_sha256':sha(SUMMARY),
  'seed_count':10,'generations_per_seed':200,
  'seeds_sha256':{str(x['seed']):x['sha256'] for x in nominal['per_seed']},
  'next_action':'Adopt already completed scientific gates into a new R4 read-only acceptance receipt; launch no duplicate trial.'
 }
 OUT.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
 inherited={name:item for name,item in record['stages'].items() if item['status']=='PASSED'}
 inherited['nominal_control']={
  'status':'PASSED','returncode':0,'finished_utc':stamp(),
  'command':['same-source checkpoint resume','scripts/robust_v2/run_controlled_nominal.py'],
  'log':LOG.relative_to(ROOT).as_posix(),
  'log_sha256':sha(LOG),
  'marker_sha256':sha(SUMMARY),
  'original_failed_R3_log_sha256':sha(failure),
  'nominal_recovery_proof_sha256':sha(OUT),
 }
 R4.write_text(json.dumps({
  'schema':'serial-controlled-gate-execution-v1',
  'created_utc':stamp(),
  'stages':inherited,
  'status':'ADOPT_VERIFIED_TERMINAL_RECEIPTS_AWAITING_R4_CONTROLLER',
  'training':record['training'],
  'original_R3_failure_preserved_sha256':R3_before,
  'recovery_proof_sha256':sha(OUT),
  'scope':'Reconstructible new receipt; no scientific rerun, no acceptance threshold change, no claim of independent CFD.'},indent=2),encoding='utf-8')
 print('NOMINAL_R4_10x200_RESUME_VERIFIED_AND_ADOPTED',flush=True)
if __name__=='__main__':main()
