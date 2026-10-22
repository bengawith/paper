"""Serial, receipt-backed continuation. Never duplicate a running scientific task.

Stages are fail-closed and preserve all frozen input/output. This controller
does not modify source, accept results on its own, or hide a failed gate.
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
import psutil

ROOT=Path(__file__).resolve().parents[4]
REPORTS=ROOT/'reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL'
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
HERE=Path(__file__).resolve().parent
RECEIPT=HERE/'CONTROLLED_PIPELINE_STATUS_R4.json'
TRAIN_LOG=REPORTS/'training_final.log'
TRAIN_STATE=REPORTS/'TRAINING_PROGRESS.json'
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
STAGES=[
 ('ruff',['uv','run','ruff','check','src/robust_airfoil','tests/robust_v2','scripts/robust_v2'],None),
 ('mypy',['uv','run','mypy','src/robust_airfoil'],None),
 ('full_tests',[sys.executable,'-m','pytest','-m','not network','--run-xfoil','-q'],None),
 ('evaluation',[sys.executable,'-u','scripts/robust_v2/evaluate_repaired_lineage.py','--lineage','controlled'],REPORTS/'REPAIRED_TEST_METRICS.json'),
 ('risk_oof',[sys.executable,'-u','scripts/robust_v2/audit_repaired_risk.py','--lineage','controlled'],REPORTS/'RISK_OOF/REPAIRED_RISK_OOF.json'),
 ('confirmation_freeze',[sys.executable,'-u','scripts/robust_v2/freeze_confirmation_predictor.py'],LINEAGE/'confirmation/PREDICTIONS_BEFORE_LABELS_RECEIPT.json'),
 ('confirmation_solver',[sys.executable,'-u','scripts/robust_v2/run_confirmation_solver.py'],LINEAGE/'confirmation/SOLVER_LABELS_COMPLETE.json'),
 ('confirmation_analysis',[sys.executable,'-u','scripts/robust_v2/analyse_controlled_confirmation.py'],REPORTS/'CONFIRMATION/CONFIRMATION_RESULTS.json'),
 ('production',[sys.executable,'-u','scripts/robust_v2/run_repaired_production.py','--lineage','controlled'],LINEAGE/'optimisation_production/PRODUCTION_SUMMARY.json'),
 ('nominal_control',[sys.executable,'-u','scripts/robust_v2/run_controlled_nominal.py'],LINEAGE/'nominal_control/NOMINAL_CONTROL_SUMMARY.json'),
]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def load():
 if RECEIPT.exists():return json.loads(RECEIPT.read_text(encoding='utf-8'))
 return {'schema':'serial-controlled-gate-execution-v1','created_utc':utc(),'stages':{},'status':'WAITING_FOR_TRAINING','scope':'Completes verified gates sequentially without claiming journal readiness or independent CFD'}
def save(state):
 t=RECEIPT.with_suffix('.next')
 t.write_text(json.dumps(state,indent=2),encoding='utf-8')
 t.replace(RECEIPT)
def validated_marker(name,marker):
 if marker is None:return None
 data=json.loads(marker.read_text(encoding='utf-8'))
 if name=='evaluation' and (data.get('states',0)<1000 or not data.get('model_frozen_before_test')):raise ValueError('Incomplete controlled evaluation')
 if name=='risk_oof' and data.get('operational_policy_changed'):raise ValueError('Risk policy unexpectedly changed')
 if name=='confirmation_freeze' and data.get('confirmation_labels_generated'):raise ValueError('Freeze is not before-label')
 if name=='confirmation_solver' and data.get('requested_geometries')!=200:raise ValueError('Incomplete confirmation geometry accounting')
 if name=='production' and (data.get('status')!='FULL_PROFILE_COMPLETED' or len(data.get('per_seed',[]))!=10 or any(x.get('generations')!=200 for x in data['per_seed'])):raise ValueError('Incomplete full production')
 return sha(marker)
def run():
 state=load();save(state)
 while True:
  if TRAIN_STATE.exists():
   train=json.loads(TRAIN_STATE.read_text(encoding='utf-8'))
   if train.get('status')=='FIVE_MEMBERS_TRAINED' and train.get('completed_members')==5:
    manifest=LINEAGE/'full_ensemble/ensemble_manifest.json'
    if not manifest.is_file():raise RuntimeError('Five-member receipt exists but ensemble manifest absent')
    state['training']={'status':'COMPLETE','sha256':sha(manifest),'best_epochs':train.get('best_epochs')};break
  if not psutil.pid_exists(34400):
   raise RuntimeError('Training wrapper PID34400 finished without an accepted five-member receipt; inspect before continuing')
  state['last_training_check_utc']=utc();save(state)
  time.sleep(20)
 save(state)
 for name,cmd,marker in STAGES:
  item=state['stages'].get(name)
  if item and item.get('status')=='PASSED':
   if marker and validated_marker(name,marker)!=item['marker_sha256']:raise RuntimeError(f'Previously verified {name} changed')
   continue
  if marker and marker.exists():
   raise RuntimeError(f'{name} has an unrecorded existing marker; inspect rather than duplicate or silently accept')
  log=HERE/f'controlled_stage_r4_{name}.log'
  if log.exists():raise FileExistsError('Existing stage log needs review: '+str(log))
  state['status']='RUNNING_'+name
  state['stages'][name]={'status':'RUNNING','started_utc':utc(),'command':cmd,'log':str(log.relative_to(ROOT))}
  save(state)
  print('START_GATE',name,utc(),flush=True)
  started=time.monotonic()
  with log.open('w',encoding='utf-8') as f:
   f.write('COMMAND '+json.dumps(cmd)+'\nUTC '+utc()+'\n');f.flush()
   proc=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=False)
  completed={'status':'PASSED' if proc.returncode==0 else 'FAILED','returncode':proc.returncode,'finished_utc':utc(),'elapsed_seconds':time.monotonic()-started,'command':cmd,'log':str(log.relative_to(ROOT)),'log_sha256':sha(log)}
  if proc.returncode==0 and marker:
   if not marker.is_file():completed.update(status='FAILED',error='Required marker absent after success')
   else:completed['marker_sha256']=validated_marker(name,marker)
  state['stages'][name]=completed
  save(state)
  print('END_GATE',name,completed['status'],completed['elapsed_seconds'],flush=True)
  if completed['status']!='PASSED':raise RuntimeError(name+' failed; inspect '+str(log))
 state['status']='ALL_EXECUTED_GATES_PASS_NOT_INDEPENDENT_CFD_OR_AUTHOR_SUBMISSION'
 state['finished_utc']=utc();save(state)
if __name__=='__main__':
 try:run()
 except Exception as exc:
  state=load();state['status']='STOPPED_FOR_REVIEW';state['error']=repr(exc);state['stopped_utc']=utc();save(state)
  print('PIPELINE_STOP',repr(exc),flush=True)
  raise
