"""Launch fixed-design 2048-scenario and XFOIL evidence only after serial gates."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
STATUS=HERE/'CONTROLLED_PIPELINE_STATUS.json'
OUTPUT=HERE/'SELECTED_STUDY_EXECUTION.json'
CONTROLLER_PID=37508

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

if OUTPUT.exists():
 raise FileExistsError('Existing selected study execution receipt requires inspection')
while True:
 if STATUS.is_file():
  status=json.loads(STATUS.read_text(encoding='utf-8'))
  if status.get('status')=='ALL_EXECUTED_GATES_PASS_NOT_INDEPENDENT_CFD_OR_AUTHOR_SUBMISSION':
   break
  if status.get('status')=='STOPPED_FOR_REVIEW':
   raise RuntimeError('Upstream scientific gate failed; do not launch selected solver work')
 time.sleep(60)
lineage=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
for p in ['optimisation_production/PRODUCTION_SUMMARY.json','nominal_control/NOMINAL_CONTROL_SUMMARY.json']:
 if not (lineage/p).is_file():raise FileNotFoundError('Selection prerequisite missing: '+p)
cmd=[sys.executable,'-u','scripts/robust_v2/complete_controlled_selected_study.py']
log=HERE/'selected_study_execution.log'
if log.exists():raise FileExistsError('Prior selected solver log requires inspection')
started=time.monotonic()
with log.open('w',encoding='utf-8') as f:
 f.write('START '+now()+'\n');f.flush()
 proc=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8'),stdout=f,stderr=subprocess.STDOUT,check=False)
record={'status':'PASS' if proc.returncode==0 else 'FAILED','completed_utc':now(),'seconds':time.monotonic()-started,'exit':proc.returncode,'command':cmd,'log':str(log),'full_selected_summary':str(lineage/'selected_study/SELECTED_XFOIL_SUMMARY.json')}
OUTPUT.write_text(json.dumps(record,indent=2),encoding='utf-8')
if proc.returncode:raise SystemExit(proc.returncode)
print('SELECTED_STUDY_COMPLETED',flush=True)
