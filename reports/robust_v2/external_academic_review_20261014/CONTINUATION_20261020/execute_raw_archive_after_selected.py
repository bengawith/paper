"""Archive all observed raw-XFOIL evidence only after 819-case selected study."""
from __future__ import annotations
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
SELECTED=HERE/'SELECTED_STUDY_EXECUTION.json'
RECEIPT=HERE/'RAW_ARCHIVE_EXECUTION.json'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if RECEIPT.exists():raise FileExistsError('Raw archive execution receipt already exists')
for _ in range(360):
 if SELECTED.is_file():
  status=json.loads(SELECTED.read_text(encoding='utf-8'))
  if status['status']=='PASS':break
  if status['status']=='FAILED':raise RuntimeError('Selected solver stage failed; do not package incomplete evidence')
 time.sleep(60)
else:raise TimeoutError('Complete selected solver evidence not yet available')
output=ROOT/'reports/robust_v2/external_academic_review_20261014'
if (output/'RAW_XFOIL_EVIDENCE.zip').exists():
 raise FileExistsError('Raw evidence archive already exists; inspect rather than rerun')
log=HERE/'RAW_ARCHIVE_EXECUTION.log'
with log.open('w',encoding='utf-8') as file:
 file.write('START '+utc()+'\n');file.flush()
 proc=subprocess.run([sys.executable,'-u',str(HERE/'archive_complete_raw_solver_evidence.py')],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8'),stdout=file,stderr=subprocess.STDOUT,check=False)
receipt={'status':'ARCHIVE_HASH_AND_ZIP_CRC_VERIFIED' if proc.returncode==0 else 'FAILED',
         'completed_utc':utc(),'returncode':proc.returncode,'log':log.relative_to(ROOT).as_posix()}
RECEIPT.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
if proc.returncode:raise SystemExit(proc.returncode)
print('ALL_RAW_XFOIL_CASES_ARCHIVED',flush=True)
