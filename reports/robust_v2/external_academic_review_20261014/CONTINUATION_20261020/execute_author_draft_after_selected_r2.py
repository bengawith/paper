"""Generate factual manuscript and Word only after every numerical gate is terminal."""
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
PIPELINE=HERE/'CONTROLLED_PIPELINE_STATUS_R2.json'
RECEIPT=HERE/'AUTHOR_DRAFT_EXECUTION.json'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if RECEIPT.exists():raise FileExistsError('Previously generated author draft has a receipt; preserve for review')
while not SELECTED.is_file():
 if PIPELINE.is_file() and json.loads(PIPELINE.read_text(encoding='utf-8')).get('status')=='STOPPED_FOR_REVIEW':
  raise RuntimeError('A scientific gate failed; no paper can be finalised')
 time.sleep(60)
selected=json.loads(SELECTED.read_text(encoding='utf-8'))
if selected.get('status')!='PASS':raise RuntimeError('Selected numerical verification did not complete')
outputs={}
for name,relative in [('paper','reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/build_controlled_manuscript.py'),
                      ('referee','reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/finalize_controlled_review.py')]:
 log=HERE/f'{name}_generation.log'
 if log.exists():raise FileExistsError('Previously attempted paper generation needs inspection: '+str(log))
 cmd=[sys.executable,'-u',relative]
 began=time.monotonic()
 with log.open('w',encoding='utf-8') as f:
  f.write('START '+stamp()+'\n');f.flush()
  job=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8'),stdout=f,stderr=subprocess.STDOUT,check=False)
 outputs[name]={'exit':job.returncode,'elapsed_seconds':time.monotonic()-began,'log':str(log)}
 RECEIPT.write_text(json.dumps({'status':'RUNNING' if job.returncode==0 else 'STOPPED_FOR_REVIEW',
    'updated_utc':stamp(),'stages':outputs},indent=2),encoding='utf-8')
 if job.returncode:raise RuntimeError(name+' scientific source reconciliation failed: '+str(log))
RECEIPT.write_text(json.dumps({'status':'DRAFT_GENERATED_AWAITING_VISUAL_QA_AND_RELEASE',
    'updated_utc':stamp(),'stages':outputs,'journal_submission_ready':False},indent=2),encoding='utf-8')
print('AUTHOR_REVIEW_DRAFT_AND_SEPARATE_REFEREE_PREPARED',flush=True)
