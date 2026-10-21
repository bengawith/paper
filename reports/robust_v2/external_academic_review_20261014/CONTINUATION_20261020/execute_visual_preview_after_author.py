"""Prepare the per-page Word-content browser preview after author draft finishes.

Retains non-native rendering limitation; does not claim manual visual QA.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
AUTHOR=HERE/'AUTHOR_DRAFT_EXECUTION.json'
RECEIPT=HERE/'VISUAL_PREVIEW_EXECUTION.json'
OUT=HERE/'visual_preview_execution.log'
if RECEIPT.exists() or OUT.exists():
 raise FileExistsError('Previous visual preview attempt requires manual inspection')
def clock():return datetime.datetime.now(datetime.timezone.utc).isoformat()
while not AUTHOR.is_file():
 time.sleep(60)
while True:
 status=json.loads(AUTHOR.read_text(encoding='utf-8'))
 if status['status']=='DRAFT_GENERATED_AWAITING_VISUAL_QA_AND_RELEASE':
  break
 if status['status']=='STOPPED_FOR_REVIEW':
  raise RuntimeError('Author draft failed; do not create or accept visual QA')
 time.sleep(30)
cmd=[sys.executable,'-u','reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/prepare_visual_qa.py']
start=time.monotonic()
with OUT.open('w',encoding='utf-8') as log:
 log.write('UTC '+clock()+'\n');log.flush()
 proc=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT,check=False)
record={'status':'PREVIEW_GENERATED_NEEDS_MANUAL_PAGE_INSPECTION' if proc.returncode==0 else 'STOPPED_FOR_REVIEW',
 'utc':clock(),'elapsed_seconds':time.monotonic()-start,'exit':proc.returncode,
 'log':OUT.relative_to(ROOT).as_posix(),'manual_all_pages_accepted':False}
RECEIPT.write_text(json.dumps(record,indent=2),encoding='utf-8')
if proc.returncode:raise RuntimeError('Word-derived preview failed: '+str(OUT))
print('WORD_CONTENT_PREVIEW_READY_FOR_MANUAL_QA',flush=True)
