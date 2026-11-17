"""Prepare traceable per-page Word-derived previews after a terminal author draft.

This *does not* declare visual QA complete, does not claim native Word page
identity, and never overwrites existing rendering or prior failed evidence.
"""
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
AUTHOR=HERE/'AUTHOR_DRAFT_EXECUTION.json'
RUNNING=HERE/'SELECTED_STUDY_EXECUTION.json'
RECEIPT=HERE/'VISUAL_PREVIEW_EXECUTION.json'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if RECEIPT.exists():raise FileExistsError('Existing rendering receipt needs review')
while True:
 if AUTHOR.exists():
  a=json.loads(AUTHOR.read_text(encoding='utf-8'))
  if a.get('status')=='DRAFT_GENERATED_AWAITING_VISUAL_QA_AND_RELEASE':break
  if a.get('status')=='STOPPED_FOR_REVIEW':raise RuntimeError('Author document generation failed')
 if RUNNING.exists() and json.loads(RUNNING.read_text(encoding='utf-8')).get('status')=='FAILED':
  raise RuntimeError('Selected study did not complete')
 time.sleep(60)
log=HERE/'visual_preview_execution.log'
if log.exists():raise FileExistsError('Prior visual preview log needs review')
command=[sys.executable,'-u','reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/prepare_visual_qa.py']
with log.open('w',encoding='utf-8') as output:
 output.write('START '+utc()+'\n');output.flush()
 task=subprocess.run(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8'),stdout=output,stderr=subprocess.STDOUT,check=False)
receipt={'status':'PAGES_PREPARED_INSPECTION_PENDING' if task.returncode==0 else 'FAILED','exit':task.returncode,'updated_utc':utc(),
         'command':command,'log':str(log.relative_to(ROOT)),'human_inspection_claim':False,'native_word_render_claim':False}
RECEIPT.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
if task.returncode:raise SystemExit(task.returncode)
print('PREVIEWS_READY_MANUAL_PAGE_QA_REQUIRED',flush=True)
