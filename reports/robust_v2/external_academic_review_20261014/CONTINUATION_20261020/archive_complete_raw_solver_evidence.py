"""Archive exact local XFOIL raw evidence without rewriting original case receipts.

Wait for all 819 strict selected cases and 200 confirmation shapes. This archive
is an auditable offline research handoff, not proof of independent CFD.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import zipfile
import datetime
ROOT=Path(__file__).resolve().parents[4]
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
REVIEW=ROOT/'reports/robust_v2/external_academic_review_20261014'
OUT=REVIEW/'RAW_XFOIL_EVIDENCE.zip'
MANIFEST=REVIEW/'RAW_XFOIL_EVIDENCE_SHA256.json'
ROOTS=[LINEAGE/'solver_labels',LINEAGE/'confirmation/solver_labels',LINEAGE/'selected_study/xfoil']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 if OUT.exists() or MANIFEST.exists():raise FileExistsError('Previous raw-evidence archive exists; do not overwrite')
 for marker in [LINEAGE/'DATA_MANIFEST.json',LINEAGE/'confirmation/SOLVER_LABELS_COMPLETE.json',LINEAGE/'selected_study/SELECTED_XFOIL_SUMMARY.json']:
  if not marker.is_file():raise FileNotFoundError('Scientific raw-case evidence is incomplete: '+str(marker))
 selected=json.loads((LINEAGE/'selected_study/SELECTED_XFOIL_SUMMARY.json').read_text(encoding='utf-8'))
 if selected['cases_requested']!=len(selected['cases']):raise ValueError('Selected case accounting incomplete')
 cases=[p for p in ROOTS if p.exists()]
 if len(cases)!=len(ROOTS):raise FileNotFoundError('One scientific evidence tree is missing')
 files=[p for root in cases for p in root.rglob('*') if p.is_file()]
 files=sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())
 if len(files)<15000:raise ValueError('Insufficient raw case files for the full experiment')
 temp=OUT.with_name('RAW_XFOIL_EVIDENCE.inprogress.zip')
 if temp.exists():raise FileExistsError('Previous interrupted archive needs inspection')
 entries=[]
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as archive:
  for index,path in enumerate(files):
   relative=path.relative_to(ROOT).as_posix()
   observed=sha(path)
   archive.write(path,arcname=relative)
   if sha(path)!=observed:raise ValueError('A raw evidence file changed during packaging: '+relative)
   entries.append({'path':relative,'sha256':observed,'bytes':path.stat().st_size})
   if index%2000==0:print('ARCHIVED_RAW_FILE',index,'/',len(files),flush=True)
 with zipfile.ZipFile(temp,'r') as archive:
  if archive.testzip() is not None:raise RuntimeError('Raw solver archive CRC check failed')
  names=set(archive.namelist())
  if names!={v['path'] for v in entries}:raise RuntimeError('Raw archive manifest mismatch')
 temp.replace(OUT)
 record={'schema':'controlled-local-XFOIL-offline-evidence-v1',
         'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'root_tree_paths':[p.relative_to(ROOT).as_posix() for p in cases],
         'original_file_count':len(entries),
         'original_total_bytes':sum(x['bytes'] for x in entries),
         'archive_bytes':OUT.stat().st_size,'archive_sha256':sha(OUT),
         'source_case_counts':{'controlled_geometries':1256,'sealed_confirmation_geometries':200,
                               'strict_selected_requested':selected['cases_requested']},
         'source_frozen_numerical_outcomes_not_modified':True,
         'files':entries,'claim_scope':'Raw same-family XFOIL solver evidence, NOT independent CFD or experimental proof.'}
 MANIFEST.write_text(json.dumps(record,indent=2),encoding='utf-8')
 print('COMPLETE_RAW_XFOIL_ARCHIVE',len(entries),'files',OUT.stat().st_size,'bytes',flush=True)
if __name__=='__main__':main()
