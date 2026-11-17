"""Source-bound final copyedit; no research numbers or solver results altered."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import runpy
from docx import Document
ROOT=Path(__file__).resolve().parents[4]
REVIEW=ROOT/'reports/robust_v2/external_academic_review_20261014'
HERE=Path(__file__).resolve().parent
PAPER=ROOT/'paper'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def replace_and_preserve(path,old,new,label):
 before=path.read_text(encoding='utf-8')
 if before.count(old)!=1:raise ValueError('Unexpected source content for '+label)
 dest=HERE/('BEFORE_FINAL_COPYEDIT_'+label+'.md')
 if dest.exists():raise FileExistsError('Copyedit already performed: '+str(dest))
 dest.write_bytes(path.read_bytes())
 changed=before.replace(old,new)
 path.write_text(changed,encoding='utf-8')
 return {'path':path.relative_to(ROOT).as_posix(),'previous_sha256':sha(dest),'current_sha256':sha(path),'backup':dest.relative_to(ROOT).as_posix()}
def main():
 for name,path in [('manuscript',PAPER/'LATEST_MANUSCRIPT_FOR_AVA.docx'),('referee',REVIEW/'SCIENTIFIC_REVIEW_REPORT.docx')]:
  backup=HERE/('BEFORE_FINAL_COPYEDIT_'+name+'.docx')
  if backup.exists():raise FileExistsError('The original Word source was already preserved: '+str(backup))
  backup.write_bytes(path.read_bytes())
 updates=[]
 updates.append(replace_and_preserve(PAPER/'manuscript.md',
  'Calibration-only ensemble disagreement changes held-out high-error ranking AUROC by 0.267, subject to the cohort\'s observed-label selection.',
  'On the sealed cohort with observable solver labels, the calibration-trained frozen combined support/disagreement score raises high-error ranking AUROC by 0.267 over support alone; the disagreement-only score performs slightly better still (0.821 versus 0.802).',
  'manuscript'))
 updates.append(replace_and_preserve(REVIEW/'ACADEMIC_REFEREE_REPORT.md',
   '\n\n**Editorial recommendation: MAJOR REVISIONS BEFORE SUBMISSION.**\n',
   '\n',
   'referee'))
 # Export exact revised standalone manuscripts separately, no V1 concatenation.
 exported=runpy.run_path(str(REVIEW/'export_current_review.py'))['export']
 exported(PAPER/'manuscript.md',PAPER/'LATEST_MANUSCRIPT_FOR_AVA.docx')
 exported(REVIEW/'ACADEMIC_REFEREE_REPORT.md',REVIEW/'SCIENTIFIC_REVIEW_REPORT.docx')
 paper_doc=Document(PAPER/'LATEST_MANUSCRIPT_FOR_AVA.docx')
 ref_doc=Document(REVIEW/'SCIENTIFIC_REVIEW_REPORT.docx')
 if len(paper_doc.tables)!=7 or len(paper_doc.inline_shapes)!=8:raise ValueError('Current standalone manuscript invalid')
 if len(ref_doc.tables)!=0 or len(ref_doc.inline_shapes)!=0:raise ValueError('Unexpected referee structure')
 manifest=REVIEW/'EVIDENCE_SHA256_MANIFEST.json'
 previous=manifest.read_bytes()
 backup=HERE/'BEFORE_FINAL_COPYEDIT_EVIDENCE_SHA256_MANIFEST.json'
 if backup.exists():raise FileExistsError('Evidence manifest already archived for copyedit')
 backup.write_bytes(previous)
 data=json.loads(previous.decode('utf-8'))
 targets={
  'paper/manuscript.md',
  'paper/LATEST_MANUSCRIPT_FOR_AVA.docx',
  'reports/robust_v2/external_academic_review_20261014/ACADEMIC_REFEREE_REPORT.md',
  'reports/robust_v2/external_academic_review_20261014/SCIENTIFIC_REVIEW_REPORT.docx',
 }
 found=set()
 for item in data['files']:
  if item['relative_path'] in targets:
   p=ROOT/item['relative_path']
   item['sha256']=sha(p)
   item['bytes']=p.stat().st_size
   found.add(item['relative_path'])
 if found!=targets:raise ValueError('Previously hashed Word/report evidence missing: '+repr(targets-found))
 manifest.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
 receipt={'schema':'scientific-manuscript-final-clarity-copyedit-v1',
   'edits':updates,'model_data_objectives_thresholds_or_solver_receipts_edited':False,
   'old_source_manifest_sha256':sha(backup),'new_source_manifest_sha256':sha(manifest),
   'final_manuscript_sha256':sha(PAPER/'manuscript.md'),
   'final_docx_sha256':sha(PAPER/'LATEST_MANUSCRIPT_FOR_AVA.docx'),
   'final_referee_sha256':sha(REVIEW/'SCIENTIFIC_REVIEW_REPORT.docx'),
   'remaining_gate':'Render and inspect the two revised Word documents; original figures, 29 bibliography entries and all numerical claims are preserved.'}
 output=HERE/'FINAL_COPYEDIT_RECEIPT.json'
 if output.exists():raise FileExistsError('Copyedit revision already exists')
 output.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 print('FINAL_ACADEMIC_COPYEDIT_COMPLETE',receipt['final_docx_sha256'],flush=True)
if __name__=='__main__':main()
