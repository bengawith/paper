from pathlib import Path
import json,hashlib
here=Path(__file__).resolve().parent
review=here.parent
root=here.parents[3]
source=here/'prepare_visual_final.py'
target=here/'prepare_visual_v3.py'
if target.exists():raise FileExistsError('Third visual QA rendering revision already exists')
text=source.read_text(encoding='utf-8')
old='RENDERED = REVIEW / "RENDERED_FINAL"'
assert text.count(old)==1
text=text.replace(old,'RENDERED = REVIEW / "RENDERED_V3"')
old='REVIEW / "VISUAL_QA_FINAL.json"'
assert text.count(old)==1
text=text.replace(old,'REVIEW / "VISUAL_QA_V3.json"')
target.write_text(text,encoding='utf-8')
# runpy's exporter refreshed the referee too; bind the genuinely current bytes.
manifest=review/'EVIDENCE_SHA256_MANIFEST.json'
backup=here/'BEFORE_VISUAL_V3_REFEREE_HASH_UPDATE.json'
if backup.exists():raise FileExistsError('Third referee manifest backup exists')
backup.write_bytes(manifest.read_bytes())
data=json.loads(manifest.read_text(encoding='utf-8'))
ref=review/'SCIENTIFIC_REVIEW_REPORT.docx'
hits=0
for rec in data['files']:
 if rec['relative_path']=='reports/robust_v2/external_academic_review_20261014/SCIENTIFIC_REVIEW_REPORT.docx':
  rec['sha256']=hashlib.sha256(ref.read_bytes()).hexdigest()
  rec['bytes']=ref.stat().st_size
  hits+=1
assert hits==1
manifest.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print('VISUAL_V3_PREPARED_AND_REFEREE_HASH_BOUND')
