"""Remove only trailing blank line; bind unchanged Word semantic content to source."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent
MANUSCRIPT=ROOT/'paper/manuscript.md'
EVIDENCE=REVIEW/'EVIDENCE_SHA256_MANIFEST.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 original=MANUSCRIPT.read_bytes()
 if not original.endswith(b'\n\n') or original.endswith(b'\n\n\n'):
  raise ValueError('Unexpected source; do not rewrite manuscript ending')
 back=HERE/'BEFORE_EOF_WHITESPACE_V4.md'
 manifest_old=HERE/'BEFORE_EOF_WHITESPACE_V4_EVIDENCE.json'
 record=HERE/'MANUSCRIPT_EOF_ONLY_V4_RECEIPT.json'
 if any(p.exists() for p in [back,manifest_old,record]):
  raise FileExistsError('EOF correction already applied')
 back.write_bytes(original)
 manifest_old.write_bytes(EVIDENCE.read_bytes())
 MANUSCRIPT.write_bytes(original[:-1])
 data=json.loads(EVIDENCE.read_text(encoding='utf-8'))
 count=0
 for item in data['files']:
  if item['relative_path']=='paper/manuscript.md':
   item['sha256']=sha(MANUSCRIPT)
   item['bytes']=MANUSCRIPT.stat().st_size
   count+=1
 if count!=1:raise ValueError('Manuscript evidence entry absent')
 EVIDENCE.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
 record.write_text(json.dumps({
  'schema':'manuscript-EOF-blank-line-only-v4',
  'before_sha256':sha(back),'after_sha256':sha(MANUSCRIPT),
  'old_evidence_sha256':sha(manifest_old),'new_evidence_sha256':sha(EVIDENCE),
  'bytes_removed':1,'removed_byte':'0A','nonwhitespace_changed':False,
  'underlying_accepted_word_docx_content_not_modified':True,
  'source_text_equivalent_after_rstrip_newlines':back.read_text(encoding='utf-8').rstrip('\n')==MANUSCRIPT.read_text(encoding='utf-8').rstrip('\n'),
  'conclusions_or_numeric_evidence_changed':False},indent=2),encoding='utf-8')
 print('ONLY_FINAL_BLANK_LINE_REMOVED',sha(MANUSCRIPT))
if __name__=='__main__':main()
