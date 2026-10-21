"""Accept exact DOCX-derived Chromium page inspection with honest native-Word limit.

The reviewer visually inspected contact sheets 01-03 (14 manuscript pages)
and 01 (one referee page). No overflow/clipping/blank pages or missing figures
observed. The first abstract was shortened after identifying a split,
and a one-line second referee page was removed; prior renders preserved.
This marks only the *browser-derived* page preview, NOT native Microsoft Word.
"""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
REVIEW=ROOT/'reports/robust_v2/external_academic_review_20261014'
HERE=Path(__file__).resolve().parent
LATEST=REVIEW/'VISUAL_QA_V3.json'
ACCEPTED=REVIEW/'VISUAL_QA.json'
BACKUP=HERE/'VISUAL_QA_PRE_V3_ACCEPTANCE.json'
RENDER=REVIEW/'RENDERED_V3'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def main():
 if BACKUP.exists():raise FileExistsError('Visual QA acceptance already recorded in this revision')
 data=json.loads(LATEST.read_text(encoding='utf-8'))
 if data['all_pages_visually_inspected'] or not data['native_microsoft_word_render_unavailable']:
  raise ValueError('This acceptance is solely for actual browser-preview pages')
 documents={
  'manuscript':ROOT/'paper/LATEST_MANUSCRIPT_FOR_AVA.docx',
  'referee':REVIEW/'SCIENTIFIC_REVIEW_REPORT.docx',
 }
 counts={'manuscript':14,'referee':1}
 checks=[]
 for name,source in documents.items():
  section=data[name]
  if section['page_count']!=counts[name] or len(section['pages'])!=counts[name]:
   raise ValueError('Rendered page count differs from inspected contact sheets')
  if section['docx_sha256']!=sha(source):
   raise ValueError('Word source updated after page rendering: '+name)
  for page in section['pages']:
   image=ROOT/page['image']
   if not image.is_file() or image.stat().st_size<4000:
    raise ValueError('Missing or blank preview page image: '+str(image))
   if page['preview_white_fraction_approx']>.97:
    raise ValueError('Suspiciously empty rendered page: '+str(image))
   page['visually_inspected']=True
   checks.append({'source':name,'page':page['page'],'preview_image_sha256':sha(image),
      'visual_result':'PAGE_PRESENT_NO_DETECTED_CLIPPING_OR_BLANKNESS'})
  sheets=list(RENDER.glob(name+'-contact-sheet-*.png'))
  if len(sheets)!=(counts[name]+5)//6:raise ValueError('Contact sheets are incomplete: '+name)
 data.update(all_pages_visually_inspected=True,
   native_word_page_parity_proven=False,
   inspection_scope='All 14 manuscript and one referee page manually inspected through actual Chromium DOCX-derived contact sheets; not native Microsoft Word pagination.',
   visual_defects_corrected=['First abstract split across two pages in v2; condensed without changing scientific data and confirmed entire abstract on v3 page1.',
    'Single orphaned editorial recommendation on referee page2; removed repeated sentence and confirmed one-page referee.'],
   unresolved_native_layout_caveat='True Microsoft Word page rendering was unavailable/stalled; recipient should check in Word prior to journal submission.',
   per_page_sha256=checks)
 if ACCEPTED.exists():shutil.copy2(ACCEPTED,BACKUP)
 ACCEPTED.write_text(json.dumps(data,indent=2),encoding='utf-8')
 LATEST.write_text(json.dumps(data,indent=2),encoding='utf-8')
 print('FINAL_BROWSER_PREVIEW_VISUALLY_INSPECTED',len(checks),'pages; NO NATIVE_WORD_CLAIM',flush=True)
if __name__=='__main__':main()
