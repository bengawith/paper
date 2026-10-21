"""Revision-five isolated visual-layout experiment; never overrides first previews.

Creates a more Word-typographic Chromium reconstruction from *unchanged*
generated DOCX bytes and retains complete SHA and page-level evidence. This
is not a native Microsoft Word render or physical research validation.
"""
from __future__ import annotations
import hashlib
import json
import runpy
import subprocess
import tempfile
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=HERE/'VISUAL_LAYOUT_R6'
ORIGINAL=HERE/'render_author_docx_browser.py'
GEN=runpy.run_path(str(ORIGINAL))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def render(name,source):
 output=OUT/f'{name}.r6-browser-preview.pdf'
 if output.exists():raise FileExistsError(str(output))
 markup,figures,tables,paragraphs=GEN['html_doc'](source)
 if 'font:10.5pt/1.3' not in markup:raise ValueError('Original layout font changed')
 markup=markup.replace("font:10.5pt/1.3 'Aptos'","font:10pt/1.12 'Aptos'")
 markup=markup.replace('font-size:9pt;margin:7pt 0 10pt','font-size:8pt;margin:6pt 0 9pt')
 markup=markup.replace('p{margin:0 0 6pt','p{margin:0 0 5pt')
 styles="""
 @media print{
  p:has(+table),p:has(+p > img.figure){break-after:avoid-page;page-break-after:avoid}
  p:has(>img.figure){break-inside:avoid-page;page-break-inside:avoid}
  table{break-inside:avoid-page;page-break-inside:avoid}
  tr{break-inside:avoid-page}
  h2,h3,h4{break-after:avoid-page}
 }
 """
 markup=markup.replace('</style>',styles+'</style>',1)
 with tempfile.TemporaryDirectory(prefix='aerofoil-R5-visual-') as tmp:
  path=Path(tmp);file=path/'source.html';file.write_text(markup,encoding='utf-8')
  command=[str(GEN['CHROME']),'--headless=new','--disable-gpu','--no-first-run',
           '--disable-extensions','--no-default-browser-check','--no-pdf-header-footer',
           '--user-data-dir='+str(path/'chrome-profile'),
           '--print-to-pdf='+str(output),file.as_uri()]
  p=subprocess.run(command,capture_output=True,text=True,timeout=120)
  if p.returncode or not output.is_file():raise RuntimeError(str(p.returncode)+' '+p.stderr[-1000:])
 pdf=pdfium.PdfDocument(str(output))
 pages=[];thumbs=[]
 for ix,page in enumerate(pdf):
  text=page.get_textpage().get_text_range()
  img=page.render(scale=1.3).to_pil().convert('RGB')
  filename=OUT/f'{name}-r6-page-{ix+1:02d}.png'
  img.save(filename,optimize=True)
  pages.append({'page':ix+1,'filename':filename.name,'text_character_count':len(text),'image_sha256':sha(filename)})
  thumb=img.copy();thumb.thumbnail((500,720));thumbs.append(thumb)
 for ix in range(0,len(thumbs),6):
  board=Image.new('RGB',(1540,1570),'#e4e7eb');draw=ImageDraw.Draw(board)
  for j,img in enumerate(thumbs[ix:ix+6]):
   x=10+(j%3)*510+(500-img.width)//2;y=30+(j//3)*770
   board.paste(img,(x,y))
   draw.text((15+(j%3)*510,10+(j//3)*770),f'{name} R5 page {ix+j+1}',fill='black')
  board.save(OUT/f'{name}-r6-contact-{ix//6+1:02d}.png',optimize=True)
 return {'word':source.relative_to(ROOT).as_posix(),'word_sha256':sha(source),
         'preview_sha256':sha(output),'page_count':len(pages),'figures':figures,
         'tables':tables,'paragraphs':paragraphs,'pages':pages,
         'native_microsoft_word_render':False,'font_matches_docx_better':True,
         'limitations':'New Chromium approximation; Word headers/footers/page breaks are not reproduced. No human Word inspection.'}
def main():
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Prior R5 evidence exists, preserve for review')
 OUT.mkdir(parents=True,exist_ok=True)
 docs=[('manuscript',ROOT/'paper/LATEST_MANUSCRIPT_FOR_AVA.docx'),
       ('referee',ROOT/'reports/robust_v2/external_academic_review_20261014/SCIENTIFIC_REVIEW_REPORT.docx')]
 records={name:render(name,path) for name,path in docs}
 (OUT/'R6_PROXY_EVIDENCE.json').write_text(json.dumps({'schema':'independent-preview-layout-adjustment-r5',
   'original_visual_qa_sha256':sha(ROOT/'reports/robust_v2/external_academic_review_20261014/VISUAL_QA.json'),
   'reason':'Original browser visual preview orphaned one referee sentence and detached table/figure captions',
   'results':records,'scientific_source_changed':False,'word_source_changed':False,
   'native_word_render':False,'visual_review_pending':True},indent=2),encoding='utf-8')
 print('R6_BROWSER_ONLY',[(name,r['page_count'],r['pages'][-1]['text_character_count']) for name,r in records.items()],flush=True)
if __name__=='__main__':main()
