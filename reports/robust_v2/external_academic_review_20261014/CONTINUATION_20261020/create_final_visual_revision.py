from pathlib import Path
here=Path(__file__).resolve().parent
source=here/'prepare_visual_qa.py'
target=here/'prepare_visual_final.py'
if target.exists():raise FileExistsError('Final renderer source revision already exists')
text=source.read_text(encoding='utf-8')
old='RENDERED = REVIEW / "RENDERED"'
assert text.count(old)==1
text=text.replace(old,'RENDERED = REVIEW / "RENDERED_FINAL"')
old='REVIEW / "VISUAL_QA.json"'
assert text.count(old)==1
text=text.replace(old,'REVIEW / "VISUAL_QA_FINAL.json"')
target.write_text(text,encoding='utf-8')
print('FINAL_REVISION_PREPARED')