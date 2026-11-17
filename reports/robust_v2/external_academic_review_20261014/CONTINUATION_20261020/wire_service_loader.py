"""Reconcile the concurrent service-field correction and wire its real consumers.

Preserve current file bytes first; exact content is rechecked immediately before
write. No controlled-label pinned source file or active process is modified.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent/'SERVICE_GRID_INTEGRATION'
HERE.mkdir(exist_ok=True)
records=[]
for name in ['run_repaired_production.py','profile_repaired_evaluator.py']:
    path=ROOT/'scripts/robust_v2'/name
    original=path.read_bytes()
    text=original.decode('utf-8')
    old='from robust_airfoil.research_lineage import resolve_lineage'
    assert text.count(old)==1
    updated=text.replace(old,'from robust_airfoil.research_lineage import load_optimisation_for_lineage, resolve_lineage')
    pattern=r"opt\s*=\s*OptimisationConfig\.model_validate\(load_yaml\(ROOT\s*/\s*'configs/robust_v2/optimisation.yaml'\)\)"
    updated,count=re.subn(pattern,'opt = load_optimisation_for_lineage(ROOT, args.lineage)',updated)
    assert count==1,(name,count)
    (HERE/(name+'.before')).write_bytes(original)
    if path.read_bytes()!=original:
        raise RuntimeError('Concurrent source edit detected before service integration: '+name)
    path.write_text(updated,encoding='utf-8')
    records.append({'path':path.relative_to(ROOT).as_posix(),'before_sha256':hashlib.sha256(original).hexdigest(),'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
# The added unused import belongs to this session. Retain the other session's
# valid Config field and validator exactly; remove only our redundant import.
path=ROOT/'src/robust_airfoil/config.py'
original=path.read_bytes()
text=original.decode('utf-8')
if text.count('from math import isfinite\n')==1 and text.count('isfinite')==2 and 'np.isfinite(values)' in text:
    (HERE/'config.py.before').write_bytes(original)
    if path.read_bytes()!=original:
        raise RuntimeError('Concurrent config edit detected')
    path.write_text(text.replace('from math import isfinite\n',''),encoding='utf-8')
    records.append({'path':path.relative_to(ROOT).as_posix(),'before_sha256':hashlib.sha256(original).hexdigest(),'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Removed this session unused import only; preserved concurrent service-grid implementation'})
(HERE/'EDIT_RECEIPT.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('SERVICE_GRID_ACTUAL_CONSUMERS_CONNECTED',flush=True)
