"""Resolve double-CR script files before scientific production starts.

Only two non-imported process entrypoints have CRCRLF; preserve exact old
bytes, prove both ASTs unchanged, do not touch already frozen source/modules.
"""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
FILES=[
 'scripts/robust_v2/run_repaired_production.py',
 'scripts/robust_v2/profile_repaired_evaluator.py',
]
OUT=HERE/'EOL_REPAIR_SOURCE_PRESERVED'
OUT.mkdir(exist_ok=True)
rows=[]
for name in FILES:
 p=ROOT/name
 raw=p.read_bytes()
 count=raw.count(bytes([13,13,10]))
 if count<5:
  raise ValueError('Double-CR line endings not present as expected: '+name)
 after=raw.replace(bytes([13,13,10]),bytes([10]))
 old_tree=ast.parse(raw.decode('utf-8'))
 new_tree=ast.parse(after.decode('utf-8'))
 # Double CR is also embedded in the module docstring. The only allowed AST
 # change is removing artificial blank lines from that non-executable string.
 if isinstance(old_tree.body[0],ast.Expr) and isinstance(old_tree.body[0].value,ast.Constant):
  old_tree.body[0].value.value=old_tree.body[0].value.value.replace('\n\n','\n')
 old_ast=ast.dump(old_tree,include_attributes=False)
 new_ast=ast.dump(new_tree,include_attributes=False)
 if old_ast!=new_ast:
  raise ValueError('An executable AST node changed, refusing EOL rewrite: '+name)
 backup=OUT/Path(name).name
 if backup.exists():
  raise FileExistsError('Preserved original bytes already exist: '+str(backup))
 backup.write_bytes(raw)
 p.write_bytes(after)
 rows.append({'path':name,'double_CRLF_converted':count,'original_sha256':hashlib.sha256(raw).hexdigest(),
              'new_sha256':hashlib.sha256(after).hexdigest(),'AST_identical':True,
              'before_byte_for_byte_backup':backup.relative_to(ROOT).as_posix()})
manifest=HERE/'EOL_REPAIR_MANIFEST.json'
if manifest.exists():
 raise FileExistsError('EOL repair manifest exists; do not repeat')
manifest.write_text(json.dumps({'schema':'entrypoint-only-ast-preserving-eol-repair-v1',
 'scientific_library_source_files_changed':False,'old_production_preflight': 'Completed on equivalent parsed script before this cosmetic correction',
 'rows':rows},indent=2),encoding='utf-8')
print('ENTRYPOINT_EOL_REPAIR_AST_IDENTICAL',len(rows))
