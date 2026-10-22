"""Remove the measured solver-only training-framework dependency.

Preserve the exact old driver and prove the pure solver path/feature contract
unchanged. The original protocol and all completed case receipts are read-only.
"""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent/'RUNTIME_MEMORY_REPAIR'
HERE.mkdir(exist_ok=True)
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
paths=['src/robust_airfoil/validation/xfoil_runner.py','src/robust_airfoil/modelling/dataset.py','src/robust_airfoil/constants.py','scripts/robust_v2/generate_controlled_corpus.py']
before={}
for relative in paths:
    path=ROOT/relative
    before[relative]=path.read_text(encoding='utf-8')
    target=HERE/'before'/relative
    if target.exists():
        raise FileExistsError('Import repair already has a baseline; inspect rather than reapply')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(path.read_bytes())

old_protocol=json.loads((LINEAGE/'LABEL_PROTOCOL.json').read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(ROOT/paths[0]) == old_protocol['context']['scientific_source_sha256']['validation/xfoil_runner.py']
assert sha(ROOT/paths[3]) == old_protocol['context']['generator_sha256']

# Move the existing expressions verbatim; dataset continues to re-export the
# same public contracts for training consumers.
dataset=before[paths[1]]
feature_line=next(line for line in dataset.splitlines() if line.startswith('FEATURE_COLUMNS ='))
target_line=next(line for line in dataset.splitlines() if line.startswith('TARGET_COLUMNS ='))
constants=before[paths[2]]+'\n\n'+feature_line+'\n'+target_line+'\n'
(ROOT/paths[2]).write_text(constants,encoding='utf-8')
dataset=dataset.replace(feature_line+'\n'+target_line,'from robust_airfoil.constants import FEATURE_COLUMNS, TARGET_COLUMNS')
(ROOT/paths[1]).write_text(dataset,encoding='utf-8')

source=before[paths[0]]
source=source.replace('from typing import Any','from typing import TYPE_CHECKING, Any')
source=source.replace('from robust_airfoil.modelling.calibrate import apply_trust_model\n','')
source=source.replace('from robust_airfoil.modelling.dataset import FEATURE_COLUMNS','from robust_airfoil.constants import FEATURE_COLUMNS')
source=source.replace('from robust_airfoil.modelling.ensemble import LoadedEnsemble\n','')
source=source.replace('from robust_airfoil.optimisation.objectives import weighted_required_lift_drag\n','')
anchor='\n\n@dataclass(frozen=True)\nclass XFoilCase:'
assert source.count(anchor)==1
source=source.replace(anchor,'\n\nif TYPE_CHECKING:\n    from robust_airfoil.modelling.ensemble import LoadedEnsemble\n'+anchor)
anchor='def _weighted_drag(points: pd.DataFrame, optimisation: OptimisationConfig) -> float | None:\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'    from robust_airfoil.optimisation.objectives import weighted_required_lift_drag\n\n')
start=source.index('def _surrogate_comparison(')
pos=source.index(') -> dict[str, object]:\n',start)+len(') -> dict[str, object]:\n')
source=source[:pos]+'    from robust_airfoil.modelling.calibrate import apply_trust_model\n\n'+source[pos:]
start=source.index('def run_candidate_validation(')
pos=source.index(') -> dict[str, Any]:\n',start)+len(') -> dict[str, Any]:\n')
source=source[:pos]+'    from robust_airfoil.modelling.ensemble import LoadedEnsemble\n\n'+source[pos:]
(ROOT/paths[0]).write_text(source,encoding='utf-8')

solver_names=['XFoilCase','XFoilResult','build_xfoil_commands','_requested_alpha_grid','_sweep_completion','_segment_status','_run_xfoil_segment']
def nodes(text,names):
    tree=ast.parse(text)
    return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}
old_nodes=nodes(before[paths[0]],solver_names)
new_nodes=nodes(source,solver_names)
assert set(old_nodes)==set(solver_names) and old_nodes==new_nodes
old_contract={node.targets[0].id:ast.dump(node.value,include_attributes=False) for node in ast.parse(before[paths[1]]).body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['FEATURE_COLUMNS','TARGET_COLUMNS']}
new_contract={node.targets[0].id:ast.dump(node.value,include_attributes=False) for node in ast.parse(constants).body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['FEATURE_COLUMNS','TARGET_COLUMNS']}
assert old_contract==new_contract
for name,expected in old_protocol['context']['scientific_source_sha256'].items():
    if name!='validation/xfoil_runner.py':
        assert sha(ROOT/'src/robust_airfoil'/name)==expected, name
receipts={p.parent.name:sha(p) for p in (LINEAGE/'solver_labels').glob('*/CASE_RECEIPT.json')}
proof={'schema':'solver-import-only-equivalence-proof-v1',
       'parent_protocol_sha256':sha(LINEAGE/'LABEL_PROTOCOL.json'),
       'parent_context_hash':old_protocol['context_hash'],
       'compatible_existing_receipts_sha256':receipts,
       'pure_solver_ast_unchanged':True,'feature_contract_ast_unchanged':True,
       'pure_solver_ast_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in old_nodes.items()},
       'feature_contract_ast_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in old_contract.items()},
       'preserved_baseline_files_sha256':{relative:sha(HERE/'before'/relative) for relative in paths},
       'ruling':'Import/dependency-only execution repair, not a physical or label-admission change. Keep the3GiB/20GiB floors. Existing case bytes and original protocol are not rewritten.'}
(HERE/'COMPATIBILITY_PROOF.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print('SOLVER_IMPORT_BOUNDARY_REPAIRED',len(receipts),'preserved case receipts',flush=True)
