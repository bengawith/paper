"""Version the import-only execution repair without changing or erasing old cases."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
p=ROOT/'scripts/robust_v2/generate_controlled_corpus.py'
t=p.read_text(encoding='utf-8')
assert '--execution-revision' not in t

def replace(old,new):
    global t
    assert t.count(old)==1,(old[:100],t.count(old))
    t=t.replace(old,new)

replace('import argparse\n','import argparse\nimport ast\nimport hashlib\n')
replace('from robust_airfoil.modelling.dataset import FEATURE_COLUMNS','from robust_airfoil.constants import FEATURE_COLUMNS')
replace('def context() -> dict[str, Any]:','def context(execution_revision: int = 1) -> dict[str, Any]:')
replace("    source_files = ['geometry/cst.py'","    source_files = ['constants.py','geometry/cst.py'")
replace("    return {'schema':'exact-cst-local-xfoil-corpus-v1'","    identity = {'schema':'exact-cst-local-xfoil-corpus-v1'")
anchor="            'geometry_domain':'All1256 previously admitted12-parameter CST reconstructions, not claimed identical to original source coordinates'}\n"
replace(anchor,anchor+'''    identity['execution_revision'] = execution_revision
    if execution_revision == 2:
        proof_path = ROOT/'reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/RUNTIME_MEMORY_REPAIR/COMPATIBILITY_PROOF.json'
        identity['import_only_compatibility_proof_sha256'] = sha256_file(proof_path)
    return identity


def verified_compatibility() -> dict[str, Any]:
    proof_path = ROOT/'reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/RUNTIME_MEMORY_REPAIR/COMPATIBILITY_PROOF.json'
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    parent_path = LINEAGE/'LABEL_PROTOCOL.json'
    if sha256_file(parent_path) != proof['parent_protocol_sha256']:
        raise ValueError('The preserved parent label protocol changed')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    if proof['parent_context_hash'] != parent['context_hash']:
        raise ValueError('Wrong parent label context')
    tree = ast.parse((ROOT/'src/robust_airfoil/validation/xfoil_runner.py').read_text(encoding='utf-8'))
    actual = {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
              for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
              and node.name in proof['pure_solver_ast_sha256']}
    if actual != proof['pure_solver_ast_sha256']:
        raise ValueError('Pure solver calculation changed; old cases cannot be adopted')
    tree = ast.parse((ROOT/'src/robust_airfoil/constants.py').read_text(encoding='utf-8'))
    features = {node.targets[0].id:hashlib.sha256(ast.dump(node.value,include_attributes=False).encode()).hexdigest()
                for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id in proof['feature_contract_ast_sha256']}
    if features != proof['feature_contract_ast_sha256']:
        raise ValueError('Input feature order changed')
    for name,expected in parent['context']['scientific_source_sha256'].items():
        if name != 'validation/xfoil_runner.py' and sha256_file(ROOT/'src/robust_airfoil'/name) != expected:
            raise ValueError('Unapproved label calculation change: '+name)
    return proof
''')
replace('def verify_receipt(directory: Path, expected_context: str) -> dict[str, Any]:',
        'def verify_receipt(directory: Path, expected_context: str, compatibility: dict[str, Any] | None = None) -> dict[str, Any]:')
replace("    if record['context_hash'] != expected_context:\n        raise ValueError('Stored solver case belongs to a different scientific context')",
'''    if record['context_hash'] != expected_context:
        if (compatibility is None or record['context_hash'] != compatibility['parent_context_hash']
                or compatibility['compatible_existing_receipts_sha256'].get(directory.name) != sha256_file(directory/'CASE_RECEIPT.json')):
            raise ValueError('Stored solver case belongs to an unverified different scientific context')''')
replace("        return verify_receipt(directory,protocol['context_hash'])","        return verify_receipt(directory,protocol['context_hash'],protocol.get('compatibility'))")
replace("    parser.add_argument('--workers',type=int,choices=[1,2],default=2)",
        "    parser.add_argument('--workers',type=int,choices=[1,2],default=2)\n    parser.add_argument('--execution-revision',type=int,choices=[1,2],default=1)")
replace('    current=context()\n    protocol_path=LINEAGE/\'LABEL_PROTOCOL.json\'',
        "    compatibility=verified_compatibility() if args.execution_revision == 2 else None\n    current=context(args.execution_revision)\n    protocol_path=LINEAGE/('LABEL_PROTOCOL_R2.json' if args.execution_revision == 2 else 'LABEL_PROTOCOL.json')")
replace("        protocol={'context':current,'context_hash':hash_object(current),'created_utc':utc_now(),",
        "        protocol={'context':current,'context_hash':hash_object(current),'created_utc':utc_now(),'compatibility':compatibility,")
replace("            records.append(verify_receipt(case,protocol['context_hash']))",
        "            records.append(verify_receipt(case,protocol['context_hash'],protocol.get('compatibility')))")
replace("    if len(records)!=len(shapes) or hash_object(context())!=protocol['context_hash']:",
        "    if len(records)!=len(shapes) or hash_object(context(args.execution_revision))!=protocol['context_hash']:")
replace("             'label_protocol_sha256':sha256_file(protocol_path)",
        "             'label_protocol_file':protocol_path.name,'execution_revision':args.execution_revision,'compatible_prior_case_count':len(compatibility['compatible_existing_receipts_sha256']) if compatibility else 0,\n             'label_protocol_sha256':sha256_file(protocol_path)")
p.write_text(t,encoding='utf-8')
print('EXPLICIT_R2_LABEL_EXECUTION_CONNECTED')
