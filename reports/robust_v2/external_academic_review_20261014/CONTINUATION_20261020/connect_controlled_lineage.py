"""Connect existing real training/evaluation consumers to an explicit revision CLI."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
SCRIPTS=ROOT/'scripts/robust_v2'
for name in ['train_repaired_lineage.py','evaluate_repaired_lineage.py','audit_repaired_risk.py','profile_repaired_evaluator.py','run_repaired_production.py']:
    path=SCRIPTS/name
    text=path.read_text(encoding='utf-8')
    assert 'resolve_lineage(' not in text, 'Do not apply the revision routing patch twice'
    if 'import argparse\n' not in text:
        anchor='from __future__ import annotations\n'
        assert text.count(anchor)==1
        text=text.replace(anchor,anchor+'\nimport argparse\n')
    anchor='from robust_airfoil.hashing import '
    start=text.index(anchor)
    end=text.index('\n',start)
    text=text[:end+1]+'from robust_airfoil.research_lineage import resolve_lineage\n'+text[end+1:]
    settings="    parser.add_argument('--lineage', choices=['rebuilt','controlled'], default='rebuilt', help='Explicit immutable scientific revision; never overwrites the other lineage')\n    args=parser.parse_args()\n    paths=resolve_lineage(ROOT,args.lineage)\n    LINEAGE=paths['results']\n"
    if name=='run_repaired_production.py':
        assert text.count('    args=parser.parse_args()\n')==1
        text=text.replace('    args=parser.parse_args()\n',settings+"    REVIEW=paths['reports']\n    OUTPUT=LINEAGE/'optimisation_production'\n")
        text=text.replace("ROOT/'data/robust_v2/processed/rebuilt_20261014/airfoils.parquet'","paths['data']/'airfoils.parquet'")
    else:
        local = "    DATA=paths['data']\n    REVIEW=paths['reports']\n" if name=='train_repaired_lineage.py' else "    OUT=paths['reports']\n"
        if name=='audit_repaired_risk.py':
            local="    OUT=paths['reports']/'RISK_OOF'\n"
        anchor='def main() -> None:\n'
        assert text.count(anchor)==1
        text=text.replace(anchor,anchor+'    parser=argparse.ArgumentParser()\n'+settings+local)
        if name=='profile_repaired_evaluator.py':
            text=text.replace("ROOT/'data/robust_v2/processed/rebuilt_20261014/airfoils.parquet'","paths['data']/'airfoils.parquet'")
    if name=='evaluate_repaired_lineage.py':
        text=text.replace("{'lineage':'robust-v2-rebuilt-20261014',","{'lineage':LINEAGE.name,'label_source':'actual local XFOIL on exact represented CST geometry' if args.lineage=='controlled' else 'archived source polars',")
        text=text.replace("'Historical test previously inspected in earlier work, not pristine prospective evidence.'","'Geometry identities were previously inspected; the protected cohort is not pristine prospective sampling. No new test outcomes were used to fit or stop the models.'")
    if name=='train_repaired_lineage.py':
        text=text.replace("'member_count':5,'base_seed'","'lineage':LINEAGE.name,'member_count':5,'base_seed'")
    path.write_text(text,encoding='utf-8')
    print('CONNECTED',name,flush=True)
