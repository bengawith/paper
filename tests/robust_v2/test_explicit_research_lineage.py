"""Historical and controlled-coordinate runs must never share output locations."""
import runpy

import pytest

from robust_airfoil.constants import ROOT


def resolver():
    path = ROOT/'src/robust_airfoil/research_lineage.py'
    assert path.is_file(), 'Explicit research-lineage resolution is missing'
    return runpy.run_path(str(path))['resolve_lineage']


def test_controlled_results_data_and_reports_are_isolated_from_historical_rebuild(tmp_path):
    resolve = resolver()
    old = resolve(tmp_path,'rebuilt')
    new = resolve(tmp_path,'controlled')
    assert old['results'] != new['results'] and old['data'] != new['data'] and old['reports'] != new['reports']
    assert new['results'] == tmp_path/'results/robust_v2/lineages/robust-v2-controlled-20261020'
    assert new['data'] == tmp_path/'data/robust_v2/processed/controlled_20261020'
    assert new['reports'].name == 'CONTROLLED_MODEL'
    assert not any(tmp_path.iterdir()), 'Resolving paths must not create or overwrite artifacts'


@pytest.mark.parametrize('bad', ['../current', 'robust-v2-current-20260908', '', 'new-favourite-run'])
def test_unknown_or_pathlike_lineage_is_rejected(tmp_path,bad):
    with pytest.raises(ValueError):
        resolver()(tmp_path,bad)


@pytest.mark.parametrize('filename',['train_repaired_lineage.py','evaluate_repaired_lineage.py','audit_repaired_risk.py','run_repaired_production.py'])
def test_real_scripts_expose_explicit_controlled_lineage_selection(filename):
    text = (ROOT/'scripts/robust_v2'/filename).read_text(encoding='utf-8')
    assert '--lineage' in text and 'resolve_lineage(' in text
