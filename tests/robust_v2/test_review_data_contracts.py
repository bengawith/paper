"""Regression contracts for geometry admission and calibration grouping."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robust_airfoil.data import builder, splits


def test_training_builder_rejects_a_crossing_cst_refit(tmp_path, coordinate_text, polar_text, monkeypatch):
    geometry = tmp_path / 'demo.dat'
    polar = tmp_path / 'demo-il-1000000.csv'
    geometry.write_text(coordinate_text, encoding='utf-8')
    polar.write_text(polar_text, encoding='utf-8')
    crossed = np.array([.2] * 5 + [-.2] * 5 + [0., 0.])
    monkeypatch.setattr(builder, 'fit_cst', lambda *args: SimpleNamespace(parameters=crossed, rms_error=0., max_error=0.))
    summary = builder.build_long_form_dataset([polar], [geometry], tmp_path/'out', 'fixture')
    assert summary['mapped_airfoils'] == 0
    assert summary['accepted_polar_rows'] == 0
    assert 'invalid_refitted_geometry' in (tmp_path/'out/rejected.json').read_text()


def test_calibration_groups_keep_near_duplicate_airfoils_together():
    records = pd.DataFrame({'airfoil_id': ['a', 'a-copy', 'b'], 'cluster_id': [8, 8, 9], 'split': ['calibration'] * 3})
    frame = pd.DataFrame({'airfoil_id': ['a', 'b', 'a-copy', 'a']})
    actual = splits.calibration_geometry_groups(frame, records)
    np.testing.assert_array_equal(actual, [8, 9, 8, 8])


@pytest.mark.parametrize('bad', ['unknown', 'development'])
def test_risk_audit_forbids_unmapped_or_non_calibration_data(bad):
    records = pd.DataFrame({'airfoil_id': ['a'], 'cluster_id': [8], 'split': [bad]})
    frame = pd.DataFrame({'airfoil_id': ['missing' if bad == 'unknown' else 'a']})
    with pytest.raises(ValueError):
        splits.calibration_geometry_groups(frame, records)
