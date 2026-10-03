"""Software-only arithmetic/format fixtures; none are aerodynamic training data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import robust_airfoil.pipeline as pipeline
from robust_airfoil.data.polar_parser import parse_polar_text, validate_polar_conditions
from robust_airfoil.data.splits import assert_locked_test_access
from robust_airfoil.modelling.evaluate import cvar
from robust_airfoil.modelling.freeze import freeze_artifacts, verify_frozen_artifacts
from robust_airfoil.modelling.train import _batch_loss
from robust_airfoil.optimisation.objectives import (
    ava_lift_sensitivity,
    required_lift_drag,
    robust_drag_objectives,
)
from robust_airfoil.run_state import RunState
from robust_airfoil.validation.xfoil_runner import XFoilCase, _segment_status, _sweep_completion


def case(start=0.0, end=2.0, step=1.0):
    return XFoilCase(Path('unused.dat'), 1000000.0, 0.0, 9.0, start, end, step, 70, 'test')

def test_lift_reward_does_not_cancel_itself():
    w = np.full(5, 0.2)
    weak = ava_lift_sensitivity(np.ones(5), np.zeros(5), w, 0)
    strong = ava_lift_sensitivity(np.full(5, 2.0), np.zeros(5), w, 0)
    assert strong < weak
    assert strong == pytest.approx(-2.0)

def test_ava_scales_are_fixed_configuration():
    assert ava_lift_sensitivity(np.array([2.0]), np.array([0.1]), np.array([1.0]), 2.0, lift_scale=2.0, sensitivity_scale=0.1) == pytest.approx(1.0)

@pytest.mark.parametrize('values,q,expected', [([0, 0, 0, 0, 10], 0.5, 4.0), ([0, 0, 10], 0.5, 20 / 3), ([0, 1, 2, 3, 4], 0.5, 3.2), ([0, 1, 2], 0.0, 1.0), ([3, 3, 3], 0.95, 3.0)])
def test_empirical_cvar_keeps_fractional_mass(values, q, expected):
    assert cvar(np.array(values), q) == pytest.approx(expected)

@pytest.mark.parametrize('values', [[], [1, np.nan], [1, np.inf]])
def test_bad_samples_cannot_be_silently_dropped(values):
    with pytest.raises(ValueError):
        cvar(np.array(values), 0.95)
    assert np.isinf(robust_drag_objectives(np.array(values))['expected_weighted_cd'])

def test_poststall_only_rising_crossing_is_infeasible():
    assert np.isinf(required_lift_drag(np.array([0, 1, 2, 3.0]), np.array([0, 0.5, 0.2, 1.0]), np.full(4, 0.01), 0.8))

def test_nan_gap_is_not_interpolated_over():
    assert np.isinf(required_lift_drag(np.array([0, 1, 2.0]), np.array([0, np.nan, 1.0]), np.full(3, 0.01), 0.5))

def test_duplicate_alpha_conflict_is_not_arbitrarily_selected():
    assert np.isinf(required_lift_drag(np.array([0, 1, 1, 2.0]), np.array([0, 0.5, 0.7, 1.0]), np.full(4, 0.01), 0.6))

def test_identical_duplicate_and_exact_root_work():
    assert required_lift_drag(np.array([0, 1, 1, 2.0]), np.array([0, 0.5, 0.5, 1.0]), np.full(4, 0.01), 0.5) == pytest.approx(0.01)

def test_normal_prepeak_crossing_is_valid_even_with_later_descent():
    assert required_lift_drag(np.array([0, 1, 2, 3.0]), np.array([0, 0.5, 1.0, 0.2]), np.full(4, 0.01), 0.6) == pytest.approx(0.01)

@pytest.mark.parametrize('alphas', [[0.1, 1.1, 2.1], [0, 0.5, 2], [0, 1, 1.5]])
def test_xfoil_row_count_cannot_imply_target_coverage(alphas):
    expected, coverage, _ = _sweep_completion(pd.DataFrame({'alpha_deg': alphas}), case())
    assert expected == 3 and coverage < 1
    assert _segment_status(0, pd.DataFrame({'alpha_deg': alphas}), case())[0] == 'failed'

def test_xfoil_correct_coverage_cannot_hide_bad_exit():
    assert _segment_status(1, pd.DataFrame({'alpha_deg': [0, 1, 2]}), case())[0] == 'failed'

def test_xfoil_duplicate_extra_rows_do_not_increase_coverage():
    assert _sweep_completion(pd.DataFrame({'alpha_deg': [0, 0, 1, 1, 2, 2, 9]}), case()) == (3, 1.0, True)

def test_xfoil_descending_sweep():
    assert _sweep_completion(pd.DataFrame({'alpha_deg': [2, 1, 0]}), case(2, 0, -1)) == (3, 1.0, True)

def test_source_csv_metadata_and_empty_values_preserve_columns():
    text = 'Polar key,xf-fixture-il-1000000\nAirfoil,fixture-il\nReynolds number,1000000\nNcrit,9\nMach,0\nAlpha,Cl,Cd,Cdp,Cm,Top_Xtr,Bot_Xtr\n0,.2,,.004,-.05,.8,.9\n1,.3,.01,.005,-.05,.8,.9\n'
    parsed = parse_polar_text(text)
    validate_polar_conditions(parsed.metadata)
    assert np.isnan(parsed.points.loc[0, 'cd'])
    assert parsed.points.loc[0, 'cm'] == pytest.approx(-0.05)
    assert parsed.metadata['polar_key'] == 'xf-fixture-il-1000000'

def test_xfoil_spaced_exponents_are_parsed():
    text = 'Re = 1.000 e 6 Mach = 0.000 Ncrit = 9.0\nAlpha CL CD CDp CM Top_Xtr Bot_Xtr\n0 .2 .01 .005 -.05 .8 .9\n'
    parsed = parse_polar_text(text)
    validate_polar_conditions(parsed.metadata)

@pytest.mark.parametrize('metadata', [{}, {'reynolds_number': 1000000.0, 'mach': 0.0, 'ncrit': None}, {'reynolds_number': 100000.0, 'mach': 0.0, 'ncrit': 9.0}, {'reynolds_number': 1000000.0, 'mach': 0.0, 'ncrit': 0.0}])
def test_conditions_fail_closed(metadata):
    with pytest.raises(ValueError):
        validate_polar_conditions(metadata)

def test_html_cannot_masquerade_as_a_polar():
    with pytest.raises(ValueError):
        parse_polar_text('<html>Alpha CL CD</html>')

def test_every_frozen_artifact_must_be_verified(tmp_path):
    model = tmp_path / 'm'
    scaler = tmp_path / 's'
    model.write_text('fixture')
    scaler.write_text('fixture2')
    artifacts = {'model': model, 'scaler': scaler}
    manifest = tmp_path / 'f.json'
    freeze_artifacts(manifest, artifacts, {})
    assert_locked_test_access(manifest, 'locked_test', artifacts)
    with pytest.raises(ValueError):
        verify_frozen_artifacts(manifest, {'model': model})
    with pytest.raises(PermissionError):
        assert_locked_test_access(manifest, 'locked_test')
    scaler.write_text('altered')
    with pytest.raises(PermissionError):
        assert_locked_test_access(manifest, 'locked_test', artifacts)

def test_frozen_metadata_cannot_override_contract(tmp_path):
    p = tmp_path / 'm'
    p.write_text('fixture')
    with pytest.raises(ValueError):
        freeze_artifacts(tmp_path / 'f.json', {'m': p}, {'artifact_hashes': {}})

def test_completed_state_without_outputs_cannot_skip(tmp_path):
    s = RunState(tmp_path / 's.json')
    s.begin('x', 'x')
    s.finish('x', 'passed', {})
    assert not s.can_skip('x', {}, {})

@pytest.mark.parametrize('model_status', ['failed', 'held'])
def test_model_failure_or_hold_blocks_advanced_phase(monkeypatch, model_status):
    calls = []

    class FakeState:
        def invalidate(self, phase_id, reason):
            assert phase_id == '17-21'
            assert model_status in reason

    def fake_phase(*args, **kwargs):
        del kwargs
        phase_id = args[1]
        calls.append(phase_id)
        if phase_id == '05-08':
            return {'mapped_airfoils': 25}, 'passed'
        if phase_id == '09-16':
            return {}, model_status
        if phase_id == '17-21':
            pytest.fail('advanced phase must not start without a passed frozen model phase')
        if phase_id == '22':
            return {'decision': 'HOLD'}, 'held'
        return {}, 'passed'

    monkeypatch.setattr(pipeline, 'load_all_configs', lambda: {})
    monkeypatch.setattr(pipeline, 'RunState', lambda path: FakeState())
    monkeypatch.setattr(pipeline, '_phase', fake_phase)

    assert pipeline.run_viability(resume=True) == 0
    assert '17-21' not in calls

def test_advanced_phase_rejects_tampered_frozen_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / 'artifact.txt'
    artifact.write_text('original', encoding='utf-8')
    manifest = tmp_path / 'frozen.json'
    freeze_artifacts(manifest, {'artifact': artifact}, {'lineage_id': 'test'})
    artifact.write_text('tampered', encoding='utf-8')
    monkeypatch.setattr(pipeline, 'FROZEN_MANIFEST_PATH', manifest)

    with pytest.raises(ValueError, match='Frozen artifact hash mismatch'):
        pipeline._phase14_21({})

def test_uniform_airfoil_sampling_does_not_get_double_weighted():

    class Fake(torch.nn.Module):

        def forward(self, x):
            return {'cl': x[:, 0], 'log_cd': x[:, 0], 'cm': x[:, 0]}
    batch = {'features': torch.tensor([[1.0], [3.0]]), 'airfoil_weight': torch.tensor([1.0, 0.01])}
    for name in ('cl', 'log_cd', 'cm'):
        batch[name] = torch.zeros(2)
        batch['mask_' + name] = torch.ones(2, dtype=torch.bool)
    uniform, _ = _batch_loss(Fake(), batch, torch.device('cpu'), 'huber', uniform_airfoil_sampling=True)
    weighted, _ = _batch_loss(Fake(), batch, torch.device('cpu'), 'huber')
    assert uniform.item() == pytest.approx(1.5)
    assert weighted.item() < 0.6

@pytest.mark.parametrize('target', ['cl', 'cd', 'cm'])
def test_xfoil_numerical_target_presence_is_required(target):
    points = pd.DataFrame({'alpha_deg': [0.0, 1.0, 2.0], 'cl': [0.1, 0.2, 0.3], 'cd': [0.01, 0.01, 0.01], 'cm': [-0.1, -0.1, -0.1]})
    points.loc[1, target] = np.nan
    assert _segment_status(0, points, case())[0] == 'failed'

def test_xfoil_strict_complete_case_passes():
    points = pd.DataFrame({'alpha_deg': [0.0, 1.0, 2.0], 'cl': [0.1, 0.2, 0.3], 'cd': [0.01, 0.01, 0.01], 'cm': [-0.1, -0.1, -0.1]})
    assert _segment_status(0, points, case()) == ('ok', None)

def test_xfoil_alpha_only_is_not_complete_aerodynamic_evidence():
    assert _segment_status(0, pd.DataFrame({'alpha_deg': [0.0, 1.0, 2.0]}), case())[0] == 'failed'
