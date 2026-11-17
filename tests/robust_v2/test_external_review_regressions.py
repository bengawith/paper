"""Software-only fixtures for defects found in the independent academic review.

These synthetic arrays are regression fixtures, never aerodynamic evidence.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from robust_airfoil.config import OptimisationConfig, UncertaintyConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS, TARGET_COLUMNS
from robust_airfoil.optimisation import study
from robust_airfoil.optimisation.problems import DesignBounds


def test_optimiser_rejects_angles_outside_calibrated_coverage():
    evaluator = study.RobustCandidateEvaluator.__new__(study.RobustCandidateEvaluator)
    frame = pd.DataFrame(np.zeros((3, len(FEATURE_COLUMNS))), columns=pd.Index(FEATURE_COLUMNS))
    frame['alpha_deg'] = [-2.0, 0.0, 2.0]
    for target in TARGET_COLUMNS:
        frame['ensemble_std_' + target] = 0.0
    evaluator.feature_center = np.zeros(len(FEATURE_COLUMNS))
    evaluator.feature_scale = np.ones(len(FEATURE_COLUMNS))
    evaluator.support_tree = cKDTree(np.repeat(frame[FEATURE_COLUMNS].to_numpy(float), 5, axis=0))
    evaluator.support_threshold = 0.1
    evaluator.disagreement_thresholds = dict.fromkeys(TARGET_COLUMNS, 0.1)
    evaluator.alpha_coverage = (-1.0, 1.0)
    np.testing.assert_array_equal(evaluator._support_and_disagreement(frame, (3,)), [False, True, False])


def test_displaced_surfaces_are_compared_at_common_physical_x():
    upper = np.array([[0, 0], [.2, .11], [.8, .04], [1, 0]], float)
    lower = np.array([[0, 0], [.7, .10], [.9, .03], [1, 0]], float)
    # Index-wise y differences are nonnegative, but the polylines intersect.
    assert np.all(upper[:, 1] >= lower[:, 1])
    result = validate_geometry(upper, lower)
    assert not result.valid
    assert 'surface_crossing' in result.reasons


@pytest.mark.parametrize('bad', [np.array([]), np.zeros((0, 2)), np.array([0.0, 1.0])])
def test_malformed_geometry_fails_closed_without_index_error(bad):
    assert not validate_geometry(bad, bad).valid


class _FixtureEnsemble:
    first_scaling = SimpleNamespace(feature_center=np.zeros(13), feature_scale=np.ones(13))
    artifact_hashes = {'fixture': 'unmodified'}
    device = 'cpu'

    def predict(self, frame):
        result = frame.copy()
        result['prediction_cl'] = 0.1 + 0.01 * result.alpha_deg
        result['prediction_log_cd'] = np.log(.01)
        result['prediction_cm'] = -.03
        for target in TARGET_COLUMNS:
            result['ensemble_std_' + target] = 0.0
        return result


def _configuration():
    opt = OptimisationConfig.model_validate(load_yaml(ROOT / 'configs/robust_v2/optimisation.yaml'))
    unc = UncertaintyConfig.model_validate(load_yaml(ROOT / 'configs/robust_v2/uncertainty.yaml'))
    return opt, unc


def test_nominal_required_lift_failure_is_not_a_feasible_design():
    opt, _ = _configuration()
    parameters = np.array([-.13, -.13, -.1, -.08, -.05, .15, .17, .16, .1, .07, 0, 0])
    development = study._frame(np.tile(parameters, (5, 1)), np.asarray(opt.ava_baseline.alpha_deg))
    calibration = {
        'support_distance_quantiles': {'trusted_threshold_from_residual_tolerance': 1e6},
        'ensemble_disagreement_quantiles': {t: {'trusted_threshold_from_residual_tolerance': 1.0} for t in TARGET_COLUMNS},
        'alpha_coverage_deg': [-20, 20],
    }
    evaluator = study.RobustCandidateEvaluator(parameters, np.zeros((2, 12)), np.zeros(2, bool), development, _FixtureEnsemble(), calibration, opt)
    constraints, diagnostics = evaluator._nominal_constraints(parameters[None, :])
    assert not np.all(constraints[0] <= 0)
    assert diagnostics[0]['nominal_required_lift_feasible'] is False


def _profile_arguments(tmp_path):
    opt, unc = _configuration()
    profile = opt.debug_nsga2.model_copy(update={'population': 4, 'generations': 1, 'uncertainty_samples': 2, 'independent_seeds': 1})
    development = pd.DataFrame(np.zeros((5, len(FEATURE_COLUMNS))), columns=pd.Index(FEATURE_COLUMNS))
    return dict(name='external-regression', profile=profile, optimisation=opt, reference_parameters=np.zeros(12), master_deltas=np.zeros((2, 16)), master_invalid=np.zeros(2, bool), development_points=development, ensemble=_FixtureEnsemble(), calibration={}, bounds=DesignBounds(-np.ones(12), np.ones(12)), output_root=tmp_path, profile_index=0, ensemble_manifest_hash='fixture-manifest', uncertainty=unc)


def test_objective_source_changes_invalidate_optimisation_cache(monkeypatch, tmp_path):
    class StopBeforeEvaluation(Exception):
        pass

    captured = []
    changed = [False]
    original_hash = study.hash_object

    def capture(value):
        result = original_hash(value)
        captured.append(result)
        return result

    def stop(*args, **kwargs):
        raise StopBeforeEvaluation

    monkeypatch.setattr(study, 'hash_object', capture)
    monkeypatch.setattr(study, 'sha256_file', lambda path: 'changed-objectives' if Path(path).name == 'objectives.py' and changed[0] else 'same-fixture-hash')
    monkeypatch.setattr(study, 'RobustCandidateEvaluator', stop)
    for value in (False, True):
        changed[0] = value
        with pytest.raises(StopBeforeEvaluation):
            study._run_nsga_profile(**_profile_arguments(tmp_path))
    assert captured[0] != captured[1]


def test_stale_seed_evidence_is_not_overwritten(monkeypatch, tmp_path):
    arguments = _profile_arguments(tmp_path)
    seed = arguments['optimisation'].seed + 1000
    path = tmp_path / f'nsga2_external-regression_seed_{seed}.json'
    old = json.dumps({'schema_version': 3, 'sample_count': 2, 'history': [{}], 'cache_context_hash': 'old-context'})
    path.write_text(old)

    class NoEvaluation:
        def evaluate(self, *args):
            pytest.fail('A stale evidence file must be refused before new evaluations')

    monkeypatch.setattr(study, 'RobustCandidateEvaluator', lambda *args, **kwargs: NoEvaluation())
    with pytest.raises(RuntimeError, match='fresh output revision'):
        study._run_nsga_profile(**arguments)
    assert path.read_text() == old


def test_ensemble_rejects_changed_checkpoint_before_deserialisation(monkeypatch, tmp_path):
    from robust_airfoil.modelling import ensemble as module

    member = tmp_path / 'member_000'
    member.mkdir()
    checkpoint = member / 'checkpoint.pt'
    scaler = member / 'scaling.joblib'
    checkpoint.write_bytes(b'original fixture checkpoint')
    scaler.write_bytes(b'fixture scaler')
    hashes = {'member_000_checkpoint': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'member_000_scaling': hashlib.sha256(scaler.read_bytes()).hexdigest()}
    (tmp_path / 'ensemble_manifest.json').write_text(json.dumps({'member_count': 1, 'artifact_hashes': hashes}))
    checkpoint.write_bytes(b'changed fixture checkpoint')
    monkeypatch.setattr(module, 'load_member', lambda *args: (None, None))
    with pytest.raises(ValueError, match='hash mismatch'):
        module.LoadedEnsemble(tmp_path, members=1)
