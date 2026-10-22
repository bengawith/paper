"""Independent numerical equivalence checks for production scenario acceleration."""
import numpy as np
import pytest

from robust_airfoil.config import UncertaintyConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.geometry import cst
from robust_airfoil.optimisation import candidate_uncertainty as candidates
from robust_airfoil.uncertainty.fields import smooth_normal_perturbation


def config():
    return UncertaintyConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/uncertainty.yaml'))


def parameters():
    return np.array([-.13,-.13,-.1,-.08,-.05,.15,.17,.16,.1,.07,0.,.002])


def test_batched_constrained_fits_match_independent_scalar_least_squares():
    implementation = getattr(cst,'BatchCSTFitter',None)
    assert implementation is not None, 'The tested batch fitter is not implemented'
    x = (1-np.cos(np.linspace(0,np.pi,201)))/2
    upper,lower = cst.reconstruct_cst(parameters(),x)
    thickness = .1*np.sqrt(x)*(1-x)**1.25
    uppers = np.stack([upper,thickness,upper+.0003*np.sin(np.pi*x)])
    lowers = np.stack([lower,-thickness,lower+.0002*np.sin(np.pi*x)])
    batch = implementation(x).fit_many(uppers,lowers)
    for i in range(3):
        expected = cst.fit_cst(np.c_[x,uppers[i]],np.c_[x,lowers[i]])
        np.testing.assert_allclose(batch.parameters[i],expected.parameters,atol=1e-10,rtol=1e-10)
        assert batch.max_error[i] == pytest.approx(expected.max_error,abs=1e-12)
        assert batch.rms_error[i] == pytest.approx(expected.rms_error,abs=1e-12)


@pytest.mark.parametrize('rho',[-.6,0.,.7])
def test_prepared_displacements_are_each_candidates_actual_surface_normals(rho):
    implementation = getattr(candidates,'PreparedCandidateScenarios',None)
    assert implementation is not None, 'Prepared candidate-specific scenarios are not implemented'
    cfg = config().model_copy(update={'upper_lower_correlation':rho})
    draws = np.random.default_rng(880).normal(size=(16,16))
    prepared = implementation(draws,cfg)
    candidate = parameters()
    x = prepared.x
    uy,ly = cst.reconstruct_cst(candidate,x)
    upper,lower = np.c_[x,uy],np.c_[x,ly]
    actual_upper,actual_lower = prepared.displaced_surfaces(candidate)
    for i,row in enumerate(draws):
        coupled = rho*row[:8]+np.sqrt(1-rho*rho)*row[8:]
        expected = smooth_normal_perturbation(upper,lower,row[:8],coupled,.01,cfg.correlation_length_chord,cfg.leading_edge_taper,cfg.trailing_edge_zero_displacement)
        np.testing.assert_allclose(actual_upper[i],expected.upper,atol=2e-15,rtol=2e-14)
        np.testing.assert_allclose(actual_lower[i],expected.lower,atol=2e-15,rtol=2e-14)


def test_prepared_refits_match_slow_real_coordinate_path_not_reference_deltas():
    implementation = getattr(candidates,'PreparedCandidateScenarios',None)
    assert implementation is not None, 'Prepared candidate-specific scenarios are not implemented'
    cfg = config()
    draws = np.random.default_rng(881).normal(size=(32,16))
    prepared = implementation(draws,cfg,dense_verification_points=201)
    for variation in [0.,.01]:
        candidate = parameters().copy()
        candidate[7] += variation
        expected = candidates.candidate_surface_scenarios(candidate,draws,cfg)
        actual = prepared.evaluate(candidate)
        np.testing.assert_array_equal(actual.invalid,expected.invalid)
        np.testing.assert_allclose(actual.parameters,expected.parameters,atol=1e-9,rtol=1e-9)


def test_prepared_failure_placeholder_is_flagged_and_never_relabelled_valid():
    implementation = getattr(candidates,'PreparedCandidateScenarios',None)
    assert implementation is not None, 'Prepared candidate-specific scenarios are not implemented'
    cfg = config()
    invalid = parameters()
    invalid[:5],invalid[5:10] = .3,-.3
    result = implementation(np.zeros((4,16)),cfg).evaluate(invalid)
    assert result.invalid.all()
    assert all(result.invalid_reasons)
    np.testing.assert_array_equal(result.parameters,np.tile(invalid,(4,1)))
