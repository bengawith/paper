"""Equivalent documented coordinate orderings must encode the same airfoil."""
import numpy as np
import pytest

from robust_airfoil.geometry.normalise import normalise_geometry


def _surfaces():
    x = np.array([0, .01, .025, .05, .1, .2, .4, .6, .8, .95, 1.0])
    camber = .08 * np.sin(np.pi*x)
    thickness = .06 * np.sqrt(x) * (1-x)
    return np.column_stack([x,camber+thickness]), np.column_stack([x,camber-thickness])


def test_lednicer_and_selig_orders_produce_identical_normalised_geometry():
    upper, lower = _surfaces()
    lednicer = np.vstack([upper,lower])
    selig = np.vstack([upper[::-1],lower[1:]])
    actual = normalise_geometry(lednicer)
    expected = normalise_geometry(selig)
    np.testing.assert_allclose(actual.coordinates,expected.coordinates,atol=1e-12,rtol=0)
    assert actual.chord == pytest.approx(expected.chord)


def test_surface_membership_is_not_decided_by_camber_height():
    upper,lower = _surfaces()
    # Both surfaces have positive y over much of the chord. A median-y split
    # therefore changes the geometry, rather than merely identifying a surface.
    actual = normalise_geometry(np.vstack([upper,lower]),cosine_points=31)
    assert actual.upper[0,1] == pytest.approx(0.0,abs=1e-12)
    assert actual.lower[0,1] == pytest.approx(0.0,abs=1e-12)
    assert actual.upper[-1,1] == pytest.approx(0.0,abs=1e-12)
    assert actual.lower[-1,1] == pytest.approx(0.0,abs=1e-12)


def test_unordered_geometry_is_rejected_not_median_reconstructed():
    upper,lower = _surfaces()
    points = np.vstack([upper,lower])
    points[[3,17]] = points[[17,3]]
    with pytest.raises(ValueError,match='ordering'):
        normalise_geometry(points)
