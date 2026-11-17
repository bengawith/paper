import numpy as np
import pytest

from robust_airfoil.optimisation.objectives import (
    required_lift_drag,
    robust_drag_objectives,
    weighted_required_lift_drag,
)


def test_required_lift_and_tail_risk_metrics():
    alpha = np.array([0.0, 2.0, 4.0, 6.0])
    cl = np.array([0.0, 0.4, 0.8, 1.2])
    cd = np.array([0.01, 0.011, 0.014, 0.020])
    assert required_lift_drag(alpha, cl, cd, 0.6) == pytest.approx(0.0125)
    weighted = weighted_required_lift_drag(alpha, cl, cd, np.array([0.4, 0.8]), np.array([0.25, 0.75]))
    assert weighted == pytest.approx(0.01325)


def test_required_lift_uses_unique_rising_alpha_branch():
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = np.array([0.0, 0.5, 1.0, 0.8, 0.4])
    cd = np.array([0.01, 0.012, 0.02, 0.03, 0.04])
    assert required_lift_drag(alpha, cl, cd, 0.6) == pytest.approx(0.0136)


def test_required_lift_rejects_multiple_rising_crossings():
    alpha = np.array([0.0, 2.0, 4.0, 6.0])
    cl = np.array([0.0, 0.8, 0.2, 0.9])
    cd = np.array([0.01, 0.02, 0.03, 0.04])
    assert np.isinf(required_lift_drag(alpha, cl, cd, 0.5))
    risk = robust_drag_objectives(np.array([0.01, 0.011, 0.012, 0.1]))
    assert risk["cvar_95_weighted_cd"] >= risk["expected_weighted_cd"]
