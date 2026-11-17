import numpy as np

from robust_airfoil.uncertainty.fields import smooth_normal_perturbation


def test_smooth_field_is_correlated_and_zero_at_trailing_edge():
    x = (1 - np.cos(np.linspace(0, np.pi, 101))) / 2
    upper = np.column_stack([x, 0.08 * np.sin(np.pi * x)])
    lower = np.column_stack([x, -0.04 * np.sin(np.pi * x)])
    result = smooth_normal_perturbation(upper, lower, np.ones(8), -np.ones(8), 0.001)
    assert result.valid
    assert result.upper_displacement[-1] == 0
    assert result.lower_displacement[-1] == 0
    assert np.max(np.abs(np.diff(result.upper_displacement))) < 0.001
