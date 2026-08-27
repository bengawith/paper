import numpy as np

from robust_airfoil.geometry.cst import fit_cst
from robust_airfoil.geometry.normalise import normalise_geometry, resample_surfaces_to_common_x
from robust_airfoil.geometry.parser import parse_coordinate_text


def test_normalise_produces_shared_cosine_grid(coordinate_text):
    result = normalise_geometry(parse_coordinate_text(coordinate_text), cosine_points=41)
    assert result.upper.shape == result.lower.shape == (41, 2)
    assert np.allclose(result.upper[:, 0], result.lower[:, 0])
    assert np.all(result.upper[:, 1] >= result.lower[:, 1] - 1e-10)
    assert np.isclose(result.upper[0, 0], 0) and np.isclose(result.upper[-1, 0], 1)


def test_displaced_surfaces_can_be_refit_on_shared_grid():
    x = (1 - np.cos(np.linspace(0, np.pi, 101))) / 2
    upper = np.column_stack([x, 0.08 * np.sin(np.pi * x)])
    lower = np.column_stack([x, -0.04 * np.sin(np.pi * x)])
    upper[1:-1, 0] += 0.0002 * np.sin(np.pi * x[1:-1])
    lower[1:-1, 0] -= 0.0002 * np.sin(np.pi * x[1:-1])
    refit_upper, refit_lower = resample_surfaces_to_common_x(upper, lower, x)
    assert np.array_equal(refit_upper[:, 0], refit_lower[:, 0])
    assert np.array_equal(refit_upper[:, 0], x)
    assert np.isfinite(fit_cst(refit_upper, refit_lower).parameters).all()
