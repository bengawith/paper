import numpy as np

from robust_airfoil.geometry.validity import validate_geometry


def test_crossing_surface_is_invalid():
    x = np.linspace(0, 1, 10)
    result = validate_geometry(np.column_stack([x, np.zeros(10)]), np.column_stack([x, np.full(10, 0.1)]))
    assert not result.valid
    assert "surface_crossing" in result.reasons


def test_positive_thickness_is_valid():
    x = np.linspace(0, 1, 10)
    thickness = 0.1 * np.sin(np.pi * x)
    assert validate_geometry(np.column_stack([x, thickness]), np.column_stack([x, -thickness])).valid
