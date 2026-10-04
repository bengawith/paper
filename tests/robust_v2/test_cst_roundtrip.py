import numpy as np

from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst


def test_exact_cst_roundtrip():
    x = (1 - np.cos(np.linspace(0, np.pi, 101))) / 2
    parameters = np.array([-0.08, -0.06, -0.04, -0.02, -0.01, 0.12, 0.16, 0.13, 0.08, 0.03, 0.005, 0.002])
    upper_y, lower_y = reconstruct_cst(parameters, x)
    fit = fit_cst(np.column_stack([x, upper_y]), np.column_stack([x, lower_y]))
    assert fit.rms_error < 1e-10
    assert np.allclose(fit.parameters, parameters, atol=1e-8)
