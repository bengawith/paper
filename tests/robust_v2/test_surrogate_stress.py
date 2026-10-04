import numpy as np
import pandas as pd

from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.surrogate_stress import paired_disagreement, sobol_design_pool


def test_paired_disagreement_matches_only_converged_angles():
    alpha = np.array([0.0, 2.0, 4.0])
    cl_ml = np.array([0.30, 0.50, 0.70])
    cd_ml = np.array([0.0100, 0.0110, 0.0120])
    xfoil = pd.DataFrame(
        {
            "alpha_deg": [0.0, 2.0],  # 4 deg did not converge
            "cl": [0.32, 0.55],
            "cd": [0.0090, 0.0100],
        }
    )
    result = paired_disagreement(alpha, cl_ml, cd_ml, xfoil)
    assert result["matched_points"] == 2
    assert result["requested_points"] == 3
    assert np.isclose(result["max_absolute_cl_gap"], 0.05)
    assert np.isclose(result["mean_absolute_cl_gap"], (0.02 + 0.05) / 2)
    # relative cd gaps: |0.010-0.009|/0.009, |0.011-0.010|/0.010
    assert np.isclose(result["max_relative_cd_gap"], max(0.001 / 0.009, 0.001 / 0.010))


def test_paired_disagreement_ignores_nonpositive_drag_rows():
    alpha = np.array([0.0, 2.0])
    cl_ml = np.array([0.30, 0.50])
    cd_ml = np.array([0.0100, 0.0110])
    xfoil = pd.DataFrame({"alpha_deg": [0.0, 2.0], "cl": [0.32, 0.55], "cd": [0.0, 0.0100]})
    result = paired_disagreement(alpha, cl_ml, cd_ml, xfoil)
    assert result["matched_points"] == 1
    assert result["matched_alpha_deg"] == [2.0]


def test_paired_disagreement_handles_empty_polar():
    alpha = np.array([0.0, 2.0])
    result = paired_disagreement(alpha, np.array([0.3, 0.5]), np.array([0.01, 0.011]), pd.DataFrame())
    assert result["matched_points"] == 0
    assert result["mean_absolute_cl_gap"] is None


def test_sobol_design_pool_respects_bounds_and_shape():
    bounds = DesignBounds(lower=np.array([-1.0, 0.0]), upper=np.array([1.0, 2.0]))
    pool = sobol_design_pool(bounds, 16, seed=7)
    assert pool.shape == (16, 2)
    assert np.all(pool >= bounds.lower - 1e-9)
    assert np.all(pool <= bounds.upper + 1e-9)
