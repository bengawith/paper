import numpy as np
import pandas as pd
import pytest

from robust_airfoil.predict import alpha_grid, assemble_polar, parse_cst


def test_parse_cst_accepts_twelve_parameters():
    vec = parse_cst("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.01,0.002")
    assert vec.shape == (12,)
    assert np.isclose(vec[10], 0.01)


def test_parse_cst_rejects_wrong_count():
    with pytest.raises(ValueError):
        parse_cst("0,0.1,0.2")


def test_parse_cst_rejects_nonnumeric():
    with pytest.raises(ValueError):
        parse_cst("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.01,abc")


def test_alpha_grid_inclusive_and_ordered():
    grid = alpha_grid(0.0, 10.0, 2.0)
    assert grid.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_alpha_grid_rejects_bad_step_and_direction():
    with pytest.raises(ValueError):
        alpha_grid(0.0, 10.0, 0.0)
    with pytest.raises(ValueError):
        alpha_grid(10.0, 0.0, 1.0)


def test_assemble_polar_schema_and_trust_flag():
    frame = pd.DataFrame(
        {
            "alpha_deg": [2.0, 0.0],
            "prediction_cl": [0.5, 0.3],
            "prediction_log_cd": [np.log(0.011), np.log(0.010)],
            "prediction_cm": [-0.05, -0.05],
            "ensemble_std_cl": [0.01, 0.02],
            "ensemble_std_log_cd": [0.03, 0.04],
            "ensemble_std_cm": [0.001, 0.002],
            "support_distance": [0.5, 0.6],
            "trusted_domain": [True, False],
        }
    )
    polar = assemble_polar(frame)
    assert [p["alpha_deg"] for p in polar] == [0.0, 2.0]  # sorted
    assert np.isclose(polar[1]["cd"], 0.011)
    assert polar[0]["within_trust_domain"] is False
    assert polar[1]["within_trust_domain"] is True
