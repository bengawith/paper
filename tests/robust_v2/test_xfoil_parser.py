from pathlib import Path

import pandas as pd
import pytest

from robust_airfoil.validation.xfoil_runner import (
    XFoilCase,
    _comparison,
    _segment_status,
    _sweep_completion,
    build_xfoil_commands,
)


def test_xfoil_command_contains_reproducible_settings():
    case = XFoilCase(Path("shape.dat"), 1e6, 0.0, 9.0, 0.0, 10.0, 1.0, 70, "positive")
    command = build_xfoil_commands(case, "geometry.dat", "polar.txt")
    assert "VISC 1000000" in command
    assert "N 9" in command
    assert "ITER 70" in command
    assert "ASEQ 0 10 1" in command
    assert "PACC\npolar.txt\n\nASEQ" in command
    assert "PLOP" not in command
    assert "ALFA 0" not in command


def test_xfoil_rejects_wrong_sweep_direction():
    case = XFoilCase(Path("shape.dat"), 1e6, 0.0, 9.0, 0.0, 10.0, -1.0, 70, "positive")
    with pytest.raises(ValueError):
        build_xfoil_commands(case, "geometry.dat", "polar.txt")


def test_xfoil_comparison_handles_missing_polar_schema():
    comparison = _comparison(pd.DataFrame(), None)
    assert comparison["local_converged_points"] == 0
    assert comparison["archive_comparison"] == "unavailable"


def test_xfoil_sweep_requires_coverage_and_span():
    case = XFoilCase(Path("shape.dat"), 1e6, 0.0, 9.0, 0.0, 10.0, 1.0, 70, "positive")
    expected, coverage, complete = _sweep_completion(
        pd.DataFrame({"alpha_deg": range(9)}), case
    )
    assert expected == 11
    assert coverage > 0.8
    assert not complete


def test_xfoil_segment_rejects_nonzero_exit_with_complete_polar():
    case = XFoilCase(Path("shape.dat"), 1e6, 0.0, 9.0, 0.0, 2.0, 1.0, 70, "positive")
    points = pd.DataFrame({"alpha_deg": [0.0, 1.0, 2.0]})
    status, reason = _segment_status(1, points, case)
    assert status == "failed"
    assert reason is not None and "return_code=1" in reason
