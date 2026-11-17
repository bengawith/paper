"""Regression tests for strict direct-design verification status and evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from robust_airfoil.validation.xfoil_runner import XFoilResult


def test_design_verification_rejects_partial_sweep_even_if_polar_contains_lift_targets(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib.util

    source = Path(__file__).resolve().parents[2] / "scripts/robust_v2/verify_designs_xfoil.py"
    spec = importlib.util.spec_from_file_location("verify_designs_xfoil", source)
    assert spec is not None and spec.loader is not None
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)

    points = pd.DataFrame(
        {
            "alpha_deg": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "cl": [0.2, 0.45, 0.7, 0.9, 1.1, 1.2],
            "cd": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015],
            "cm": [-0.05] * 6,
        }
    )
    monkeypatch.setattr(verify, "OUT", tmp_path)
    monkeypatch.setattr(
        verify, "validate_geometry", lambda upper, lower: SimpleNamespace(valid=True)
    )
    monkeypatch.setattr(verify, "fit_cst", lambda upper, lower: SimpleNamespace(max_error=0.0))
    monkeypatch.setattr(
        verify,
        "resample_surfaces_to_common_x",
        lambda upper, lower, x: (upper, lower),
    )
    monkeypatch.setattr(
        verify,
        "run_xfoil",
        lambda *args: XFoilResult(
            status="failed",
            return_code=0,
            timed_out=False,
            command_text="",
            stdout_path=tmp_path / "stdout.txt",
            stderr_path=tmp_path / "stderr.txt",
            polar_path=None,
            parsed_points=points,
            failure_reason="missing requested alpha",
        ),
    )
    upper = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]])
    lower = np.array([[0.0, 0.0], [0.5, -0.1], [1.0, 0.0]])
    opt = SimpleNamespace(
        service_targets=SimpleNamespace(
            cl_required=[0.4, 0.7, 1.0],
            weights=[0.25, 0.5, 0.25],
        )
    )
    unc = SimpleNamespace(maximum_cst_refit_error_fraction_chord=0.01)
    result = verify._run(np.zeros(12), upper, lower, "case", tmp_path / "xfoil.exe", opt, unc)
    assert result is None


def test_promoted_strict_design_manifest_has_complete_trace() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "results/robust_v2/submission_review_20261014/xfoil_strict"
    summary = directory / "design_verification_summary.json"
    if not summary.exists():
        return  # CI need not download large local XFOIL evidence
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["designs"]["within_trust_knee"]["shared_converged"] <= 16
    for design, entry in data["designs"].items():
        statuses = []
        for case in ["nominal", *[f"shared_{i:02d}" for i in range(16)]]:
            path = directory / design / case / "run_status.json"
            assert path.exists()
            status = json.loads(path.read_text(encoding="utf-8"))
            if case.startswith("shared"):
                acceptable = status["status"] == "ok"
                if acceptable:
                    from robust_airfoil.optimisation.objectives import weighted_required_lift_drag

                    points = pd.read_csv(directory / design / case / "combined_points.csv")
                    value = weighted_required_lift_drag(
                        points["alpha_deg"].to_numpy(float),
                        points["cl"].to_numpy(float),
                        points["cd"].to_numpy(float),
                        np.asarray([0.4, 0.7, 1.0]),
                        np.asarray([0.25, 0.5, 0.25]),
                    )
                    acceptable = bool(np.isfinite(value))
                statuses.append(acceptable)
            if status["status"] == "ok":
                assert status["coverage_fraction"] == 1.0
                assert status["span_complete"]
                assert status["clean_exit"]
        assert sum(statuses) == entry["shared_converged"]
