"""Aggregate two independent 0.5%-chord, 128-scenario, strict-XFOIL checks.

Preserves solver-failure accounting and pairs by stored *same latent scenario index*.
Descriptive robust tail numbers are conditional on convergence; no iid-CI claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.optimisation.objectives import weighted_required_lift_drag

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/robust_v2/submission_review_20261014"
OUT = ROOT / "reports/robust_v2/submission_review_20261014/ML_DIAGNOSTICS"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = [
    BASE / "xfoil_eps005_n128",
    BASE / "xfoil_eps005_n128_b",
]
T = np.asarray([0.4, 0.7, 1.0])
W = np.asarray([0.25, 0.5, 0.25])


def usable_value(path: Path) -> float | None:
    if not (path / "run_status.json").is_file():
        return None
    st = json.loads((path / "run_status.json").read_text(encoding="utf-8"))
    if not (
        st.get("status") == "ok"
        and st.get("return_code") == 0
        and st.get("span_complete")
        and st.get("coverage_fraction") == 1.0
    ):
        return None
    f = path / "combined_points.csv"
    if not f.is_file():
        return None
    df = pd.read_csv(f)
    if not set(("alpha_deg", "cl", "cd")).issubset(df.columns):
        return None
    result = float(
        weighted_required_lift_drag(
            df.alpha_deg.to_numpy(float),
            df.cl.to_numpy(float),
            df.cd.to_numpy(float),
            T,
            W,
        )
    )
    return result if np.isfinite(result) else None


def empirical_upper_cvar(v: np.ndarray, tail_fraction: float = 0.05) -> float:
    if len(v) == 0:
        return float("nan")
    order = np.sort(v)[::-1]
    mass = len(v) * tail_fraction
    full = int(np.floor(mass))
    fraction = mass - full
    weighted = float(np.sum(order[:full]))
    if full < len(v):
        weighted += fraction * float(order[full])
    return weighted / mass


def summarize(directory: Path) -> dict:
    if not directory.exists():
        return {"status": "missing", "directory": str(directory)}
    summary_path = directory / "design_verification_summary.json"
    if not summary_path.exists():
        return {"status": "running", "directory": str(directory)}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert int(summary["shared_perturbations"]) == 128
    assert abs(float(summary["manufacturing_amplitude_fraction_chord"]) - 0.005) < 1e-10
    ref = "reference_naca2412"
    candidate = "within_trust_knee"
    outside = "outside_trust_deterministic"
    by_design = {
        design: [usable_value(directory / design / f"shared_{i:02d}") for i in range(128)]
        for design in (ref, candidate, outside)
    }
    pair = [
        (a, b)
        for a, b in zip(by_design[ref], by_design[candidate], strict=True)
        if a is not None and b is not None
    ]
    if not pair:
        raise RuntimeError("No paired scenarios")
    rv = np.array([a for a, b in pair], float)
    cv = np.array([b for a, b in pair], float)
    diff = rv - cv
    records = {
        "status": "complete",
        "directory": directory.relative_to(ROOT).as_posix(),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "scramble_seed_offset": summary.get("sobol_scramble_seed_offset", 50000),
        "scramble_seed_note": (
            "Legacy source used fixed offset 50000"
            if "sobol_scramble_seed_offset" not in summary
            else "Recorded in output"
        ),
        "requested": 128,
        "reference_usable": sum(v is not None for v in by_design[ref]),
        "candidate_usable": sum(v is not None for v in by_design[candidate]),
        "outside_usable": sum(v is not None for v in by_design[outside]),
        "paired_usable": len(pair),
        "paired_reference_mean": float(rv.mean()),
        "paired_candidate_mean": float(cv.mean()),
        "paired_improvement_mean": float(diff.mean()),
        "paired_improvement_fraction": float(diff.mean() / rv.mean()),
        "paired_improvement_positive_count": int((diff > 0).sum()),
        "paired_reference_empirical_cvar95": empirical_upper_cvar(rv),
        "paired_candidate_empirical_cvar95": empirical_upper_cvar(cv),
        "paired_empirical_cvar95_candidate_change_fraction": (
            empirical_upper_cvar(cv) / empirical_upper_cvar(rv) - 1
        ),
        "max_observed_reference": float(rv.max()),
        "max_observed_candidate": float(cv.max()),
        "failure_selection_warning": "All values conditioned on complete sweeps and available lift roots. Solver failures are never imputed; tails are not population estimates.",
    }
    return records


def main() -> None:
    runs = [summarize(p) for p in SOURCES]
    complete = [r for r in runs if r["status"] == "complete"]
    result = {
        "schema": "xfoil-two-independent-scramble-sensitivity-0p5pct-v1",
        "condition": "Re=1e6, Mach=0, Ncrit=9, 0.5% chord prescribed correlated perturbation",
        "runs": runs,
        "independent_scrambles_finished": len(complete),
        "limitations": [
            "Complete and required-lift feasible cases only; missingness is nonrandom and may depend on geometry.",
            "Two independent QMC scrambles are too few for a formal uncertainty interval across designs.",
            "CVaR95 from the finite accepted subset is descriptive, not a manufacturing population tail guarantee.",
            "All comparisons use XFOIL-class numerical physics rather than independent RANS or experiment.",
        ],
    }
    if complete:
        result["across_scramble_paired_mean_improvements"] = [
            r["paired_improvement_mean"] for r in complete
        ]
        result["across_scramble_paired_fractional_improvements"] = [
            r["paired_improvement_fraction"] for r in complete
        ]
        result["all_completed_same_direction"] = all(
            r["paired_improvement_mean"] > 0 for r in complete
        )
    target = OUT / "XFOIL_EPS005_TWO_SCRAMBLE_AUDIT.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": [
                    {k: v for k, v in r.items() if k not in ("directory", "summary_sha256")}
                    for r in complete
                ],
                "n": len(complete),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
