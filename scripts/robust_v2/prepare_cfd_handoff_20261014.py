"""Make a compact CFD case handoff from independently rerun strict XFOIL files.

All selected perturbations must have clean-exit and complete requested alpha
coverage for both reference and candidate. No CFD results are simulated.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.optimisation.objectives import weighted_required_lift_drag

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/robust_v2/submission_review_20261014/xfoil_strict"
OUT = ROOT / "reports/robust_v2/submission_review_20261014/CFD_HANDOFF"
INPUTS = OUT / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)
SUMMARY = json.loads((SRC / "design_verification_summary.json").read_text(encoding="utf-8"))
TARGETS = (0.4, 0.7, 1.0)
WEIGHTS = (0.25, 0.50, 0.25)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(design: str, scenario: str) -> bool:
    p = SRC / design / scenario / "run_status.json"
    if not p.exists():
        return False
    v = json.loads(p.read_text(encoding="utf-8"))
    return (
        v["status"] == "ok"
        and v["return_code"] == 0
        and v["coverage_fraction"] == 1.0
        and v["span_complete"]
    )


def drag(design: str, scenario: str) -> float:
    df = pd.read_csv(SRC / design / scenario / "combined_points.csv")
    return float(
        weighted_required_lift_drag(
            df["alpha_deg"].to_numpy(float),
            df["cl"].to_numpy(float),
            df["cd"].to_numpy(float),
            np.asarray(TARGETS),
            np.asarray(WEIGHTS),
        )
    )


def alpha_at_lift(df: pd.DataFrame, target: float) -> float | None:
    x = df.sort_values("alpha_deg").drop_duplicates("alpha_deg")
    a = x["alpha_deg"].to_numpy(float)
    cl = x["cl"].to_numpy(float)
    crossing = []
    for j in range(len(a) - 1):
        if cl[j + 1] <= cl[j]:
            # Do not extrapolate or use re-rising post-stall branches.
            continue
        if cl[j] <= target <= cl[j + 1]:
            crossing.append(
                float(a[j] + (a[j + 1] - a[j]) * (target - cl[j]) / (cl[j + 1] - cl[j]))
            )
    return crossing[0] if len(crossing) == 1 else None


ref = "reference_naca2412"
good = "within_trust_knee"
bad = "outside_trust_deterministic"
shared = [f"shared_{i:02d}" for i in range(16)]
common = [
    k
    for k in shared
    if complete(ref, k)
    and complete(good, k)
    and np.isfinite(drag(ref, k))
    and np.isfinite(drag(good, k))
]
if len(common) < 3:
    raise RuntimeError(f"Insufficient strict paired scenarios: {len(common)}")
ordered = sorted(common, key=lambda k: drag(good, k))
median_case = ordered[len(ordered) // 2]
adverse_case = ordered[-1]
cases = [(ref, "nominal"), (good, "nominal"), (bad, "nominal")]
for scenario in (median_case, adverse_case):
    cases.extend([(ref, scenario), (good, scenario)])
# Rebuild only this program-generated CFD input directory; never touch original XFOIL evidence.
if INPUTS.exists():
    shutil.rmtree(INPUTS)
INPUTS.mkdir(parents=True, exist_ok=True)
manifest = {
    "schema": "robust-v2-cfd-handoff-v1",
    "source_git_commit": "8b14165ce74396e0c504b1f165dd6eb248d51cae",
    "source_xfoil_dir": SRC.relative_to(ROOT).as_posix(),
    "source_summary_sha256": sha(SRC / "design_verification_summary.json"),
    "reynolds_number": 1_000_000,
    "mach": 0.0,
    "source_ncrit": 9.0,
    "requested_lift": list(TARGETS),
    "lift_weights": list(WEIGHTS),
    "selection": {
        "common_strict_successful_scenarios": len(common),
        "median_sample": median_case,
        "largest_sampled_candidate_drag": adverse_case,
        "note": "Largest in the available 10 strictly paired cases, not a population 95%-tail quantile.",
    },
    "cases": [],
}
for design, scenario in cases:
    src = SRC / design / scenario
    if not complete(design, scenario):
        raise AssertionError((design, scenario))
    dest = INPUTS / design / scenario
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("geometry.dat", "combined_points.csv", "run_status.json", "command.txt"):
        shutil.copy2(src / name, dest / name)
    df = pd.read_csv(dest / "combined_points.csv")
    row = {
        "design": design,
        "scenario": scenario,
        "source_path": src.relative_to(ROOT).as_posix(),
        "xfoil_weighted_cd": drag(design, scenario),
        "files": {
            name: sha(dest / name)
            for name in ("geometry.dat", "combined_points.csv", "run_status.json", "command.txt")
        },
        "xfoil_initial_alpha_deg_for_target_cl": {str(t): alpha_at_lift(df, t) for t in TARGETS},
        "target_alpha_is_a_starting_guess_not_cfd_output": True,
    }
    manifest["cases"].append(row)
(OUT / "CFD_CASE_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("CFD handoff prepared:", len(cases), "cases; common scenarios", len(common))
print("Scenarios:", median_case, adverse_case)
