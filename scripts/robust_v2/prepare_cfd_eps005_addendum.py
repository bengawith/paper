"""Add four OPTIONAL 0.5%-chord CFD inputs from strictly matched XFOIL scenarios."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.optimisation.objectives import weighted_required_lift_drag

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/robust_v2/submission_review_20261014/xfoil_eps005_n128"
DEST = ROOT / "reports/robust_v2/submission_review_20261014/CFD_HANDOFF"
OUT = DEST / "OPTIONAL_0P5_PERCENT"
ref = "reference_naca2412"
design = "within_trust_knee"
req = np.asarray([0.4, 0.7, 1.0], float)
weights = np.asarray([0.25, 0.5, 0.25], float)


def weighted(design_id: str, sample: str) -> float | None:
    folder = SRC / design_id / sample
    try:
        status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
        if not (
            status["status"] == "ok"
            and status["return_code"] == 0
            and status["coverage_fraction"] == 1.0
            and status["span_complete"]
        ):
            return None
        p = pd.read_csv(folder / "combined_points.csv")
        d = weighted_required_lift_drag(
            p.alpha_deg.to_numpy(float), p.cl.to_numpy(float), p.cd.to_numpy(float), req, weights
        )
        return float(d) if np.isfinite(d) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


pairs = []
for index in range(128):
    s = f"shared_{index:02d}"
    a, b = weighted(ref, s), weighted(design, s)
    if a is not None and b is not None:
        pairs.append((s, a, b))
if len(pairs) < 20:
    raise RuntimeError(f"Not enough complete paired scenarios: {len(pairs)}")
ordered = sorted(pairs, key=lambda p: p[2])
chosen = [ordered[len(ordered) // 2], ordered[-1]]
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
records = []
for label, (sample, ref_cd, design_cd) in zip(
    ("median_completed", "largest_completed_drag"), chosen, strict=True
):
    for name, cd in ((ref, ref_cd), (design, design_cd)):
        dest = OUT / label / name
        dest.mkdir(parents=True, exist_ok=True)
        source = SRC / name / sample
        files = {}
        for file_name in ("geometry.dat", "combined_points.csv", "run_status.json", "command.txt"):
            a = source / file_name
            b = dest / file_name
            shutil.copy2(a, b)
            files[file_name] = hashlib.sha256(b.read_bytes()).hexdigest()
        records.append(
            {
                "label": label,
                "design": name,
                "source_scenario": sample,
                "source": source.relative_to(ROOT).as_posix(),
                "xfoil_weighted_cd": cd,
                "files": files,
            }
        )
manifest = {
    "schema": "cfd-optional-0p5pct-v1",
    "source_scramble": "first 128-case scrambled Sobol at 0.5% chord",
    "complete_paired_pool_count": len(pairs),
    "selection_method": "median and max candidate drag among strictly complete target-lift-feasible paired cases; not physical population quantile",
    "re": 1000000,
    "mach": 0,
    "ncrit": 9,
    "epsilon_fraction_chord": 0.005,
    "cases": records,
}
(OUT / "OPTIONAL_CFD_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print("Wrote optional 0.5% CFD cases", len(records), "paired pool", len(pairs))
