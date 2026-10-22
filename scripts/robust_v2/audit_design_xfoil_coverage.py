"""Audit strict finite-polar alpha coverage in previously archived XFOIL design runs.

This does NOT infer missing process exit codes: source run_xfoil result status
was not persisted alongside every case, so coverage alone cannot certify complete.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
base = root / "results/robust_v2/lineages/robust-v2-current-20260908/design_verification"
dest = root / "reports/robust_v2/submission_review_20261014"
dest.mkdir(parents=True, exist_ok=True)
rows = []
for design in ("reference_naca2412", "within_trust_knee", "outside_trust_deterministic"):
    for label in ["nominal"] + [f"shared_{i:02d}" for i in range(16)]:
        case = base / design / label
        pol = case / "combined_points.csv"
        flags = []
        valid_alphas = []
        if pol.exists():
            with pol.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        a = float(row["alpha_deg"])
                        cl = float(row["cl"])
                        cd = float(row["cd"])
                        cm = float(row["cm"])
                        if all(map(math.isfinite, (a, cl, cd, cm))) and cd > 0:
                            valid_alphas.append(a)
                    except (ValueError, KeyError, TypeError):
                        pass
        else:
            flags.append("no_combined_polar")
        missing = [a for a in range(12) if not any(abs(x - a) < 0.00051 for x in valid_alphas)]
        if missing:
            flags.append("incomplete_requested_alpha")
        if not (case / "command.txt").exists():
            flags.append("missing_command")
        # Completion status cannot be guaranteed without process return-code proof.
        rows.append(
            {
                "design": design,
                "case": label,
                "valid_points": len(valid_alphas),
                "missing_alphas": missing,
                "requested_grid_covered": len(missing) == 0,
                "stored_return_code_known": False,
                "flags": flags,
                "path": case.relative_to(root).as_posix(),
            }
        )
counts = {
    d: dict(
        Counter(
            "covered" if r["requested_grid_covered"] else "incomplete"
            for r in rows
            if r["design"] == d
        )
    )
    for d in ("reference_naca2412", "within_trust_knee", "outside_trust_deterministic")
}
result = {
    "schema": "strict-archived-xfoil-coverage-audit-v1",
    "alpha_grid": [0, 11, 1],
    "counts": counts,
    "cases": rows,
    "warning": "Full status cannot be reconstructed without persisted subprocess exit-code/timeout metadata. Requested-angle coverage is necessary but not sufficient.",
}
(dest / "XFOIL_ARCHIVE_COVERAGE_AUDIT.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(counts, indent=2))
print("Nominal", [(r["design"], r["missing_alphas"]) for r in rows if r["case"] == "nominal"])
