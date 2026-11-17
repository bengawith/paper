"""Compress corrected ten-seed low-budget NSGA-II evidence without hiding failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROOT_OUT = ROOT / "reports/robust_v2/submission_review_20261014/CORRECTED_OPTIMISATION"
seeds_dir = ROOT_OUT / "smoke_seeds"
summary = json.loads((ROOT_OUT / "TEN_SEED_SMOKE_SUMMARY.json").read_text(encoding="utf-8"))
records = []
for file in sorted(seeds_dir.glob("nsga2_corrected-smoke_seed_*.json")):
    data = json.loads(file.read_text(encoding="utf-8"))
    obj = np.asarray(data["objectives"], float)
    constraints = np.asarray(data["constraint_values"], float)
    feasible = np.all(constraints <= 0, axis=1)
    records.append(
        {
            "seed": int(data["seed"]),
            "seed_file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "return_population_fallback": bool(data["returned_population_fallback"]),
            "feasible_count": int(feasible.sum()),
            "feasible_objectives": obj[feasible].tolist(),
        }
    )
if len(records) != 10:
    raise RuntimeError("Require all ten independently seeded smoke results")
if sum(r["feasible_count"] > 0 for r in records) != summary["agreement"]["feasible_run_count"]:
    raise AssertionError("Feasibility mismatches agreement")
report = {
    "schema": "corrected-smoke-pareto-figure-v1",
    "publication_scope": "Corrected candidate-specific 10-seed 32-pop x 10-gen x 16-sample SMOKE only; not production Pareto optimality.",
    "protocol": summary["profile"],
    "agreement": summary["agreement"],
    "seeds": records,
}
out = ROOT_OUT / "CORRECTED_SMOKE_FRONT.json"
out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(
    "Feasible front entries",
    sum(x["feasible_count"] for x in records),
    "seeds",
    len(records),
    "distance",
    summary["agreement"]["pairwise_objective_distance_mean"],
)
