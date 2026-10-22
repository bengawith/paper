"""Independent read-only audit of the completed matched nominal-control campaign.

This is NOT a second optimisation or a substitute for the original immutable
seed records. It reconciles ten terminal finite-budget receipts after the
separately preserved R3 Windows progress-file sharing failure/R4 recovery.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
FOLDER = LINEAGE / "nominal_control"
REPORT = HERE / "MATCHED_NOMINAL_TEN_SEED_INDEPENDENT_RECONCILIATION.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict:
    summary_file = FOLDER / "NOMINAL_CONTROL_SUMMARY.json"
    protocol_file = FOLDER / "NOMINAL_CONTROL_PROTOCOL.json"
    robust_file = LINEAGE / "optimisation_production/PRODUCTION_SUMMARY.json"
    robust_protocol_file = LINEAGE / "optimisation_production/PRODUCTION_PROTOCOL.json"
    summary = load(summary_file)
    protocol = load(protocol_file)
    robust = load(robust_file)
    robust_protocol = load(robust_protocol_file)
    r3_file = HERE / "CONTROLLED_PIPELINE_STATUS_R3.json"
    r4_file = HERE / "CONTROLLED_PIPELINE_STATUS_R4.json"
    r3 = load(r3_file)
    r4 = load(r4_file)

    if summary.get("status") != "FULL_NOMINAL_PROFILE_COMPLETED":
        raise ValueError("Nominal control did not finish.")
    if summary["profile"] != {"population": 128, "generations": 200, "independent_seeds": 10}:
        raise ValueError("Matched nominal profile was changed.")
    if len(summary["per_seed"]) != 10 or len(robust["per_seed"]) != 10:
        raise ValueError("Full ten-seed study is missing.")
    if r3.get("status") != "STOPPED_FOR_REVIEW" or r3["stages"]["nominal_control"]["status"] != "FAILED":
        raise ValueError("Original R3 failure must remain recorded and unmodified.")
    if r4.get("status") != "ALL_EXECUTED_GATES_PASS_NOT_INDEPENDENT_CFD_OR_AUTHOR_SUBMISSION":
        raise ValueError("R4 recovery gate is not terminal.")
    if any(s.get("status") != "PASSED" for s in r4["stages"].values()):
        raise ValueError("R4 has a nonpassing scientific gate.")
    if summary["context_hash"] != protocol["context_hash"]:
        raise ValueError("Nominal source context hash mismatch.")
    if protocol["context"]["scientific_context"]["robust_summary_sha256"] != sha(robust_file):
        raise ValueError("Robust summary does not match nominal preselection inputs.")
    if protocol["context"]["scientific_context"]["robust_protocol_sha256"] != sha(robust_protocol_file):
        raise ValueError("Robust protocol does not match nominal scientific context.")

    src = ROOT / "src/robust_airfoil"
    for source, expected in robust_protocol["context"]["source_sha256"].items():
        if sha(src / source) != expected:
            raise ValueError("Frozen scientific source changed: " + source)
    if sha(src / "optimisation/nominal_control.py") != protocol["context"]["operator_sha256"]:
        raise ValueError("Nominal operator source has changed.")

    rows = []
    for index, record in enumerate(summary["per_seed"]):
        seed = 20264824 + index
        if record["seed"] != seed or record["generations"] != 200:
            raise ValueError("Nominal seed order or budget changed.")
        case_file = FOLDER / record["file"]
        if sha(case_file) != record["sha256"]:
            raise ValueError("Nominal result content hash mismatch.")
        run = load(case_file)
        history = run["history"]
        if run["seed"] != seed or len(history) != 200:
            raise ValueError("Nominal full generation record incomplete.")
        if run["context_hash"] != summary["context_hash"]:
            raise ValueError("Per-seed scientific context mismatch.")
        if [h["generation"] for h in history] != list(range(1, 201)):
            raise ValueError("Generation provenance has a gap.")
        if [h["evaluations"] for h in history] != [128 * n for n in range(1, 201)]:
            raise ValueError("Unequal evolutionary-budget count.")
        if any(h["population_count"] != 128 for h in history):
            raise ValueError("Generation did not retain declared population.")
        x = np.asarray(run["solutions"], dtype=float)
        f = np.asarray(run["objectives"], dtype=float)
        g = np.asarray(run["constraint_values"], dtype=float)
        if x.shape != (128, 12) or f.shape != (128, 1) or g.shape != (128, 6):
            raise ValueError("Nominal final population shape mismatch.")
        feas = np.isfinite(f[:, 0]) & np.isfinite(g).all(axis=1) & np.all(g <= 0, axis=1)
        count = int(feas.sum())
        if count != record["feasible_entries"] or count != run["feasible_count"]:
            raise ValueError("Recorded final feasible denominator disagrees.")
        if not np.isfinite(run["final_recheck_max_objective_difference"]):
            raise ValueError("Missing nominal post-hoc objective re-evaluation.")
        minimum = float(np.min(f[feas, 0])) if count else None
        if minimum != record["minimum_feasible_nominal_drag"]:
            raise ValueError("Reported feasible objective minimum differs from source.")
        rows.append({
            "seed": seed, "completed_generations": 200, "candidate_evaluations": 25600,
            "feasible_final_entries": count, "minimum_feasible_nominal_drag": minimum,
            "maximum_nominal_recheck_difference": run["final_recheck_max_objective_difference"],
            "source_sha256": record["sha256"],
        })
    outcome = {
        "schema": "matched-nominal-independent-readonly-reconciliation-v1",
        "nominal_summary_sha256": sha(summary_file),
        "nominal_protocol_sha256": sha(protocol_file),
        "robust_summary_sha256": sha(robust_file),
        "r3_preserved_failure_receipt_sha256": sha(r3_file),
        "r4_terminal_recovery_receipt_sha256": sha(r4_file),
        "seeds": len(rows),
        "completed_generations": sum(x["completed_generations"] for x in rows),
        "candidate_evaluations": sum(x["candidate_evaluations"] for x in rows),
        "feasible_seed_count": sum(x["feasible_final_entries"] > 0 for x in rows),
        "feasible_final_entries": sum(x["feasible_final_entries"] for x in rows),
        "minimum_final_feasible_nominal_drag": min(x["minimum_feasible_nominal_drag"] for x in rows if x["minimum_feasible_nominal_drag"] is not None),
        "source_hashes_unchanged_since_robust_campaign": True,
        "r3_windows_checkpoint_failure_preserved": True,
        "scope": "Independent finite-algorithm receipt reconciliation, not independent CFD, validated manufacturing probabilities or a global optimum.",
        "per_seed": rows,
    }
    return outcome


if __name__ == "__main__":
    if REPORT.exists():
        raise FileExistsError("Independent audit receipt already exists.")
    result = audit()
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))
