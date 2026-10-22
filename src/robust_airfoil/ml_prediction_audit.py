"""Audit supplied prediction tables without generating, filling, or relabelling data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robust_airfoil.ml_evidence import selective_regression_report

REQUIRED_COLUMNS = {
    "case_id",
    "cluster_id",
    "model_id",
    "model_sha256",
    "protocol_hash",
    "pool_role",
    "target",
    "reference_value",
    "prediction_value",
    "uncertainty_score",
    "accepted_by_trust",
    "reference_complete",
    "solver_status",
}
_POOL_ROLES = {
    "development_validation",
    "calibration",
    "diagnostic",
    "confirmation",
    "historical_exposed",
}
_COMPLETE_SOLVER_STATUSES = {"xfoil_complete", "archived_reference_available"}


def boolean(value: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"true", "1"}:
        return True
    if normalised in {"false", "0"}:
        return False
    raise ValueError(f"Invalid Boolean: {value!r}")


def number(value: str | None) -> float:
    return float(value) if value and value.strip() else float("nan")


def run(input_csv: Path, output: Path, target: str, tolerance: float) -> dict[str, Any]:
    """Write one empirical audit record for one model/target/protocol/pool table."""
    if input_csv.resolve() == output.resolve():
        raise ValueError("Output must not overwrite its prediction input")
    with input_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows = [row for row in reader if row["target"] == target]
    if not rows:
        raise ValueError("No rows match the requested target")
    for key in ("model_id", "model_sha256", "protocol_hash", "pool_role", "target"):
        values = {row[key] for row in rows}
        if len(values) != 1 or not all(values):
            raise ValueError(f"Expected exactly one nonempty {key}")
    for key in ("model_sha256", "protocol_hash"):
        if not re.fullmatch(r"[0-9a-f]{64}", rows[0][key]):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    if rows[0]["pool_role"] not in _POOL_ROLES:
        raise ValueError("Unrecognised pool role")
    case_ids = [row["case_id"] for row in rows]
    if not all(case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique and nonempty within target")
    complete = [boolean(row["reference_complete"]) for row in rows]
    for row, complete_reference in zip(rows, complete, strict=True):
        if complete_reference and row["solver_status"] not in _COMPLETE_SOLVER_STATUSES:
            raise ValueError("Reference completion conflicts with solver status")
    has_lower = "interval_lower" in rows[0]
    has_upper = "interval_upper" in rows[0]
    if has_lower != has_upper:
        raise ValueError("Both interval columns must be supplied")
    diagnostics = selective_regression_report(
        [number(row["reference_value"]) for row in rows],
        [number(row["prediction_value"]) for row in rows],
        [number(row["uncertainty_score"]) for row in rows],
        [row["cluster_id"] for row in rows],
        [boolean(row["accepted_by_trust"]) for row in rows],
        complete,
        tolerance=tolerance,
        interval_lower=[number(row["interval_lower"]) for row in rows] if has_lower else None,
        interval_upper=[number(row["interval_upper"]) for row in rows] if has_upper else None,
    )
    record = {
        "schema_version": "ml-prediction-audit-v1",
        "executed_utc": datetime.now(UTC).isoformat(),
        "input_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "source_filename": input_csv.name,
        "target": target,
        "model_id": rows[0]["model_id"],
        "model_sha256": rows[0]["model_sha256"],
        "protocol_hash": rows[0]["protocol_hash"],
        "pool_role": rows[0]["pool_role"],
        "audit_scope": "empirical_diagnostics_of_provided_rows_not_label_provenance_certification",
        "data_reuse_authorised": False,
        "scientific_go_granted": False,
        "diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["cl", "log_cd", "cd", "cm"], required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments.input, arguments.output, arguments.target, arguments.tolerance)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(json.dumps({"status": "audit_refused", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "diagnostics_written",
                "output": str(arguments.output),
                "attempted_requests": result["diagnostics"]["attempted_requests"],
                "scientific_go_granted": False,
            },
            indent=2,
        )
    )
    return 0
