"""Lineage-bound frozen-model evidence for the Robust V2 ML acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from robust_airfoil.hashing import sha256_file
from robust_airfoil.ml_evidence import paired_group_bootstrap
from robust_airfoil.provenance import utc_now, write_json_atomic


def _macro_mse(frame: pd.DataFrame, target: str, prediction: str) -> float:
    valid = frame[f"mask_{target}"].astype(bool) & frame[target].notna() & frame[prediction].notna()
    squared = (frame.loc[valid, target].to_numpy(float) - frame.loc[valid, prediction].to_numpy(float)) ** 2
    groups = frame.loc[valid, "airfoil_id"].astype(str).to_numpy()
    if not len(squared):
        raise ValueError(f"Locked test has no resolved {target} labels")
    return float(pd.DataFrame({"group": groups, "loss": squared}).groupby("group")["loss"].mean().mean())


def _target_evidence(
    locked: pd.DataFrame,
    development: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    valid_development = development.loc[development[f"mask_{target}"].astype(bool), target]
    if valid_development.empty:
        raise ValueError(f"Development split has no resolved {target} labels")
    dummy_column = f"dummy_{target}"
    evaluated = locked.copy()
    evaluated[dummy_column] = float(valid_development.mean())
    model_column = f"prediction_{target}"
    model_error = _macro_mse(evaluated, target, model_column)
    dummy_error = _macro_mse(evaluated, target, dummy_column)
    valid = (
        evaluated[f"mask_{target}"].astype(bool)
        & evaluated[target].notna()
        & evaluated[model_column].notna()
    )
    truth = evaluated.loc[valid, target].to_numpy(float)
    model_losses = (evaluated.loc[valid, model_column].to_numpy(float) - truth) ** 2
    dummy_losses = (evaluated.loc[valid, dummy_column].to_numpy(float) - truth) ** 2
    groups = evaluated.loc[valid, "airfoil_id"].astype(str).tolist()
    return {
        "metric": "equal-airfoil-weight macro MSE on the locked test split",
        "evaluated_rows": int(valid.sum()),
        "independent_airfoil_groups": int(len(set(groups))),
        "model_macro_error": model_error,
        "dummy_macro_error": dummy_error,
        "macro_error_improvement_vs_dummy": (dummy_error - model_error) / dummy_error,
        "paired_group_bootstrap_model_minus_dummy": paired_group_bootstrap(
            model_losses, dummy_losses, groups, repetitions=2000
        ),
    }


def build_ml_evidence(
    results_root: Path,
    reports_root: Path,
    model_points_path: Path,
    protocol_hash: str,
) -> dict[str, Any]:
    """Evaluate only the frozen ensemble's serialized locked-test predictions.

    The command neither retrains the model nor accesses test labels for model
    selection. It records a narrow acceptance gate and explicitly leaves wider
    experimental claims to separately registered evidence.
    """
    manifest_path = results_root / "FROZEN_MODEL_MANIFEST.json"
    predictions_path = results_root / "locked_test/predictions.parquet"
    split_path = results_root / "splits/full.json"
    required = [manifest_path, predictions_path, split_path, model_points_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen-model evidence requires: {missing}")

    split_records = json.loads(split_path.read_text(encoding="utf-8"))["records"]
    development_ids = {
        str(record["airfoil_id"])
        for record in split_records
        if record["split"] == "development"
    }
    if not development_ids:
        raise ValueError("Split manifest has no development airfoils")
    points = pd.read_parquet(model_points_path)
    development: pd.DataFrame = points.loc[
        points["airfoil_id"].astype(str).isin(sorted(development_ids))
    ].copy()
    locked = pd.read_parquet(predictions_path)
    if locked.empty:
        raise ValueError("Locked-test prediction artifact is empty")
    for target in ("cl", "log_cd", "cm"):
        column = f"prediction_{target}"
        if column not in locked:
            raise ValueError(f"Locked-test prediction artifact is missing {column}")

    target_metrics = {target: _target_evidence(locked, development, target) for target in ("cl", "log_cd", "cm")}
    gate_passed = all(
        target_metrics[target]["macro_error_improvement_vs_dummy"] >= 0.20
        for target in ("cl", "log_cd")
    )
    calibration_path = results_root / "calibration/trust_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else {}
    output = reports_root / "modelling/ml_evidence.json"
    payload: dict[str, Any] = {
        "schema_version": "robust-v2-ml-evidence-v1",
        "status": "passed" if gate_passed else "held",
        "ml_gate_passed": gate_passed,
        "scientific_go_granted": False,
        "model_id": "frozen-cluster-bootstrap-ensemble",
        "model_sha256": sha256_file(manifest_path),
        "protocol_hash": protocol_hash,
        "input_sha256": sha256_file(model_points_path),
        "split_manifest_sha256": sha256_file(split_path),
        "locked_prediction_sha256": sha256_file(predictions_path),
        "evaluation_scope": "frozen model; development-only Dummy baseline; untouched locked test",
        "target_metrics": target_metrics,
        "trust_domain": {
            "locked_test_trusted_fraction": float(locked["trusted_domain"].mean())
            if "trusted_domain" in locked
            else None,
            "calibration_empirical_95_interval_coverage": calibration.get(
                "empirical_95_interval_coverage", {}
            ),
            "calibration_trusted_fraction": calibration.get("trusted_calibration_fraction"),
        },
        "registered_evidence_scope": {
            "acceptance_gate": "completed",
            "learning_curve_10_25_50_100_200_full_three_seeds": "not_run",
            "external_held_out_benchmark": "not_run",
            "matched_solver_cost_study": "not_run",
            "reason": "This artifact does not substitute unmeasured evidence with proxy claims.",
        },
        "created_utc": utc_now(),
    }
    write_json_atomic(output, payload)
    return payload
