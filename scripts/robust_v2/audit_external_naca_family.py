"""Independent receipt/case/missingness audit of analytic family challenge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/robust_v2/final_release_20261022/EXTERNAL_NACA_FAMILY"
OUT = BASE / "INDEPENDENT_RECEIPT_AND_TRUST_AUDIT.json"
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Prior independent external-family audit already exists")
    selection_path = BASE / "GEOMETRY_SELECTION_BEFORE_LABELS.json"
    prediction_path = BASE / "PREDICTIONS_BEFORE_XFOIL_LABELS.json"
    result_path = BASE / "NEW_FAMILY_RESULTS.json"
    source_path = BASE / "ALL_40_SOLVER_RESULTS.json"
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    before = json.loads(prediction_path.read_text(encoding="utf-8"))
    observed = json.loads(result_path.read_text(encoding="utf-8"))
    solver = json.loads(source_path.read_text(encoding="utf-8"))
    if selected["accepted"] != 40 or selected["original_1256_source_sha256"] != sha(
        ROOT / "data/robust_v2/processed/controlled_20261020/airfoils.parquet"
    ):
        raise ValueError("Selected external geometry population changed")
    if before["geometry_selection_sha256"] != sha(selection_path) or before[
        "prediction_file_sha256"
    ] != sha(BASE / "before_label_model_predictions.parquet"):
        raise ValueError("Before-label frozen model prediction seal invalid")
    if before["frozen_ensemble_manifest_sha256"] != sha(LINEAGE / "FROZEN_MODEL_MANIFEST.json"):
        raise ValueError("Model was not frozen at external prediction time")
    if (
        source_path.stat().st_size == 0
        or len(solver["cases"]) != 40
        or solver["requested_states"] != 4520
        or observed["requested_states"] != 4520
        or observed["observed_states"] != solver["observed_states"]
    ):
        raise ValueError("Solver source accounting incomplete")
    all_shape = []
    for candidate, receipt in zip(selected["cases"], solver["cases"], strict=True):
        if candidate["id"] != receipt["id"]:
            raise ValueError("Receipt order differs from original source seal")
        if candidate["min_rms_to_all_existing_1256"] < selected["min_archive_rms_chord"]:
            raise ValueError("Source geometry is not genuinely separated")
        geometry = ROOT / candidate["geometry_path"]
        if sha(geometry) != candidate["geometry_sha256"]:
            raise ValueError("Source geometry bytes changed")
        p = BASE / "solver" / candidate["id"]
        cached = json.loads((p / "CASE_RECEIPT.json").read_text(encoding="utf-8"))
        if cached != receipt:
            raise ValueError("Individual case receipt no longer equals full frozen summary")
        for path, expected in receipt["files_sha256"].items():
            if sha(p / path) != expected:
                raise ValueError("Case file changed " + path)
        actual = pd.read_parquet(p / "observed.parquet")
        if len(actual) != receipt["observed"]:
            raise ValueError("Case admitted numerical count changed")
        if len(actual):
            all_shape.append(actual)
    truth = pd.concat(all_shape, ignore_index=True)
    forecast = pd.read_parquet(BASE / "before_label_model_predictions.parquet")
    joined = forecast.merge(
        truth[["airfoil_id", "alpha_deg", "cl", "cd", "cm"]],
        on=["airfoil_id", "alpha_deg"],
        how="left",
        validate="one_to_one",
    )
    observed_mask = joined["cl"].notna().to_numpy()
    if observed_mask.sum() != observed["observed_states"]:
        raise ValueError("Changed numerical missingness on external-family cohort")
    trusted = joined["trusted_domain"].to_numpy(bool)
    available = joined.loc[observed_mask]
    groups = {
        "accepted_observed": available.loc[available["trusted_domain"]],
        "rejected_observed": available.loc[~available["trusted_domain"]],
    }
    selective = {}
    for name, df in groups.items():
        selective[name] = {
            "states": len(df),
            "geometries": int(df.airfoil_id.nunique()),
            "cl_mae": float(np.abs(df["prediction_cl"] - df["cl"]).mean()) if len(df) else None,
            "cd_mae": float(np.abs(np.exp(df["prediction_log_cd"]) - df["cd"]).mean())
            if len(df)
            else None,
            "cm_mae": float(np.abs(df["prediction_cm"] - df["cm"]).mean()) if len(df) else None,
        }
    byid = before["comparison_neuralfoil"]
    confidences = []
    for case in selected["cases"]:
        row = byid[case["id"]]
        if row["status"] == "ok" and len(row["analysis_confidence"]) == 113:
            confidences.extend(row["analysis_confidence"])
        else:
            confidences.extend([np.nan] * 113)
    # Forecast dataframe was built geometry-then-increasing-alpha order.
    if len(confidences) != len(forecast):
        raise ValueError("External-family NeuralFoil confidence cardinality mismatch")
    # Validate exact order before numerical comparisons.
    expected_order = [
        (candidate["id"], float(alpha))
        for candidate in selected["cases"]
        for alpha in np.arange(-12, 16.0001, 0.25)
    ]
    actual_order = list(
        zip(forecast.airfoil_id.astype(str), forecast.alpha_deg.to_numpy(float), strict=True)
    )
    if expected_order != actual_order:
        raise ValueError("Geometry-incidence order used for confidence is inconsistent")
    missing = ~observed_mask
    nf_conf = np.asarray(confidences, float)
    valid = np.isfinite(nf_conf)
    risk_scores = {
        "neuralfoil_negative_analysis_confidence": float(
            roc_auc_score(missing[valid], -nf_conf[valid])
        ),
        "controlled_ensemble_std_cl": float(
            roc_auc_score(missing, joined["ensemble_std_cl"].to_numpy(float))
        ),
        "controlled_ensemble_std_log_cd": float(
            roc_auc_score(missing, joined["ensemble_std_log_cd"].to_numpy(float))
        ),
        "controlled_nearest_support_distance": float(
            roc_auc_score(missing, joined["support_distance"].to_numpy(float))
        ),
    }
    record = {
        "schema": "independent-analytic-40shape-source-preservation-v1",
        "source_seal_sha256": sha(selection_path),
        "before_label_predictor_receipt_sha256": sha(prediction_path),
        "complete_solver_receipt_sha256": sha(source_path),
        "external_summary_sha256": sha(result_path),
        "original_1256_geometry_catalog_unchanged": True,
        "all_40_geometry_receipts_hash_verified": True,
        "unavailable_solver_states_not_imputed": True,
        "requested_geometries": 40,
        "requested_states": 4520,
        "solver_observed_states": int(observed_mask.sum()),
        "solver_missing_states": int(missing.sum()),
        "selected_geometry_min_source_rms_chord": float(
            min(record["min_rms_to_all_existing_1256"] for record in selected["cases"])
        ),
        "empirical_trust_fraction_all_requests": float(trusted.mean()),
        "empirical_trust_fraction_observed": float(trusted[observed_mask].mean()),
        "selective_prediction_errors": selective,
        "solver_unavailable_ranking_auroc": risk_scores,
        "limitations": [
            "The analytic family was chosen geometrically after previous research but before new solver labels.",
            "This is numerical source geometry distinctness, not proof of NeuralFoil training-family independence.",
            "XFOIL remains the same numerical solver family as training and is not independent CFD.",
            "Model trust and confidence scores are descriptive, not calibrated physical probabilities.",
        ],
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "state_count": record["solver_observed_states"],
                "missing": record["solver_missing_states"],
                "min_rms": record["selected_geometry_min_source_rms_chord"],
                "accepted": selective,
                "solver_availability": risk_scores,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
