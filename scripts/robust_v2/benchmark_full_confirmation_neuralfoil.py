"""All 22,600 presealed confirmation queries versus fixed NeuralFoil xlarge.

Strict XFOIL missing states remain missing in *both* methods' accuracy denominators.
The original five-member predictions/risk scores are read before-label frozen;
no hyperparameters, thresholds or neural checkpoints are fitted to these data.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from robust_airfoil.benchmarks.neuralfoil import benchmark_neuralfoil_from_coordinates
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.provenance import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020/confirmation"
OUT = ROOT / "reports/robust_v2/final_release_20261022/NEURALFOIL_FULL_CONFIRMATION"
PRED = BASE / "predictions_before_labels.parquet"
OBS = BASE / "observed_confirmation_points.parquet"
PARAMS = FEATURE_COLUMNS[:-1]
FROZEN = BASE / "PREDICTIONS_BEFORE_LABELS_RECEIPT.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def main() -> None:
    import aerosandbox
    import neuralfoil

    OUT.mkdir(parents=True, exist_ok=True)
    complete = OUT / "RESULTS.json"
    selections = OUT / "FROZEN_COMPARISON_INPUTS.json"
    runs = OUT / "FULL_POLAR_NEURALFOIL_CASES.jsonl"
    if complete.exists():
        raise FileExistsError("Cannot regenerate terminal comparator evidence in place")
    frame = pd.read_parquet(PRED)
    truths = pd.read_parquet(OBS)
    expected = frame["airfoil_id"].nunique()
    if expected != 200 or len(frame) != 22600 or len(truths) != 17968:
        raise ValueError("Confirmation cohort cardinality unexpectedly changed")
    identities = sorted(frame["airfoil_id"].astype(str).unique())
    frozen = {
        "schema": "full-sealed-200geometry-neuralfoil-xlarge-v1",
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha(path) for path in (PRED, OBS, FROZEN)
        },
        "identities": identities,
        "request_count": len(frame),
        "model_size": "xlarge",
        "Re": 1e6,
        "Ncrit": 9.0,
        "displacement": "exact frozen CST model input, refitted to NeuralFoil 18-parameter Kulfan",
        "not_fitted_on_confirmation_labels": True,
        "not_independent_cfd": True,
    }
    if selections.exists():
        if json.loads(selections.read_text(encoding="utf-8")) != frozen:
            raise ValueError("Comparison selection changed since first run")
    else:
        write_json_atomic(selections, frozen)
    cached = (
        [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
        if runs.exists()
        else []
    )
    if len(cached) > 200 or any(
        r["airfoil_id"] != identity for r, identity in zip(cached, identities, strict=False)
    ):
        raise ValueError("Incompatible resumed NeuralFoil case receipts")
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    started = time.perf_counter()
    with runs.open("a", encoding="utf-8") as writer:
        for n, name in enumerate(identities[len(cached) :], len(cached)):
            rows = frame.loc[frame["airfoil_id"] == name].sort_values("alpha_deg")
            if len(rows) != 113:
                raise ValueError("Incomplete request geometry " + name)
            coefficients = rows.iloc[0][PARAMS].to_numpy(float)
            upper, lower = reconstruct_cst(coefficients, x)
            points = np.vstack(
                [np.column_stack([x[::-1], upper[::-1]]), np.column_stack([x[1:], lower[1:]])]
            )
            begun = time.perf_counter()
            nf = benchmark_neuralfoil_from_coordinates(
                points, rows["alpha_deg"].to_numpy(float), reynolds_number=1e6
            )
            record = {
                "airfoil_id": name,
                "status": nf.status,
                "reason": nf.reason,
                "seconds_including_NeuralFoil_shape_refit": time.perf_counter() - begun,
            }
            if nf.status == "ok":
                for key in ("cl", "cd", "cm", "analysis_confidence"):
                    attr = getattr(nf, key)
                    record[key] = np.asarray(attr).tolist() if attr is not None else None
                if any(len(record[key]) != 113 for key in ("cl", "cd", "cm")):
                    record["status"] = "length_mismatch"
                elif not all(np.isfinite(record[key]).all() for key in ("cl", "cd", "cm")):
                    record["status"] = "nonfinite"
                elif not all(v > 0 for v in record["cd"]):
                    record["status"] = "nonpositive_drag"
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            if (n + 1) % 25 == 0:
                print(
                    "NF_FULL_CONFIRMATION",
                    n + 1,
                    "/200 elapsed",
                    round(time.perf_counter() - started, 2),
                    flush=True,
                )
    receipts = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    if len(receipts) != 200:
        raise AssertionError("Every preregistered confirmation geometry must have a receipt")
    mapping = {}
    for record, name in zip(receipts, identities, strict=True):
        if record["airfoil_id"] != name:
            raise AssertionError("Baseline result geometry identity corrupted")
        if record["status"] != "ok":
            continue
        rows = frame.loc[frame["airfoil_id"] == name].sort_values("alpha_deg")
        mapping[name] = pd.DataFrame(
            {
                "airfoil_id": name,
                "alpha_deg": rows["alpha_deg"].to_numpy(float),
                "neuralfoil_cl": record["cl"],
                "neuralfoil_cd": record["cd"],
                "neuralfoil_cm": record["cm"],
                "neuralfoil_confidence": record["analysis_confidence"],
            }
        )
    nf_frame = pd.concat(mapping.values(), ignore_index=True) if mapping else pd.DataFrame()
    final = frame.merge(nf_frame, on=["airfoil_id", "alpha_deg"], how="left", validate="one_to_one")
    seen = set(
        zip(truths["airfoil_id"].astype(str), truths["alpha_deg"].astype(float), strict=True)
    )
    final["_xfoil_observed"] = [
        (str(id), float(alpha)) in seen
        for id, alpha in zip(final["airfoil_id"], final["alpha_deg"], strict=True)
    ]
    if int(final["_xfoil_observed"].sum()) != len(truths):
        raise AssertionError("Solver-observation missingness changed on matched join")
    obs_frame = final.loc[final["_xfoil_observed"]].merge(
        truths[["airfoil_id", "alpha_deg", "cl", "cd", "cm", "log_cd"]],
        on=["airfoil_id", "alpha_deg"],
        validate="one_to_one",
    )
    fullpaired = obs_frame.dropna(subset=["neuralfoil_cl", "neuralfoil_cd", "neuralfoil_cm"])
    stats = {}
    for target in ("cl", "log_cd", "cm", "cd"):
        actual = fullpaired[target].to_numpy(float)
        baseline = (
            np.log(fullpaired["neuralfoil_cd"].to_numpy(float))
            if target == "log_cd"
            else fullpaired["neuralfoil_" + target].to_numpy(float)
        )
        ours = (
            np.exp(fullpaired["prediction_log_cd"].to_numpy(float))
            if target == "cd"
            else fullpaired["prediction_" + target].to_numpy(float)
        )
        groups = fullpaired["airfoil_id"].astype(str)
        pergeo = (
            fullpaired.assign(_diff=np.abs(baseline - actual) - np.abs(ours - actual))
            .groupby(groups)["_diff"]
            .mean()
        )
        rng = np.random.default_rng(20261022 + len(stats))
        bootstrap = pergeo.to_numpy()[rng.integers(0, len(pergeo), size=(3000, len(pergeo)))].mean(
            axis=1
        )
        stats[target] = {
            "controlled_mae": float(np.abs(ours - actual).mean()),
            "neuralfoil_mae": float(np.abs(baseline - actual).mean()),
            "neuralfoil_minus_controlled_geometry_macro_mae": float(pergeo.mean()),
            "fixed_model_geometry_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "paired_observed_states": len(fullpaired),
        }

    def auc(score, missing):
        valid = np.isfinite(score)
        return {
            "auroc": float(roc_auc_score(np.asarray(missing)[valid], np.asarray(score)[valid]))
            if len(set(np.asarray(missing)[valid])) > 1
            else None,
            "eligible_requested_states": int(valid.sum()),
            "unscored_requested_states": int((~valid).sum()),
        }

    missing = ~final["_xfoil_observed"].to_numpy(bool)
    nf_confidence = final["neuralfoil_confidence"].to_numpy(float)
    support_risk = final["risk_support_only"].to_numpy(float)
    ensemble_risk = final["risk_ensemble_only"].to_numpy(float)
    combined_risk = final["risk_support_plus_ensemble"].to_numpy(float)
    missing_auc = {
        "neuralfoil_minus_confidence": auc(-nf_confidence, missing),
        "our_frozen_support_risk": auc(support_risk, missing),
        "our_frozen_ensemble_disagreement_risk": auc(ensemble_risk, missing),
        "our_frozen_combined_risk": auc(combined_risk, missing),
    }
    independent = json.loads(
        (
            ROOT
            / "reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/CONFIRMATION/CONFIRMATION_RESULTS.json"
        ).read_text(encoding="utf-8")
    )
    expected_auc = independent["risk_vs_missing_solver_observation_diagnostic"]["ensemble_only"][
        "auroc"
    ]
    if abs(missing_auc["our_frozen_ensemble_disagreement_risk"]["auroc"] - expected_auc) > 1e-9:
        raise ValueError("Frozen missing-XFOIL risk AUROC failed independent reconciliation")
    result = {
        "schema": "all-22600-prerequested-200geometry-neuralfoil-diagnostic-v1",
        "preselection_sha256": sha(selections),
        "case_receipts_sha256": sha(runs),
        "original_prelabel_prediction_sha256": sha(PRED),
        "original_solver_observed_sha256": sha(OBS),
        "neuralfoil_version": getattr(neuralfoil, "__version__", "0.3.3"),
        "aerosandbox_version": getattr(aerosandbox, "__version__", "4.2.10"),
        "geometries_requested": 200,
        "geometries_completed_by_neuralfoil": len(mapping),
        "geometries_failed_by_neuralfoil": 200 - len(mapping),
        "states_requested": 22600,
        "solver_observed_states": 17968,
        "solver_missing_states": 4632,
        "same_solver_observed_paired_states": len(fullpaired),
        "per_target_observed_label_error": stats,
        "predicting_xfoil_unavailability_auroc": missing_auc,
        "neuralfoil_geometry_refit_included_in_timing_seconds": float(
            sum(x["seconds_including_NeuralFoil_shape_refit"] for x in receipts)
        ),
        "limits": [
            "Cross-model comparison measures errors against local XFOIL family only, not CFD truth.",
            "NeuralFoil internal training shape overlap is not established; never claim blind external families.",
            "The frozen confirmation shapes are development-geometry mixtures, not unrelated families.",
            "The trust-score AUROC predicts local solver availability, not real physical model validity.",
            "Model speed is not compared here on strictly identical input and hardware workflow costs.",
            "Only XFOIL-observed cases enter prediction accuracy metrics; no imputed truth for missing.",
        ],
    }
    write_json_atomic(complete, result)
    print(
        "DONE_FULL_COMPARISON",
        json.dumps(
            {
                "paired_states": len(fullpaired),
                "neuralfoil_successful_geometries": len(mapping),
                "mae": {
                    name: {
                        k: v for k, v in stat.items() if k in ("controlled_mae", "neuralfoil_mae")
                    }
                    for name, stat in stats.items()
                },
                "solver_missingness_auroc": missing_auc,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
