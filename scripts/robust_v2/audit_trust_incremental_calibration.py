"""Five-fold group cross-fitted calibration-only ensemble-disagreement incremental audit.

All scoring thresholds, target scales, feature scales and ridge fits use ONLY
the training geometries of each fold. The frozen locked test is not accessed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CAL = (
    ROOT
    / "results/robust_v2/lineages/robust-v2-current-20260908/calibration/calibration_predictions.parquet"
)
OUT = ROOT / "reports/robust_v2/submission_review_20261014/TRUST_INCREMENTAL"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_parquet(CAL)
FEATURES = {
    "support_only": ["support_distance"],
    "support_plus_ensemble": [
        "support_distance",
        "ensemble_std_cl",
        "ensemble_std_log_cd",
        "ensemble_std_cm",
    ],
    "ensemble_only": ["ensemble_std_cl", "ensemble_std_log_cd", "ensemble_std_cm"],
}
TARGETS = ["cl", "log_cd", "cm"]
features = d[FEATURES["support_plus_ensemble"]].to_numpy(float)
errors = np.column_stack(
    [np.abs(d[t].to_numpy(float) - d["prediction_" + t].to_numpy(float)) for t in TARGETS]
)
valid = np.isfinite(features).all(axis=1) & np.isfinite(errors).all(axis=1)
d = d.loc[valid].reset_index(drop=True)
errors = errors[valid]
groups = d.airfoil_id.astype(str).to_numpy()
folds = []
cv = GroupKFold(n_splits=5)
oof = {key: np.full(len(d), np.nan) for key in FEATURES}
scores = np.full(len(d), np.nan)
large = np.zeros(len(d), dtype=int)
for fold, (train_idx, val_idx) in enumerate(cv.split(d, groups=groups), start=1):
    # Crucial: do not derive target scaling or risk threshold from val errors.
    train_scale = np.maximum(np.quantile(errors[train_idx], 0.90, axis=0), 1e-8)
    train_score = np.max(errors[train_idx] / train_scale[None, :], axis=1)
    val_score = np.max(errors[val_idx] / train_scale[None, :], axis=1)
    risk_threshold = np.quantile(train_score, 0.90)
    scores[val_idx] = val_score
    large[val_idx] = (val_score >= risk_threshold).astype(int)
    for name, cols in FEATURES.items():
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        model.fit(d.iloc[train_idx][cols].to_numpy(float), np.log1p(train_score))
        oof[name][val_idx] = model.predict(d.iloc[val_idx][cols].to_numpy(float))
    folds.append(
        {
            "fold": fold,
            "train_geometries": len(set(groups[train_idx])),
            "evaluation_geometries": len(set(groups[val_idx])),
            "train_only_target_error_p90_scaling": train_scale.tolist(),
            "train_only_large_error_threshold": float(risk_threshold),
            "evaluation_high_error_rate": float(large[val_idx].mean()),
        }
    )
if not np.isfinite(scores).all() or not any(large) or all(large):
    raise RuntimeError("Invalid group OOF construction")
res = {}
for name, pred in oof.items():
    assert np.isfinite(pred).all()
    ordered = np.argsort(pred)
    count = max(1, int(0.5 * len(d)))
    res[name] = {
        "auroc_for_fold_train_defined_high_error": float(roc_auc_score(large, pred)),
        "average_precision": float(average_precision_score(large, pred)),
        "half_coverage_geometry_macro_error": float(
            pd.DataFrame({"airfoil": groups[ordered[:count]], "error": scores[ordered[:count]]})
            .groupby("airfoil")
            .error.mean()
            .mean()
        ),
        "spearman_error_correlation": float(
            pd.Series(pred).corr(pd.Series(scores), method="spearman")
        ),
    }
delta = (
    res["support_plus_ensemble"]["auroc_for_fold_train_defined_high_error"]
    - res["support_only"]["auroc_for_fold_train_defined_high_error"]
)
rng = np.random.default_rng(20261014)
unique, group_indices = np.unique(groups, return_inverse=True)
boot = []
for _ in range(300):
    weights_per_group = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
    weights = weights_per_group[group_indices]
    if np.sum(weights[large == 1]) == 0 or np.sum(weights[large == 0]) == 0:
        continue
    with_spread = roc_auc_score(large, oof["support_plus_ensemble"], sample_weight=weights)
    without_spread = roc_auc_score(large, oof["support_only"], sample_weight=weights)
    boot.append(float(with_spread - without_spread))
ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
report = {
    "schema": "trust-calibration-group-oof-all-fit-is-train-only-v2",
    "source": CAL.relative_to(ROOT).as_posix(),
    "source_sha256": hashlib.sha256(CAL.read_bytes()).hexdigest(),
    "n_rows": len(d),
    "n_geometries": len(set(groups)),
    "high_error_base_rate": float(np.mean(large)),
    "cv": "GroupKFold(5) with no shared airfoil IDs",
    "target_definition": "within each fold: max target absolute residual normalized by training-fold P90; high risk threshold training-fold P90",
    "fitting": "per-fold train-only normalization/threshold and train-only input scaling; ridge alpha=10 fixed",
    "folds": folds,
    "models": res,
    "incremental_auroc": float(delta),
    "cluster_bootstrap_replicates": len(boot),
    "cluster_bootstrap_95pct_interval": ci,
    "limits": "Calibration-distribution CV diagnostic only. Not tested on pristine optimisation-selected/CFD examples. No operational trust threshold changed.",
}
target = OUT / "TRUST_INCREMENTAL_CV.json"
target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "n_rows": report["n_rows"],
            "models": res,
            "delta": delta,
            "cluster_bootstrap_95pct_interval": ci,
        },
        indent=2,
    )
)
