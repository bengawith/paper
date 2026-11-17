"""Evaluate frozen multi-output ensemble's selective-error behaviour without fitting on test.

Normalization of support/disagreement is fixed from CALIBRATION partition.
Held-out results are descriptive; they do not tune model, thresholds, or optimizer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
LIN = ROOT / "results/robust_v2/lineages/robust-v2-current-20260908"
CAL = LIN / "calibration/calibration_predictions.parquet"
TEST = LIN / "locked_test/predictions.parquet"
OUT = ROOT / "reports/robust_v2/external_academic_review_20261014/ML_SELECTIVE_CURRENT"
OUT.mkdir(parents=True, exist_ok=True)
COLS = [
    ("cl", "prediction_cl", "ensemble_std_cl"),
    ("log_cd", "prediction_log_cd", "ensemble_std_log_cd"),
    ("cm", "prediction_cm", "ensemble_std_cm"),
]
COVERAGES = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])


def finite_group_macro_mae(frame: pd.DataFrame, truth: str, prediction: str) -> float:
    f = frame.loc[
        np.isfinite(frame[truth].to_numpy(float)) & np.isfinite(frame[prediction].to_numpy(float))
    ].copy()
    if f.empty:
        return float("nan")
    f["abs_err"] = np.abs(f[truth] - f[prediction])
    return float(f.groupby("airfoil_id")["abs_err"].mean().mean())


def main() -> None:
    cal, test = pd.read_parquet(CAL), pd.read_parquet(TEST)
    scales = {}
    for _, _, sd in COLS:
        values = cal[sd].to_numpy(float)
        values = values[np.isfinite(values)]
        scales[sd] = float(max(np.quantile(values, 0.95), 1e-12))
    supp = cal["support_distance"].to_numpy(float)
    scales["support_distance"] = float(max(np.quantile(supp[np.isfinite(supp)], 0.95), 1e-12))
    components = [test["support_distance"].to_numpy(float) / scales["support_distance"]]
    for _, _, sd in COLS:
        components.append(test[sd].to_numpy(float) / scales[sd])
    risks = np.nanmax(np.vstack(components), axis=0)
    risks[~np.isfinite(risks)] = np.inf
    ordering = np.argsort(risks, kind="stable")
    table = []
    results = {}
    for truth, pred, _ in COLS:
        true_errors = np.abs(test[truth].to_numpy(float) - test[pred].to_numpy(float))
        correlation = float(spearmanr(risks, true_errors, nan_policy="omit").statistic)
        curve = []
        for coverage in COVERAGES:
            count = max(1, int(np.floor(len(test) * coverage)))
            subset = test.iloc[ordering[:count]]
            err = finite_group_macro_mae(subset, truth, pred)
            curve.append(
                {
                    "target": truth,
                    "retained_fraction": float(coverage),
                    "n_states": int(count),
                    "n_airfoils": int(subset["airfoil_id"].nunique()),
                    "group_macro_mae": err,
                }
            )
            table.append(curve[-1])
        trusted = test.loc[test["trusted_domain"].astype(bool)]
        rejected = test.loc[~test["trusted_domain"].astype(bool)]
        results[truth] = {
            "error_risk_spearman": correlation,
            "all_macro_mae": finite_group_macro_mae(test, truth, pred),
            "existing_trusted_macro_mae": finite_group_macro_mae(trusted, truth, pred),
            "existing_rejected_macro_mae": finite_group_macro_mae(rejected, truth, pred),
            "curve": curve,
        }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (truth, _, _) in zip(axes, COLS, strict=True):
        values = results[truth]
        ax.plot(
            [r["retained_fraction"] * 100 for r in values["curve"]],
            [r["group_macro_mae"] for r in values["curve"]],
            marker="o",
            lw=1.8,
        )
        ax.axhline(values["all_macro_mae"], linestyle="--", color="0.5", label="Full held-out set")
        ax.set(
            xlabel="Retained states (%)",
            ylabel="Macro MAE (per airfoil)",
            title=f"({'abc'[list(results).index(truth)]}) {truth} | Spearman={values['error_risk_spearman']:.2f}",
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        "Historical selective prediction | retrospective, no test-set fitting", fontsize=12
    )
    fig.tight_layout()
    figure = OUT / "fig_ml_selective_coverage.png"
    fig.savefig(figure, dpi=300)
    plt.close(fig)
    pd.DataFrame(table).to_csv(OUT / "selective_coverage.csv", index=False)
    frozen_rule = json.loads(
        (LIN / "calibration/trust_calibration.json").read_text(encoding="utf-8")
    )
    support_max = float(
        frozen_rule["support_distance_quantiles"]["trusted_threshold_from_residual_tolerance"]
    )
    failed_support = test["support_distance"].to_numpy(float) > support_max
    spread_fails = {
        t: int(
            (
                test[f"ensemble_std_{t}"].to_numpy(float)
                > float(
                    frozen_rule["ensemble_disagreement_quantiles"][t][
                        "trusted_threshold_from_residual_tolerance"
                    ]
                )
            ).sum()
        )
        for t, _, _ in COLS
    }
    failed_spread_any = np.logical_or.reduce(
        [
            test[f"ensemble_std_{t}"].to_numpy(float)
            > float(
                frozen_rule["ensemble_disagreement_quantiles"][t][
                    "trusted_threshold_from_residual_tolerance"
                ]
            )
            for t, _, _ in COLS
        ]
    )
    component_audit = {
        "n_test_states": int(len(test)),
        "untrusted_states_total": int((~test["trusted_domain"].astype(bool)).sum()),
        "failed_support_distance": int(failed_support.sum()),
        "failed_disagreement_any": int(failed_spread_any.sum()),
        "failed_disagreement_without_support_failure": int(
            (failed_spread_any & ~failed_support).sum()
        ),
        "failed_per_output_spread": spread_fails,
        "interpretation": "Frozen 2026-09 trust selection is effectively support-distance-driven on this held-out test; do not claim ensemble spread materially rejects states.",
    }
    report = {
        "schema": "frozen-ml-selective-audit-v1",
        "frozen_rule_component_audit": component_audit,
        "purpose": "Descriptive held-out test diagnostic; NO model selection or trust-threshold tuning",
        "source_test_sha256": hashlib.sha256(TEST.read_bytes()).hexdigest(),
        "source_calibration_sha256": hashlib.sha256(CAL.read_bytes()).hexdigest(),
        "n_test_states": len(test),
        "n_test_airfoils": int(test["airfoil_id"].nunique()),
        "fixed_risk_rule": "maximum of support distance and 3 ensemble disagreements, each divided by 95th calibration quantile",
        "calibration_scaling": scales,
        "targets": results,
    }
    (OUT / "ml_selective_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: {x: v for x, v in val.items() if x != "curve"} for k, val in results.items()},
            indent=2,
        )
    )
    print("Wrote", figure)


if __name__ == "__main__":
    main()
