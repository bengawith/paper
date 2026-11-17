"""Produce the final standalone paper from terminal, source-bound numerical results.

No placeholders, favourable-only selection, missing-case imputation, or
replacement of historical raw evidence. This script is not run until all
required scientific receipts exist and their terminal status is checked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
REVIEW = ROOT / "reports/robust_v2/external_academic_review_20261014"
CONTROL = REVIEW / "CONTROLLED_MODEL"
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, precision=3):
    if value is None:
        return "not estimable"
    return f"{float(value):.{precision}f}"


def integer(value):
    return f"{int(value):,}"


def row(table, subset, target):
    return next(r for r in table if r["subset"] == subset and r["target"] == target)


def percentile(value):
    return number(100 * value, 1) + "%"


def rows_by_name(items):
    return {record["split"]: record for record in items}


def plot_setup():
    FIG.mkdir(exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)


def make_figures(dataset, metrics, risk, confirm, production, nominal, selected, finals, xfoil):
    plot_setup()
    pred = pd.read_parquet(LINEAGE / "locked_test/predictions.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, target in zip(axes, ["cl", "log_cd", "cm"], strict=True):
        rng = np.random.default_rng(20261021)
        subset = pred.loc[pred["mask_" + target].astype(bool)]
        if len(subset) > 2500:
            subset = subset.iloc[np.sort(rng.choice(len(subset), 2500, replace=False))]
        true = subset[target].to_numpy(float)
        guess = subset["prediction_" + target].to_numpy(float)
        axis.scatter(true, guess, s=2.5, alpha=0.25)
        lo, hi = min(true.min(), guess.min()), max(true.max(), guess.max())
        axis.plot([lo, hi], [lo, hi], ls="--", color="black", lw=0.8)
        axis.set(
            xlabel="Local XFOIL " + target,
            ylabel="Surrogate " + target,
            title=f"{target} | R²={number(row(metrics['metrics'], 'all', target)['r2'])}",
        )
    save(fig, "fig_surrogate_accuracy.png")
    groups = rows_by_name(dataset["split_counts"])
    labels = ["development", "calibration", "locked_test"]
    requested = [groups[name]["airfoils"] * 113 for name in labels]
    observed = [groups[name]["states"] for name in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    pos = np.arange(len(labels))
    ax.bar(pos, requested, label="Requested incidence–geometry states", color="0.78")
    ax.bar(pos, observed, label="Clean observed labels", color="0.36")
    ax.set_xticks(pos, labels=["Development", "Calibration", "Protected test"])
    ax.set(
        ylabel="Numerical state count",
        title="Exact-CST label coverage (solver failures not filled)",
    )
    for x, y in zip(pos, observed, strict=True):
        ax.text(x, y + 800, integer(y), ha="center", fontsize=9)
    ax.legend(loc="upper right")
    save(fig, "fig_perturbation_validity.png")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    names = ["Calibration OOF", "Sealed numerical confirmation"]
    support = [
        risk["models"]["support_only"]["auroc"],
        confirm["risk_models"]["support_only"]["auroc"],
    ]
    ensemble_only = [
        risk["models"]["ensemble_only"]["auroc"],
        confirm["risk_models"]["ensemble_only"]["auroc"],
    ]
    combined = [
        risk["models"]["support_plus_ensemble"]["auroc"],
        confirm["risk_models"]["support_plus_ensemble"]["auroc"],
    ]
    pos = np.arange(2)
    width = 0.23
    for off, values, label in [
        (-width, support, "Support only"),
        (0, ensemble_only, "Disagreement only"),
        (width, combined, "Support + ensemble"),
    ]:
        heights = [v if v is not None else 0 for v in values]
        bars = ax.bar(pos + off, heights, width, label=label)
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                number(v) if v is not None else "Not estimable",
                ha="center",
                fontsize=8,
            )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("High-error detection AUROC")
    ax.set_xticks(pos, names)
    ax.set_title("Residual-risk ranking | fixed labels and geometry-cluster separation")
    ax.legend()
    save(fig, "fig_surrogate_trust_stress.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    source = LINEAGE / "optimisation_production"
    for record in production["per_seed"]:
        case = read(source / record["file"])
        f = np.asarray(case["objectives"], float)
        g = np.asarray(case["constraint_values"], float)
        if not len(f):
            continue
        feasible = np.all(g <= 0, axis=1) & np.isfinite(f).all(axis=1)
        if feasible.any():
            ax.scatter(
                f[feasible, 0] * 1000,
                f[feasible, 1] * 1000,
                s=9,
                alpha=0.7,
                label=str(record["seed"]),
            )
    ax.set(
        xlabel="Mean penalised weighted C_D × 10³",
        ylabel="Empirical CVaR95 penalised C_D × 10³",
        title="Finite full-budget robust NSGA-II feasible fronts; 10 seeds",
    )
    ax.text(0.02, 0.02, "Not proof of a global Pareto optimum", transform=ax.transAxes, fontsize=8)
    if ax.collections:
        ax.legend(fontsize=6, ncol=2)
    save(fig, "fig_pareto_front.png")
    p = xfoil["paired_conditional_results"]
    chosen = [
        ("Nominal", "nominal_risk_knee"),
        ("1% stress", "stress_1pct_risk_knee"),
        ("0.5% A", "sensitivity_0p5pct_a_risk_knee"),
        ("0.5% B", "sensitivity_0p5pct_b_risk_knee"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = [p.get(key, {}).get("mean_reduction") for _, key in chosen]
    values = [100 * x if x is not None else 0 for x in bars]
    ax.bar([name for name, _ in chosen], values, color=["0.5"] * len(values))
    ax.axhline(0, color="black", lw=0.8)
    for i, (_, key) in enumerate(chosen):
        n = p.get(key, {}).get("pairs", 0)
        ax.text(i, values[i] + (1 if values[i] >= 0 else -1), f"n={n}", ha="center", fontsize=9)
    ax.set(
        ylabel="Paired mean drag reduction (%) | completed cases",
        title="Actual-coordinate local XFOIL: conditional paired comparisons",
    )
    save(fig, "fig_design_verification.png")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for record in production["per_seed"]:
        case = read(source / record["file"])
        history = case["history"]
        ax.plot(
            [h["generation"] for h in history],
            [h["feasible_count"] for h in history],
            linewidth=0.9,
            alpha=0.8,
        )
    ax.set(
        xlabel="Generation",
        ylabel="Feasible current-population members",
        title="All ten full-budget robust search trajectories",
    )
    ax.set_ylim(0, 130)
    save(fig, "fig_worstcase_polar.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = finals["rows"]
    amps = sorted({float(r["amplitude_fraction_chord"]) for r in data})
    for design, solid in [("reference", "-"), ("risk_knee", "--")]:
        found = [
            next(
                r
                for r in data
                if r["design"] == design and float(r["amplitude_fraction_chord"]) == a
            )
            for a in amps
        ]
        ax.plot(
            [100 * a for a in amps],
            [1000 * r["mean_penalised_weighted_cd"] for r in found],
            ls=solid,
            marker="o",
            label=f"{design}: penalised mean",
        )
        ax.plot(
            [100 * a for a in amps],
            [1000 * r["empirical_cvar95_penalised_weighted_cd"] for r in found],
            ls=solid,
            marker="s",
            label=f"{design}: empirical CVaR95",
        )
    ax.set(
        xlabel="Prescribed surface-normal displacement (% chord)",
        ylabel="Penalty-inclusive weighted C_D × 10³",
        title="Fresh fixed-design sensitivity | 2,048 common scenarios",
    )
    ax.legend()
    save(fig, "fig_manufacturing_risk.png")
    retention = CONTROL / "CONFIRMATION/FROZEN_THRESHOLD_RETENTION.csv"
    if retention.is_file():
        table = pd.read_csv(retention)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, part in table.groupby("model"):
            part = part.sort_values("actual_confirmation_retained_fraction")
            ax.plot(
                part["actual_confirmation_retained_fraction"] * 100,
                part["conditional_geometry_macro_scaled_error"],
                marker="o",
                label=name,
            )
        ax.set(
            xlabel="Observed confirmation states retained (%)",
            ylabel="Geometry-macro scaled error",
            title="Frozen calibration-defined retention on sealed confirmation",
        )
        ax.legend()
        save(fig, "fig_ml_selective_coverage.png")
    else:
        raise FileNotFoundError("Frozen confirmation retention outcomes unavailable")


def main():
    selected_path = LINEAGE / "selected_study/FROZEN_SELECTED_DESIGNS.json"
    finals_path = LINEAGE / "selected_study/FINAL_2048_SCENARIO_EVALUATION.json"
    xfoil_path = LINEAGE / "selected_study/SELECTED_XFOIL_SUMMARY.json"
    required = [
        LINEAGE / "DATA_MANIFEST.json",
        CONTROL / "REPAIRED_TEST_METRICS.json",
        CONTROL / "REPAIRED_BASELINES.json",
        CONTROL / "RISK_OOF/REPAIRED_RISK_OOF.json",
        CONTROL / "CONFIRMATION/CONFIRMATION_RESULTS.json",
        LINEAGE / "optimisation_production/PRODUCTION_SUMMARY.json",
        LINEAGE / "optimisation_production/PRODUCTION_PROTOCOL.json",
        LINEAGE / "nominal_control/NOMINAL_CONTROL_SUMMARY.json",
        selected_path,
        finals_path,
        xfoil_path,
    ]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError(
            "A required terminal controlled-study scientific receipt is missing: "
            + repr([str(p) for p in required if not p.is_file()])
        )
    (
        dataset,
        metrics,
        baseline,
        risk,
        confirm,
        production,
        protocol,
        nominal,
        selected,
        finals,
        xfoil,
    ) = map(read, required)
    if (
        production["status"] != "FULL_PROFILE_COMPLETED"
        or nominal["status"] != "FULL_NOMINAL_PROFILE_COMPLETED"
    ):
        raise ValueError("Full robust and matched nominal campaigns must reach terminal evidence")
    if (
        len(production["per_seed"]) != 10
        or len(nominal["per_seed"]) != 10
        or not selected["selection_complete_before_selected_solver"]
    ):
        raise ValueError("Missing preselection or a declared production seed")
    if xfoil["cases_requested"] != len(xfoil["cases"]):
        raise ValueError("Selected solver receipt accounting incomplete")
    if (
        confirm["requested_geometries"] != 200
        or confirm["claims"]["confirmation_used_to_fit_or_select_model"]
    ):
        raise ValueError("Prospective confirmation independence is not established")
    old = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    match = old.split("## References\n\n")
    if len(match) != 2:
        raise ValueError("Cannot reconcile inherited bibliography")
    original_refs = "## References\n\n" + match[1]
    b = {(r["model"], r["target"]): r for r in baseline["rows"]}
    parts = rows_by_name(dataset["split_counts"])
    values = {
        "requested_states": integer(dataset["requested_unique_states"]),
        "requested_geometries": integer(dataset["requested_geometries"]),
        "observed_states": integer(dataset["observed_states"]),
        "missing_states": integer(dataset["missing_requested_states"]),
        "geometries_observed": integer(dataset["geometries_with_observations"]),
        "development_airfoils": integer(parts["development"]["airfoils"]),
        "development_clusters": integer(parts["development"]["clusters"]),
        "development_states": integer(parts["development"]["states"]),
        "calibration_airfoils": integer(parts["calibration"]["airfoils"]),
        "calibration_clusters": integer(parts["calibration"]["clusters"]),
        "calibration_states": integer(parts["calibration"]["states"]),
        "test_airfoils": integer(parts["locked_test"]["airfoils"]),
        "test_clusters": integer(parts["locked_test"]["clusters"]),
        "test_states": integer(parts["locked_test"]["states"]),
        "total_observed_clusters": integer(sum(p["clusters"] for p in parts.values())),
        "full_grid_geometries": integer(dataset["complete_grid_geometries"]),
        "test_accepted": integer(metrics["accepted_states"]),
        "test_rejected": integer(metrics["rejected_states"]),
        "confirmation_shapes": integer(confirm["requested_geometries"]),
        "confirmation_observed": integer(confirm["observed_states"]),
        "confirmation_requested": integer(confirm["requested_states"]),
        "confirmation_accepted": integer(confirm["accepted_observed_states"]),
        "calib_auc_support": number(risk["models"]["support_only"]["auroc"]),
        "calib_auc_ensemble": number(risk["models"]["ensemble_only"]["auroc"]),
        "calib_auc_combined": number(risk["models"]["support_plus_ensemble"]["auroc"]),
        "calib_auc_delta": number(risk["incremental_auroc"]),
        "calib_interval": str(
            [number(x) for x in risk["fixed_oof_cluster_bootstrap_95pct_interval"]]
        ),
        "confirmation_interval": str(
            [number(x) for x in confirm["fixed_model_geometry_bootstrap_delta_interval"]]
        )
        if confirm["fixed_model_geometry_bootstrap_delta_interval"]
        else "not estimable",
        "confirm_auc_support": number(confirm["risk_models"]["support_only"]["auroc"]),
        "confirm_auc_ensemble": number(confirm["risk_models"]["ensemble_only"]["auroc"]),
        "confirm_ensemble_only_comparison": (
            "higher"
            if confirm["risk_models"]["ensemble_only"]["auroc"]
            > confirm["risk_models"]["support_plus_ensemble"]["auroc"]
            else "not higher"
        ),
        "confirm_auc_combined": number(confirm["risk_models"]["support_plus_ensemble"]["auroc"]),
        "confirmation_risk_delta": number(confirm["incremental_combined_minus_support_auroc"])
        if confirm["incremental_combined_minus_support_auroc"] is not None
        else "an unestimable amount",
        "coverage_cl": percentile(row(metrics["metrics"], "all", "cl")["state_interval_coverage"]),
        "coverage_log_cd": percentile(
            row(metrics["metrics"], "all", "log_cd")["state_interval_coverage"]
        ),
        "coverage_cm": percentile(row(metrics["metrics"], "all", "cm")["state_interval_coverage"]),
        "reference_objectives": str([number(x, 5) for x in protocol["reference_objectives"]]),
        "production_generations": integer(sum(x["generations"] for x in production["per_seed"])),
        "feasible_seeds": integer(sum(x["feasible_entries"] > 0 for x in production["per_seed"])),
        "nominal_feasible_seeds": integer(
            sum(x["feasible_entries"] > 0 for x in nominal["per_seed"])
        ),
        "feasible_final_entries": integer(
            sum(x["feasible_entries"] for x in production["per_seed"])
        ),
        "prospective_distance": number(
            production["reference_scaled_agreement"]["pairwise_objective_distance_mean"]
        ),
        "original_distance": number(
            production["original_pooled_range_diagnostic"]["pairwise_objective_distance_mean"]
        ),
        "robust_min_mean": number(production["reference_scaled_agreement"]["objective_min"][0], 5)
        if production["reference_scaled_agreement"]["objective_min"]
        else "not estimable",
        "nominal_min_drag": number(
            min(
                (
                    s["minimum_feasible_nominal_drag"]
                    for s in nominal["per_seed"]
                    if s["minimum_feasible_nominal_drag"] is not None
                ),
                default=None,
            ),
            5,
        ),
    }
    both = production["both_numerical_checks_met"]
    values["production_result"] = (
        "met both specified numerical agreement diagnostics"
        if both
        else "did not meet both specified numerical agreement diagnostics"
    )
    values["selected_result"] = (
        f"attempted {integer(xfoil['cases_requested'])} strictly defined local solver cases, with {integer(xfoil['usable_cases'])} usable outcomes and {integer(xfoil['cases_failed_or_not_admitted'])} failures or rejected cases"
    )
    for target, label in [("cl", "cl"), ("log_cd", "log_cd"), ("cm", "cm"), ("cd", "cd")]:
        r = row(metrics["metrics"], "all", target)
        values["r2_" + label] = number(r["r2"], 4)
        values["mae_" + label] = number(r["mae"], 5)
        values["macro_" + label] = number(r["geometry_macro_mae"], 5)
        if target != "cd":
            values["ridge_" + label] = number(b[("ridge", target)]["r2"], 4)
            values["hgb_" + label] = number(b[("hist_gradient_boosting", target)]["r2"], 4)
    for name, placeholder in [("reference", "reference"), ("risk_knee", "knee")]:
        r = next(
            x
            for x in finals["rows"]
            if x["design"] == name and abs(float(x["amplitude_fraction_chord"]) - 0.01) < 1e-12
        )
        values[placeholder + "_risk_mean_1pct"] = number(r["mean_penalised_weighted_cd"], 6)
        values[placeholder + "_risk_tail_1pct"] = number(
            r["empirical_cvar95_penalised_weighted_cd"], 6
        )
    pairmap = [
        ("nominal_risk_knee", "nominal_knee"),
        ("stress_1pct_risk_knee", "stress_knee"),
        ("sensitivity_0p5pct_a_risk_knee", "scramble_a"),
        ("sensitivity_0p5pct_b_risk_knee", "scramble_b"),
        ("nominal_nominal_control", "nominal_control"),
    ]
    for name, placeholder in pairmap:
        r = xfoil["paired_conditional_results"].get(name, {})
        values[placeholder + "_pairs"] = integer(r.get("pairs", 0))
        values[placeholder + "_reduction"] = (
            percentile(r["mean_reduction"])
            if r.get("mean_reduction") is not None
            else "not estimable"
        )
        values[placeholder + "_maxima"] = (
            f"{number(r['reference_max'], 5)} / {number(r['candidate_max'], 5)}"
            if r.get("pairs")
            else "not estimable"
        )
    nominal_risk_pairs = (
        xfoil["paired_conditional_results"].get("nominal_risk_knee", {}).get("pairs", 0)
    )
    nominal_control_pairs = (
        xfoil["paired_conditional_results"].get("nominal_nominal_control", {}).get("pairs", 0)
    )
    values["strict_nominal_interpretation"] = (
        "The fixed strict XFOIL nominal grid yielded no complete matched reference–candidate pair for at least one selected comparison, so a new nominal drag-reduction percentage cannot be estimated for that comparator; historical favourable nominal results must not be substituted."
        if nominal_risk_pairs == 0 or nominal_control_pairs == 0
        else "Both nominal matched comparisons satisfy the full predeclared solver grid; the percentages remain conditional numerical same-XFOIL comparisons rather than independent physical validation."
    )
    template = (HERE / "CONTROLLED_MANUSCRIPT_TEMPLATE.md").read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    unresolved = re.findall(r"\{[a-z][a-z0-9_]*\}", template)
    if unresolved:
        raise ValueError("Unresolved data-driven paper fields: " + repr(sorted(set(unresolved))))
    template += "\n\n" + original_refs
    if template.count("![Figure") != 7 or template.count("![Supplementary Figure") != 1:
        raise ValueError("Expected seven numbered manuscript figures and one supplementary")
    # Safeguard the previous paper for traceability before replacing current text.
    historical = HERE / "PRE_CONTROLLED_MANUSCRIPT.md"
    if historical.exists():
        if historical.read_text(encoding="utf-8") != old:
            raise ValueError("The baseline manuscript changed after review backup")
    else:
        historical.write_text(old, encoding="utf-8")
    PAPER.joinpath("manuscript.md").write_text(template + "\n", encoding="utf-8")
    make_figures(dataset, metrics, risk, confirm, production, nominal, selected, finals, xfoil)
    print(
        "CONTROLED_MANUSCRIPT_NUMERICALLY_RECONCILED",
        len(values),
        "machine-bound substitutions",
        flush=True,
    )


if __name__ == "__main__":
    main()
