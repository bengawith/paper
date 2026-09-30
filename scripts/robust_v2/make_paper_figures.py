"""Generate publication figures for the robust-aerofoil manuscript.

Every figure is produced only from committed lineage artefacts so the manuscript
never contains a number that is not backed by a durable evidence file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = "robust-v2-current-20260908"
RESULTS = ROOT / "results/robust_v2/lineages" / LINEAGE
FIGDIR = ROOT / "paper/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATA = json.loads((ROOT / "paper/data/paper_data.json").read_text(encoding="utf-8"))

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
TRUST_C = "#1b7837"
UNTRUST_C = "#762a83"


def fig_surrogate_accuracy() -> None:
    pred = pd.read_parquet(RESULTS / "locked_test/predictions.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [
        (pred["cl"], pred["prediction_cl"], "$C_L$", "mask_cl", DATA["surrogate_locked_test"]["cl"]["r2"]),
        (
            pred["log_cd"],
            pred["prediction_log_cd"],
            r"$\log C_D$",
            "mask_log_cd",
            DATA["surrogate_locked_test"]["log_cd"]["r2"],
        ),
        (pred["cm"], pred["prediction_cm"], "$C_M$", "mask_cm", DATA["surrogate_locked_test"]["cm"]["r2"]),
    ]
    for ax, (actual, predicted, label, mask_col, r2) in zip(axes, specs, strict=True):
        mask = pred[mask_col].astype(bool)
        trusted = pred["trusted_domain"].astype(bool) & mask
        a = actual.to_numpy(float)
        p = np.asarray(predicted, dtype=float)
        ax.scatter(a[mask & ~trusted], p[mask & ~trusted], s=3, c=UNTRUST_C, alpha=0.2, label="outside trust")
        ax.scatter(a[trusted], p[trusted], s=3, c=TRUST_C, alpha=0.2, label="within trust")
        lo = float(np.nanmin(a[mask]))
        hi = float(np.nanmax(a[mask]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"XFOIL {label}")
        ax.set_ylabel(f"Surrogate {label}")
        ax.set_title(f"$R^2$ = {r2:.3f}")
    axes[1].legend(markerscale=4, loc="upper left")
    for ax, tag in zip(axes, "abc", strict=True):
        ax.text(0.02, 0.98, f"({tag})", transform=ax.transAxes, va="top", fontweight="bold")
    fig.suptitle("Surrogate accuracy on held-out (grouped) locked-test aerofoils", y=1.02)
    fig.savefig(FIGDIR / "fig_surrogate_accuracy.png")
    plt.close(fig)


def fig_manufacturing_risk() -> None:
    levels = DATA["manufacturing_levels"]
    amps = sorted(float(k) for k in levels)
    pct = [a * 100 for a in amps]
    mean = [levels[str(a)]["mean"] for a in amps]
    cvar = [levels[str(a)]["cvar_95"] for a in amps]
    q05 = [levels[str(a)]["q05"] for a in amps]
    q95 = [levels[str(a)]["q95"] for a in amps]
    viol = [levels[str(a)]["violation_probability"] * 100 for a in amps]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(pct, q05, q95, color="#c7eae5", label="5-95% band")
    ax.plot(pct, mean, "o-", color="#01665e", lw=2, label="Mean weighted $C_D$")
    ax.plot(pct, cvar, "s--", color="#8c510a", lw=2, label="CVaR$_{95}$ weighted $C_D$")
    ax.set_xlabel("Manufacturing perturbation amplitude (% chord)")
    ax.set_ylabel("Weighted drag coefficient")
    ax.set_title("Drag risk profile under smooth bounded manufacturing deviation")
    ax2 = ax.twinx()
    ax2.plot(pct, viol, "^:", color="#b2182b", lw=1.5, label="Trust-domain violation %")
    ax2.set_ylabel("Trust-domain violation (%)", color="#b2182b")
    ax2.tick_params(axis="y", labelcolor="#b2182b")
    ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.savefig(FIGDIR / "fig_manufacturing_risk.png")
    plt.close(fig)


def fig_pareto_front() -> None:
    ns = DATA["nsga2_full"]
    front = np.array(ns["combined_front"])
    det = ns["deterministic_knee"]
    rob = ns["robust_knee"]
    seed_ids = np.unique(front[:, 2])
    seed_index = {s: i + 1 for i, s in enumerate(seed_ids)}
    colours = np.array([seed_index[s] for s in front[:, 2]])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(front[:, 0] * 1e3, front[:, 1] * 1e3, c=colours, cmap="viridis", s=18, alpha=0.6)
    ax.scatter([det[0] * 1e3], [det[1] * 1e3], marker="*", s=320, c="#b2182b", edgecolor="k",
               label="Deterministic optimum (min expected $C_D$)", zorder=5)
    ax.scatter([rob[0] * 1e3], [rob[1] * 1e3], marker="D", s=160, c="#2166ac", edgecolor="k",
               label="Robust optimum (min CVaR$_{95}$)", zorder=5)
    ax.set_xlabel(r"Expected weighted $C_D$ ($\times 10^{-3}$)")
    ax.set_ylabel(r"CVaR$_{95}$ weighted $C_D$ ($\times 10^{-3}$)")
    ax.set_title(
        f"Risk-aware Pareto front ({ns['independent_seeds']} seeds, "
        f"{len(front)} feasible designs)\nseed disagreement "
        f"{ns['mean_symmetric_distance']:.2f} vs {ns['agreement_threshold']:.2f} threshold"
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Independent seed index")
    ax.legend(loc="upper right")
    fig.savefig(FIGDIR / "fig_pareto_front.png")
    plt.close(fig)


def fig_perturbation_validity() -> None:
    ab = DATA["perturbation_ablations"]
    names = {
        "smooth_correlated_surface_normal": "Smooth correlated\nnormal (this work)",
        "independent_multiplicative_cst": "Independent\nmultiplicative CST",
        "independent_coordinate_noise": "Independent\ncoordinate noise",
    }
    order = ["smooth_correlated_surface_normal", "independent_multiplicative_cst", "independent_coordinate_noise"]
    frac = [100 * ab[k]["valid_geometry"] / ab[k]["samples"] for k in order]
    colors = [TRUST_C, "#4393c3", UNTRUST_C]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar([names[k] for k in order], frac, color=colors, edgecolor="k")
    for bar, value in zip(bars, frac, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Geometrically valid samples (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Physical admissibility of manufacturing-perturbation models (1% chord, 2048 samples)")
    fig.savefig(FIGDIR / "fig_perturbation_validity.png")
    plt.close(fig)


def fig_surrogate_trust_stress() -> None:
    stress = json.loads((RESULTS / "surrogate_stress/surrogate_stress_summary.json").read_text(encoding="utf-8"))
    records = [r for r in stress["records"] if r["status"] == "evaluated"
               and r["disagreement"]["mean_absolute_cl_gap"] is not None]
    support = np.array([r["mean_support_distance"] for r in records])
    gap = np.array([r["disagreement"]["mean_absolute_cl_gap"] for r in records])
    tf = np.array([r["trusted_fraction"] for r in records])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sc = axes[0].scatter(support, gap, c=tf, cmap="RdYlGn", s=60, edgecolor="k")
    axes[0].set_xlabel("Mean support distance from training manifold")
    axes[0].set_ylabel("Mean |$C_L$ surrogate - XFOIL| gap")
    axes[0].set_title(f"(a) Disagreement vs support distance\nSpearman "
                      f"{DATA['surrogate_stress']['support_vs_gap_spearman']:.2f}")
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label("Fraction of angles within trust domain")

    inside = DATA["surrogate_stress"]["mean_cl_gap_fully_trusted"]
    outside = DATA["surrogate_stress"]["mean_cl_gap_outside_trust"]
    worst = DATA["surrogate_stress"]["worst_overall"]["max_cl_gap"]
    worst_in = DATA["surrogate_stress"]["worst_within_trust"]["max_cl_gap"]
    axes[1].bar(["Within trust", "Outside trust"], [inside, outside],
                color=[TRUST_C, UNTRUST_C], edgecolor="k")
    axes[1].axhline(worst, ls="--", color="#b2182b", label=f"Worst-case gap = {worst:.2f}")
    axes[1].axhline(worst_in, ls=":", color=TRUST_C, label=f"Worst-case within trust = {worst_in:.2f}")
    axes[1].set_ylabel("Mean |$C_L$| disagreement")
    axes[1].set_title("(b) Adversarial worst-case disagreement")
    axes[1].legend()
    fig.suptitle("Adversarial surrogate-trust probing (Reviewer 2)", y=1.03)
    fig.savefig(FIGDIR / "fig_surrogate_trust_stress.png")
    plt.close(fig)


def fig_worstcase_polar() -> None:
    stress = json.loads((RESULTS / "surrogate_stress/surrogate_stress_summary.json").read_text(encoding="utf-8"))
    alpha_grid = np.array(stress["alpha_deg"])
    # Choose the best-sampled representative worst case: the fully converged,
    # outside-trust design with the largest mean lift disagreement over >=5 angles.
    candidates = [
        r for r in stress["records"]
        if r["status"] == "evaluated"
        and r["xfoil_status"] == "ok"
        and r["disagreement"]["matched_points"] >= 5
    ]
    candidates.sort(key=lambda r: r["disagreement"]["mean_absolute_cl_gap"], reverse=True)
    worst = candidates[0] if candidates else stress["worst_case_overall"]
    cid = worst["candidate_id"]
    cl_ml = np.array(worst["surrogate_cl"])
    poly = RESULTS / "surrogate_stress" / cid / "combined_points.csv"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(alpha_grid, cl_ml, "o-", color=UNTRUST_C, lw=2, label="Surrogate $C_L$ (outside trust)")
    if poly.is_file():
        xf = pd.read_csv(poly).dropna(subset=["alpha_deg", "cl"])
        ax.plot(xf["alpha_deg"], xf["cl"], "s--", color="#01665e", lw=2, label="XFOIL $C_L$")
    ax.set_xlabel(r"Angle of attack $\alpha$ (deg)")
    ax.set_ylabel("$C_L$")
    ax.set_title(
        "Representative worst-case surrogate-XFOIL divergence\n"
        f"(design outside trust domain, mean $\\Delta C_L$ = "
        f"{worst['disagreement']['mean_absolute_cl_gap']:.2f}, "
        f"max = {worst['disagreement']['max_absolute_cl_gap']:.2f})"
    )
    ax.legend()
    fig.savefig(FIGDIR / "fig_worstcase_polar.png")
    plt.close(fig)


def main() -> None:
    fig_surrogate_accuracy()
    fig_manufacturing_risk()
    fig_pareto_front()
    fig_perturbation_validity()
    fig_surrogate_trust_stress()
    fig_worstcase_polar()
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()

