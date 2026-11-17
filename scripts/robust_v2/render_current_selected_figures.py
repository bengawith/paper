"""Reproduce the two selected-design manuscript figures from frozen evidence.

This script reads existing numerical receipts. It performs no model inference,
optimisation, XFOIL calculation, or figure-value fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study"
FIGURES = ROOT / "paper/figures"

DESIGNS = (
    ("reference", "Reference", "#4c72b0"),
    ("risk_knee", "Risk-knee", "#228b69"),
    ("nominal_control", "Nominal control", "#d08734"),
)
AMPLITUDES = (0.001, 0.0025, 0.005, 0.01)


def load_json(name: str) -> dict:
    return json.loads((STUDY / name).read_text(encoding="utf-8"))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "savefig.dpi": 250,
        }
    )


def fresh_sensitivity() -> None:
    record = load_json("FINAL_2048_SCENARIO_EVALUATION.json")
    assert record["scenario_count"] == 2048
    rows = {
        (str(row["design"]), float(row["amplitude_fraction_chord"])): row
        for row in record["rows"]
        if row["design"] in {name for name, _, _ in DESIGNS}
    }
    assert len(rows) == 12
    x = np.array(AMPLITUDES) * 100
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for name, label, color in DESIGNS:
        rr = [rows[(name, amplitude)] for amplitude in AMPLITUDES]
        means = [row["mean_penalised_weighted_cd"] * 1000 for row in rr]
        tails = [row["empirical_cvar95_penalised_weighted_cd"] * 1000 for row in rr]
        ax.plot(x, means, "o-", color=color, lw=2, markersize=5,
                label=f"{label}: mean")
        ax.plot(x, tails, "s--", color=color, lw=1.8, markersize=4.5,
                label=f"{label}: CVaR95")
    ax.set_xticks(x, ["0.10", "0.25", "0.50", "1.00"])
    ax.set_xlabel("Prescribed surface-normal displacement (% chord)")
    ax.set_ylabel("Penalty-inclusive weighted-drag loss (×10⁻³)")
    ax.set_title("Fresh 2,048-scenario numerical sensitivity of fixed designs", pad=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, fontsize=9)
    ax.set_ylim(bottom=5.5)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.28)
    fig.savefig(FIGURES / "fig_manufacturing_risk.png")
    plt.close(fig)


def strict_matched_polars() -> None:
    record = load_json("SELECTED_XFOIL_SUMMARY.json")
    assert record["cases_requested"] == 819
    assert record["usable_cases"] == 290
    paired = record["paired_conditional_results"]
    campaigns = (
        ("nominal", "Nominal", 1),
        ("stress_1pct", "1% stress", 16),
        ("sensitivity_0p5pct_a", "0.5% A", 128),
        ("sensitivity_0p5pct_b", "0.5% B", 128),
    )
    designs = (
        ("risk_knee", "Risk-knee", "#228b69"),
        ("nominal_control", "Nominal control", "#d08734"),
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    centres = np.arange(len(campaigns))
    width = 0.34
    for design_index, (name, _label, color) in enumerate(designs):
        values = [paired[f"{campaign}_{name}"] for campaign, _, _ in campaigns]
        positions = centres + (design_index - 0.5) * width
        for position, row, (_, _, attempted) in zip(positions, values, campaigns, strict=True):
            n = int(row["pairs"])
            if n:
                reduction = float(row["mean_reduction"]) * 100
                ax.bar(position, reduction, width=width * 0.92, color=color)
                ax.text(position, reduction + 0.5, f"{reduction:.1f}%\nn={n}/{attempted}",
                        ha="center", va="bottom", fontsize=8.5)
            else:
                ax.text(position, 0.55, f"n=0/{attempted}\nN/A",
                        ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(centres, [label for _, label, _ in campaigns])
    ax.set_ylabel("Reduction in paired mean weighted drag (%)")
    ax.set_title("Strict XFOIL comparisons on complete matched polars", pad=15)
    ax.set_ylim(0, 25)
    ax.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in designs],
              loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(FIGURES / "fig_design_verification.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_style()
    fresh_sensitivity()
    strict_matched_polars()
    print("Rebuilt current Figures 5 and 7 directly from frozen selected-study receipts")


if __name__ == "__main__":
    main()
