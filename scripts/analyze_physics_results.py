"""
analyze_physics_results.py

Analyze and visualize results from physics-aware hyperparameter tuning.
Creates comprehensive plots and tables comparing different models and loss functions.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def load_results(results_dir: str = "optuna_results/physics_aware"):
    """Load all result files from the results directory."""

    results = {}
    results_path = Path(results_dir)

    # Find all result JSON files
    for file in results_path.glob("*_results.json"):
        # Parse filename: ModelName_LossName_results.json
        parts = file.stem.split("_")
        if len(parts) >= 2:
            model_name = parts[0]
            loss_name = parts[1]

            with open(file, "r") as f:
                data = json.load(f)

            if model_name not in results:
                results[model_name] = {}

            results[model_name][loss_name] = data

    return results


def create_summary_dataframe(results):
    """Create a summary DataFrame from results."""

    rows = []

    for model_name, loss_dict in results.items():
        for loss_name, data in loss_dict.items():
            lofo_results = data.get("lofo_results", [])

            if len(lofo_results) == 0:
                continue

            # Calculate averages
            avg_physics = np.mean([r["physics_score"] for r in lofo_results])
            std_physics = np.std([r["physics_score"] for r in lofo_results])
            avg_mse = np.mean([r["mse"] for r in lofo_results])
            avg_cl_max = np.mean([r["cl_max_error_mean"] for r in lofo_results])
            avg_stall = np.mean([r["alpha_stall_error_mean"] for r in lofo_results])
            avg_zero_lift = np.mean(
                [r["alpha_zero_lift_error_mean"] for r in lofo_results]
            )
            avg_slope = np.mean(
                [r["lift_curve_slope_error_mean"] for r in lofo_results]
            )

            rows.append(
                {
                    "Model": model_name,
                    "Loss": loss_name,
                    "Physics Score": avg_physics,
                    "Physics Std": std_physics,
                    "MSE": avg_mse,
                    "ΔCL_max": avg_cl_max,
                    "Δα_stall (°)": avg_stall,
                    "Δα_L=0 (°)": avg_zero_lift,
                    "Δslope": avg_slope,
                }
            )

    df = pd.DataFrame(rows)
    return df


def plot_model_comparison(df, save_dir):
    """Create bar plots comparing models across metrics."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = [
        "Physics Score",
        "MSE",
        "ΔCL_max",
        "Δα_stall (°)",
        "Δα_L=0 (°)",
        "Δslope",
    ]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Create grouped bar plot
        models = df["Model"].unique()
        losses = df["Loss"].unique()

        x = np.arange(len(models))
        width = 0.2

        for i, loss in enumerate(losses):
            subset = df[df["Loss"] == loss]
            values = [
                (
                    subset[subset["Model"] == model][metric].values[0]
                    if len(subset[subset["Model"] == model]) > 0
                    else 0
                )
                for model in models
            ]

            ax.bar(x + i * width, values, width, label=loss, alpha=0.8)

        ax.set_xlabel("Model", fontweight="bold")
        ax.set_ylabel(metric, fontweight="bold")
        ax.set_title(f"{metric} by Model and Loss", fontweight="bold")
        ax.set_xticks(x + width * (len(losses) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend(title="Loss", loc="best")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches="tight"
    )
    logger.info("Saved: %s", os.path.join(save_dir, "model_comparison.png"))
    plt.close()


def plot_lofo_variance(results, save_dir):
    """Plot variance across LOFO folds for each model-loss combination."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ["physics_score", "cl_max_error_mean", "alpha_stall_error_mean", "mse"]
    titles = ["Physics Score", "CL_max Error", "Stall Angle Error (°)", "MSE"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        data_for_plot = []
        labels = []

        for model_name, loss_dict in results.items():
            for loss_name, data in loss_dict.items():
                lofo_results = data.get("lofo_results", [])

                if len(lofo_results) > 0:
                    values = [r[metric] for r in lofo_results]
                    data_for_plot.append(values)
                    labels.append(f"{model_name}\n{loss_name}")

        # Box plot
        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_plot)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_xlabel("Model - Loss Combination", fontweight="bold")
        ax.set_ylabel(title, fontweight="bold")
        ax.set_title(f"{title} Variance Across LOFO Folds", fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "lofo_variance.png"), dpi=300, bbox_inches="tight"
    )
    logger.info("Saved: %s", os.path.join(save_dir, "lofo_variance.png"))
    plt.close()


def plot_physics_vs_mse(df, save_dir):
    """Scatter plot showing relationship between physics score and MSE."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    models = df["Model"].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

    for model, color in zip(models, colors):
        subset = df[df["Model"] == model]

        for _, row in subset.iterrows():
            ax.scatter(
                row["MSE"],
                row["Physics Score"],
                c=[color],
                s=200,
                alpha=0.6,
                marker="o",
                edgecolors="black",
                linewidth=1.5,
            )
            ax.annotate(
                row["Loss"],
                (row["MSE"], row["Physics Score"]),
                fontsize=8,
                ha="center",
                va="bottom",
            )

    # Add legend
    for model, color in zip(models, colors):
        ax.scatter(
            [],
            [],
            c=[color],
            s=200,
            alpha=0.6,
            label=model,
            marker="o",
            edgecolors="black",
            linewidth=1.5,
        )

    ax.set_xlabel("MSE", fontsize=14, fontweight="bold")
    ax.set_ylabel("Physics Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Physics Score vs MSE: Model-Loss Combinations", fontsize=16, fontweight="bold"
    )
    ax.legend(title="Model", loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "physics_vs_mse.png"), dpi=300, bbox_inches="tight"
    )
    logger.info("Saved: %s", os.path.join(save_dir, "physics_vs_mse.png"))
    plt.close()


def print_summary_statistics(df):
    """Print summary statistics to console."""
    if df is None or df.empty:
        logger.info("No summary data available.")
        return

    logger.info("%s", "\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("%s", "=" * 80)

    # Best overall
    best_idx = df["Physics Score"].idxmin()
    best_row = df.loc[best_idx]

    logger.info("BEST MODEL:")
    logger.info("%s with %s loss", best_row["Model"], best_row["Loss"])
    logger.info("Physics Score: %.4f", best_row["Physics Score"])
    logger.info("MSE: %.6f", best_row["MSE"])
    logger.info("ΔCL_max: %.4f", best_row["ΔCL_max"])
    logger.info("Δα_stall: %.2f°", best_row["Δα_stall (°)"])
    logger.info("Δα_L=0: %.2f°", best_row["Δα_L=0 (°)"])


    # Best per model
    logger.info("%s", "\nBEST PER MODEL:")
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        best_idx = model_df["Physics Score"].idxmin()
        best_row = model_df.loc[best_idx]

        logger.info(
            "%s: %s (Physics=%.4f, MSE=%.6f)",
            model,
            best_row["Loss"],
            best_row["Physics Score"],
            best_row["MSE"],
        )

    # Best per loss
    logger.info("%s", "\nBEST PER LOSS FUNCTION:")
    for loss in df["Loss"].unique():
        loss_df = df[df["Loss"] == loss]
        best_idx = loss_df["Physics Score"].idxmin()
        best_row = loss_df.loc[best_idx]

        logger.info(
            "%s: %s (Physics=%.4f, MSE=%.6f)",
            loss,
            best_row["Model"],
            best_row["Physics Score"],
            best_row["MSE"],
        )

    # Rankings
    logger.info("%s", "\nTOP 5 OVERALL:")
    top5 = df.nsmallest(5, "Physics Score")
    for idx, row in top5.iterrows():
        logger.info(
            "%s + %s: Physics=%.4f, MSE=%.6f",
            row["Model"],
            row["Loss"],
            row["Physics Score"],
            row["MSE"],
        )

    logger.info("%s", "\n" + "=" * 80 + "\n")


def main():
    """Main analysis pipeline."""

    RESULTS_DIR = "optuna_results/physics_aware"
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

    # Create figures directory
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading results...")
    results = load_results(RESULTS_DIR)

    if not results:
        print(f"No results found in {RESULTS_DIR}")
        return

    print(f"Found results for {len(results)} models")

    # Create summary DataFrame
    print("Creating summary DataFrame...")
    df = create_summary_dataframe(results)

    # Print statistics
    print_summary_statistics(df)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_model_comparison(df, FIGURES_DIR)
    plot_lofo_variance(results, FIGURES_DIR)
    plot_physics_vs_mse(df, FIGURES_DIR)

    print(f"\nAnalysis complete! All outputs saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
