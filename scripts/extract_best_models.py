"""
extract_best_models.py

Extract best hyperparameters from completed Optuna physics-aware tuning results
and use them to instantiate actual models for detailed analysis.

This script:
1. Loads comprehensive results from physics-aware tuning
2. Identifies best model-loss combinations
3. Extracts exact hyperparameters
4. Saves them in a format ready for gather_model_details.py
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

import logging

# Configure basic logging for script usage
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_comprehensive_results(
    results_dir: str = "optuna_results/physics_aware",
) -> Dict:
    """
    Load the comprehensive results file from physics-aware tuning.
    """
    results_path = Path(results_dir)

    # Look for comprehensive results file
    comprehensive_files = list(results_path.glob("comprehensive_results_*.json"))

    if not comprehensive_files:
        logger.error("No comprehensive results found in %s", results_dir)
        return None

    # Use most recent file
    latest_file = max(comprehensive_files, key=os.path.getctime)

    logger.info("Loading results from: %s", latest_file)

    with open(latest_file, "r") as f:
        results = json.load(f)

    return results


def extract_best_hyperparameters(results: Dict) -> Dict:
    """
    Extract best hyperparameters for each model type.

    Returns a dictionary with the best configuration for each model,
    selected based on average LOFO physics score.
    """
    best_configs = {}

    for model_name, loss_dict in results.items():
        logger.info("Analyzing %s", model_name)

        best_loss = None
        best_score = float("inf")
        best_params = None

        for loss_name, data in loss_dict.items():
            lofo_results = data.get("lofo_results", [])

            if len(lofo_results) == 0:
                continue

            # Calculate average physics score across LOFO folds
            avg_physics_score = np.mean([r["physics_score"] for r in lofo_results])

            logger.info("%s: Physics Score = %.4f", loss_name, avg_physics_score)

            if avg_physics_score < best_score:
                best_score = avg_physics_score
                best_loss = loss_name
                best_params = data.get("best_params", {})

        if best_params:
            best_configs[model_name] = {
                "best_loss": best_loss,
                "best_physics_score": best_score,
                "hyperparameters": best_params,
                "all_losses": {},
            }

            # Store all loss function results for comparison
            for loss_name, data in loss_dict.items():
                lofo_results = data.get("lofo_results", [])
                if len(lofo_results) > 0:
                    best_configs[model_name]["all_losses"][loss_name] = {
                        "physics_score_mean": float(
                            np.mean([r["physics_score"] for r in lofo_results])
                        ),
                        "physics_score_std": float(
                            np.std([r["physics_score"] for r in lofo_results])
                        ),
                        "mse_mean": float(np.mean([r["mse"] for r in lofo_results])),
                        "mse_std": float(np.std([r["mse"] for r in lofo_results])),
                        "cl_max_error_mean": float(
                            np.mean([r["cl_max_error_mean"] for r in lofo_results])
                        ),
                        "alpha_stall_error_mean": float(
                            np.mean([r["alpha_stall_error_mean"] for r in lofo_results])
                        ),
                    }

            logger.info("→ Best: %s (Physics Score = %.4f)", best_loss, best_score)

    return best_configs


def create_model_config_for_analysis(
    best_configs: Dict, input_size: int = 12, output_size: int = 48
) -> Dict:
    """
    Create a model configuration dictionary compatible with gather_model_details.py
    """
    model_configs = {}

    for model_name, config in best_configs.items():
        hyperparams = config["hyperparameters"]

        if model_name == "MLP":
            model_configs[model_name] = {
                "model_name": "MLP",
                "model_class": "MLPModel",
                "best_loss_function": config["best_loss"],
                "physics_score": config["best_physics_score"],
                "hyperparameters": {
                    "input_size": input_size,
                    "output_size": output_size,
                    "hidden_size": hyperparams.get("hidden_size"),
                    "num_layers": hyperparams.get("num_layers"),
                    "width_factor": hyperparams.get("width_factor"),
                    "dropout_rate": hyperparams.get("dropout_rate"),
                    "learning_rate": hyperparams.get("learning_rate"),
                    "weight_decay": hyperparams.get("weight_decay"),
                    "num_epochs": hyperparams.get("num_epochs"),
                },
            }

        elif model_name in ["GRU", "LSTM"]:
            model_class = f"{model_name}Model"
            model_configs[model_name] = {
                "model_name": model_name,
                "model_class": model_class,
                "best_loss_function": config["best_loss"],
                "physics_score": config["best_physics_score"],
                "hyperparameters": {
                    "input_size": input_size,
                    "output_size": output_size,
                    "hidden_size": hyperparams.get("hidden_size"),
                    "num_layers": hyperparams.get("num_layers"),
                    "bidirectional": hyperparams.get("bidirectional"),
                    "dropout_rate": hyperparams.get("dropout_rate"),
                    "learning_rate": hyperparams.get("learning_rate"),
                    "weight_decay": hyperparams.get("weight_decay"),
                    "num_epochs": hyperparams.get("num_epochs"),
                },
            }

        elif model_name == "CNN":
            model_configs[model_name] = {
                "model_name": "CNN",
                "model_class": "CNNModel",
                "best_loss_function": config["best_loss"],
                "physics_score": config["best_physics_score"],
                "hyperparameters": {
                    "input_size": input_size,
                    "output_size": output_size,
                    "num_filters": hyperparams.get("num_filters", [64, 128, 256]),
                    "kernel_sizes": hyperparams.get("kernel_sizes", [5, 5, 3]),
                    "dropout_rate": hyperparams.get("dropout_rate"),
                    "learning_rate": hyperparams.get("learning_rate"),
                    "weight_decay": hyperparams.get("weight_decay"),
                    "num_epochs": hyperparams.get("num_epochs"),
                },
            }

    return model_configs


def generate_comparison_table(best_configs: Dict) -> pd.DataFrame:
    """
    Generate a comparison table across all models and loss functions.
    """
    rows = []

    for model_name, config in best_configs.items():
        for loss_name, metrics in config["all_losses"].items():
            rows.append(
                {
                    "Model": model_name,
                    "Loss": loss_name,
                    "Physics Score": metrics["physics_score_mean"],
                    "Physics Std": metrics["physics_score_std"],
                    "MSE": metrics["mse_mean"],
                    "MSE Std": metrics["mse_std"],
                    "ΔCL_max": metrics["cl_max_error_mean"],
                    "Δα_stall (°)": metrics["alpha_stall_error_mean"],
                    "Is Best": "✓" if loss_name == config["best_loss"] else "",
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["Model", "Physics Score"])

    return df


def generate_hyperparameter_table(best_configs: Dict) -> pd.DataFrame:
    """
    Generate a table showing best hyperparameters for each model.
    """
    rows = []

    for model_name, config in best_configs.items():
        hyperparams = config["hyperparameters"]

        row = {
            "Model": model_name,
            "Loss Function": config["best_loss"],
            "Physics Score": config["best_physics_score"],
            "Hidden Size": hyperparams.get("hidden_size"),
            "Num Layers": hyperparams.get("num_layers"),
            "Dropout": hyperparams.get("dropout_rate"),
            "Learning Rate": hyperparams.get("learning_rate"),
            "Weight Decay": hyperparams.get("weight_decay"),
        }

        # Add model-specific parameters
        if model_name == "MLP":
            row["Width Factor"] = hyperparams.get("width_factor")
        elif model_name in ["GRU", "LSTM"]:
            row["Bidirectional"] = hyperparams.get("bidirectional")
        elif model_name == "CNN":
            row["Num Filters"] = str(hyperparams.get("num_filters"))
            row["Kernel Sizes"] = str(hyperparams.get("kernel_sizes"))

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    """
    Main extraction pipeline.
    """
    logger.info("%s", "=" * 80)
    logger.info("EXTRACTING BEST MODELS FROM OPTUNA RESULTS")
    logger.info("%s", "=" * 80)

    RESULTS_DIR = "optuna_results/physics_aware"
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "best_models")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load comprehensive results
    logger.info("[1/5] Loading comprehensive results...")
    results = load_comprehensive_results(RESULTS_DIR)

    if not results:
        return

    logger.info("Found results for %d model types", len(results))

    # 2. Extract best hyperparameters
    logger.info("[2/5] Extracting best hyperparameters...")
    best_configs = extract_best_hyperparameters(results)

    # 3. Create model configuration
    logger.info("[3/5] Creating model configurations...")
    model_configs = create_model_config_for_analysis(best_configs)

    # Save for use by gather_model_details.py
    config_file = os.path.join(OUTPUT_DIR, "best_model_configs.json")
    with open(config_file, "w") as f:
        json.dump(model_configs, f, indent=2)

    logger.info("Saved model configs to: %s", config_file)

    # 4. Generate comparison tables
    logger.info("[4/5] Generating comparison tables...")

    # All models and losses
    comparison_df = generate_comparison_table(best_configs)
    comparison_file = os.path.join(OUTPUT_DIR, "all_model_loss_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False, float_format="%.6f")
    logger.info("Saved comparison table to: %s", comparison_file)

    # Best hyperparameters
    hyperparams_df = generate_hyperparameter_table(best_configs)
    hyperparams_file = os.path.join(OUTPUT_DIR, "best_hyperparameters.csv")
    hyperparams_df.to_csv(hyperparams_file, index=False, float_format="%.6f")
    logger.info("Saved hyperparameters table to: %s", hyperparams_file)

    # 5. Print summary
    logger.info("[5/5] Summary of Best Models")
    logger.info("%s", "\n" + "=" * 80)
    logger.info("BEST MODEL FOR EACH ARCHITECTURE")
    logger.info("%s", "=" * 80)

    for model_name, config in best_configs.items():
        logger.info("%s", f"\n{model_name}:")
        logger.info("Best Loss Function: %s", config["best_loss"])
        logger.info("Physics Score: %.4f", config["best_physics_score"])
        logger.info("Hyperparameters:")
        for key, value in config["hyperparameters"].items():
            if isinstance(value, float):
                logger.info("    %s: %.6f", key, value)
            else:
                logger.info("    %s: %s", key, value)

    # Find overall best
    overall_best = min(best_configs.items(), key=lambda x: x[1]["best_physics_score"])

    print("\n" + "=" * 80)
    print("OVERALL BEST MODEL")
    print("=" * 80)
    logger.info("%s with %s loss", overall_best[0], overall_best[1]["best_loss"])
    logger.info("Physics Score: %.4f", overall_best[1]["best_physics_score"])

    # Print comparison table
    logger.info("%s", "\n" + "=" * 80)
    logger.info("COMPARISON TABLE (Top 10)")
    logger.info("%s", "=" * 80)
    logger.info("%s", "\n" + comparison_df.head(10).to_string(index=False))

    # Print hyperparameters table
    logger.info("%s", "\n" + "=" * 80)
    logger.info("BEST HYPERPARAMETERS")
    logger.info("%s", "=" * 80)
    logger.info("%s", "\n" + hyperparams_df.to_string(index=False))

    logger.info("%s", "\n" + "=" * 80)
    logger.info("EXTRACTION COMPLETE!")
    logger.info("%s", "=" * 80)
    logger.info("Files saved to: %s/", OUTPUT_DIR)
    logger.info("Next steps:")
    logger.info("  1. Review the extracted configurations")
    logger.info("  2. Run gather_model_details.py to get parameter counts and benchmarks")
    logger.info("  3. Use analyze_physics_results.py for visualizations")
    logger.info("%s", "\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
