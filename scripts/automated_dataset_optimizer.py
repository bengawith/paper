"""
Automated Dataset Optimizer for Physics-Aware Hyperparameter Tuning

Uses physics-aware metrics and LOFO cross-validation for robust evaluation.
"""

import os
import json
import logging
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dataset_optimization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Use shared utilities from organised.src.utils
try:
    from organised.src.utils import (
        build_cst_feature_columns,
        extract_cst_params_from_coordinates,
        create_exponential_masked_dataset,
        get_aerofoil_family,
        create_lofo_splits,
        QuickMLP,
        train_quick_model,
        evaluate_quick_model,
        extract_known_alpha_ranges,
        calculate_physical_metrics,
        compare_physical_metrics,
    )
except Exception as e:
    logger.error(f"Failed to import shared utilities from organised.src.utils: {e}")
    raise

def compare_physical_metrics(
    alpha: np.ndarray,
    cl_actual: np.ndarray,
    cl_pred: np.ndarray,
    alpha_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Compare physical metrics between actual and predicted.

    Args:
        alpha: Array of angle of attack values (standardised grid)
        cl_actual: Actual CL values
        cl_pred: Predicted CL values
        alpha_range: Optional (min_alpha, max_alpha) to filter to known range

    Returns:
        Dict with error values for each metric
    """
    # Filter to known alpha range if provided
    if alpha_range is not None:
        min_alpha, max_alpha = alpha_range
        mask = (alpha >= min_alpha) & (alpha <= max_alpha)
        alpha_filtered = alpha[mask]
        cl_actual_filtered = cl_actual[mask]
        cl_pred_filtered = cl_pred[mask]
    else:
        alpha_filtered = alpha
        cl_actual_filtered = cl_actual
        cl_pred_filtered = cl_pred

    actual_metrics = calculate_physical_metrics(alpha_filtered, cl_actual_filtered)
    pred_metrics = calculate_physical_metrics(alpha_filtered, cl_pred_filtered)

    errors = {}
    for key in ["cl_max", "alpha_stall", "alpha_zero_lift", "lift_curve_slope"]:
        errors[f"{key}_error"] = abs(actual_metrics[key] - pred_metrics[key])

    return errors


def optimize_cst_parameter_counts(
    csv_file: str,
    output_dir: str = "dataset_optimization_results",
    device: Optional[torch.device] = None,
) -> Dict:
    """Optimize the number of CST (Kulfan) parameters using exponential masking.

    For each n_weights_per_side config, CST params are extracted from coordinate
    files, an exponential-masked dataset is generated, and LOFO evaluation is run.
    Returns a dict with per-config results and the best configuration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("\n" + "=" * 70)
    logger.info("CST PARAMETER COUNT OPTIMIZATION")
    logger.info(
        "Testing different n_weights_per_side values (extracted from coordinates)"
    )
    logger.info("=" * 70)

    # Test different n_weights_per_side values
    configs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Load base CSV with aerodynamic data
    logger.info(f"Loading base aerodynamic data from {csv_file}...")
    df_base = pd.read_csv(csv_file)

    # Extract known alpha ranges for each airfoil
    logger.info("Extracting known alpha ranges for each airfoil...")
    known_alpha_ranges = extract_known_alpha_ranges(df_base)
    logger.info(f"Extracted alpha ranges for {len(known_alpha_ranges)} airfoils")
    example_names = list(known_alpha_ranges.keys())[:3]
    for name in example_names:
        min_a, max_a = known_alpha_ranges[name]
        logger.info(f"  {name}: [{min_a:.2f}, {max_a:.2f}]")

    alpha_cols = [f"alpha_{i}" for i in range(48)]
    CL_cols = [f"CL_{i}" for i in range(48)]
    y_alpha = df_base[alpha_cols]
    y_CL = df_base[CL_cols]
    standardised_grid = np.linspace(-20, 20, 48)

    results = {}

    for n_weights in configs:
        total_params = 2 * n_weights + 2
        logger.info(f"\n{'=' * 70}")
        logger.info(
            f"Testing n_weights_per_side={n_weights} ({total_params} total parameters)"
        )
        logger.info(f"{'=' * 70}")

        logger.info(
            f"Extracting {n_weights} CST weights per side from coordinate files..."
        )
        df_with_cst = extract_cst_params_from_coordinates(
            df_base, n_weights_per_side=n_weights, data_dir="data/aerofoil_data"
        )

        # Build X columns
        X_cols = ["aerofoil_name"] + build_cst_feature_columns(n_weights)

        # Create dataset using exponential masking with linear extrapolation
        test_df = create_exponential_masked_dataset(
            df_with_cst, X_cols, y_alpha, y_CL, standardised_grid
        )
        feature_cols = [col for col in X_cols if col != "aerofoil_name"] + [
            f"dist_{i}" for i in range(48)
        ]

        test_df["family"] = test_df["aerofoil_name"].apply(get_aerofoil_family)

        # Create LOFO splits
        lofo_splits = create_lofo_splits(test_df, n_families=5, min_samples=50)

        input_size = len(feature_cols)
        output_size = 48

        lofo_results = []

        for family, train_idx, test_idx in lofo_splits:
            logger.info(f"  Fold: {family}")

            X_train = test_df.loc[train_idx, feature_cols].values.astype(np.float32)
            y_train = test_df.loc[train_idx, CL_cols].values.astype(np.float32)
            X_test = test_df.loc[test_idx, feature_cols].values.astype(np.float32)
            y_test = test_df.loc[test_idx, CL_cols].values.astype(np.float32)

            test_airfoil_names = test_df.loc[test_idx, "aerofoil_name"].values
            test_alpha_ranges = [
                known_alpha_ranges[name] for name in test_airfoil_names
            ]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Distance mask
            mask_cols = [f"dist_{i}" for i in range(48)]
            mask_train = test_df.loc[train_idx, mask_cols].values.astype(np.float32)

            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled),
                torch.FloatTensor(y_train),
                torch.FloatTensor(mask_train),
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            model = QuickMLP(input_size, output_size).to(device)
            model = train_quick_model(
                model, train_loader, device, epochs=30, use_masked_loss=True
            )

            eval_results = evaluate_quick_model(
                model,
                X_test_scaled,
                y_test,
                standardised_grid,
                device,
                test_alpha_ranges,
            )
            lofo_results.append(eval_results)
            logger.info(f"    Physics Score: {eval_results['physics_score']:.4f}")

        config_name = f"{total_params}_params_{n_weights}weights_per_side"
        results[config_name] = {
            "n_params": total_params,
            "n_weights_per_side": n_weights,
            "mean_physics_score": np.mean([r["physics_score"] for r in lofo_results]),
            "std_physics_score": np.std([r["physics_score"] for r in lofo_results]),
            "mean_mse": np.mean([r["mse"] for r in lofo_results]),
            "lofo_results": lofo_results,
        }

        logger.info(f"\n{config_name} Summary:")
        logger.info(
            f"  Mean Physics Score: {results[config_name]['mean_physics_score']:.4f} +/- "
            f"{results[config_name]['std_physics_score']:.4f}"
        )

    best_config = min(
        [k for k in results.keys()], key=lambda k: results[k]["mean_physics_score"]
    )
    results["best_configuration"] = best_config
    results["parameter_comparison"] = {
        config: {
            "n_params": results[config]["n_params"],
            "physics_score": results[config]["mean_physics_score"],
        }
        for config in results.keys()
        if config not in ["best_configuration", "parameter_comparison"]
    }

    logger.info(f"\n{'=' * 70}")
    logger.info(f"[OPTIMAL] {best_config}")
    logger.info(f"   Physics Score: {results[best_config]['mean_physics_score']:.4f}")
    logger.info(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "cst_parameter_optimization.json")

    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    results_serializable = convert_to_json_serializable(results)

    with open(results_file, "w") as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    return results


def generate_final_dataset(
    csv_file: str, optimal_n_weights: int, output_dir: str = "data/csv"
) -> str:
    """Generate and save the final optimized dataset using exponential masking.

    Args:
        csv_file: Path to base CSV with aerodynamic data
        optimal_n_weights: Optimal n_weights_per_side value
        output_dir: Output directory for final dataset

    Returns:
        Path to saved CSV file
    """
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING FINAL OPTIMIZED DATASET (EXPONENTIAL MASKING)")
    logger.info("=" * 70)

    total_params = 2 * optimal_n_weights + 2

    logger.info(
        f"Parameters: {total_params} ({optimal_n_weights} weights per side + LE + TE)"
    )

    # Load base CSV with aerodynamic data
    logger.info(f"Loading base aerodynamic data from {csv_file}...")
    df_base = pd.read_csv(csv_file)

    # Extract CST parameters from coordinates
    logger.info(
        f"Extracting {optimal_n_weights} CST weights per side from coordinate files..."
    )
    df = extract_cst_params_from_coordinates(
        df_base, n_weights_per_side=optimal_n_weights, data_dir="data/aerofoil_data"
    )

    X_cols = ["aerofoil_name"] + build_cst_feature_columns(optimal_n_weights)

    alpha_cols = [f"alpha_{i}" for i in range(48)]
    CL_cols = [f"CL_{i}" for i in range(48)]
    y_alpha = df[alpha_cols]
    y_CL = df[CL_cols]
    standardised_grid = np.linspace(-20, 20, 48)

    # Create final dataset
    final_df = create_exponential_masked_dataset(
        df, X_cols, y_alpha, y_CL, standardised_grid
    )
    final_df["family"] = final_df["aerofoil_name"].apply(get_aerofoil_family)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"optimized_dataset_expmask_{total_params}params_kulfan.csv"
    )
    final_df.to_csv(output_file, index=False)

    logger.info(f"\n[SUCCESS] Final dataset saved to: {output_file}")
    logger.info(f"   Shape: {final_df.shape}")
    logger.info(f"   Samples: {len(final_df)}")
    logger.info(
        f"   CST Parameters: {optimal_n_weights} weights per side ({total_params} total)"
    )
    logger.info(
        f"   Features: {len([c for c in final_df.columns if c not in ['aerofoil_name', 'family'] and c.startswith(('lower', 'upper', 'leading', 'TE'))])}"
    )
    logger.info(f"   Families: {final_df['family'].nunique()}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Automated Dataset Optimizer")
    parser.add_argument(
        "--csv", type=str, default="data/csv/14KP_48CLA.csv", help="Input CSV file path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_optimization_results",
        help="Output directory for results",
    )
    # This script focuses on determining optimal CST parameter count
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for training",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("=" * 70)
    logger.info("AUTOMATED DATASET OPTIMIZATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Input file: {args.csv}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")

    # Run CST parameter optimization using exponential masking
    logger.info("\n[STEP] Optimizing CST parameter count using exponential masking...")
    param_results = optimize_cst_parameter_counts(args.csv, args.output_dir, device)
    best_config = param_results["best_configuration"]
    optimal_n_weights = param_results[best_config]["n_weights_per_side"]

    # Step 3: Generate final dataset
    logger.info(
        "\n[STEP] Generating final optimized dataset with exponential masking..."
    )
    final_file = generate_final_dataset(args.csv, optimal_n_weights, "data/csv")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
