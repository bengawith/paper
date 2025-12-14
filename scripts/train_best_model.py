"""train_best_model

Train and evaluate the best model from physics-aware tuning.
"""

import os
import sys

sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from src.utils import compare_physical_metrics, set_seed
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.models import instantiate_model, create_optimizer

USE_IMPROVED_MODELS = True
logger = logging.getLogger(__name__)
logger.info("Using models from `src.models` module")

try:
    from src.utils import set_seed
except ImportError:

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


set_seed(42)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train_best_model.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ==================== Load Best Configuration ====================


def load_best_configuration(results_dir: str) -> Tuple[str, str, Dict, float]:
    """
    Load best model configuration from tuning results.

    Returns:
    --------
    model_name : str (MLP, GRU, LSTM, CNN)
    loss_name : str (mse, huber, log_cosh, smooth_l1)
    best_params : dict
    best_score : float
    """
    logger.info("=" * 80)
    logger.info("LOADING BEST MODEL CONFIGURATION")
    logger.info("=" * 80)

    results_path = Path(results_dir)

    # Find comprehensive results file
    comprehensive_files = list(results_path.glob("comprehensive_results_*.json"))

    if not comprehensive_files:
        raise FileNotFoundError(f"No comprehensive results found in {results_dir}")

    # Use most recent file
    latest_file = max(comprehensive_files, key=os.path.getctime)
    logger.info(f"Loading from: {latest_file}")

    with open(latest_file, "r") as f:
        all_results = json.load(f)

    # Find best model-loss combination
    best_overall_score = float("inf")
    best_model_name = None
    best_loss_name = None
    best_params = None

    for model_name, loss_dict in all_results.items():
        for loss_name, data in loss_dict.items():
            lofo_results = data.get("lofo_results", [])
            if len(lofo_results) == 0:
                continue

            avg_physics_score = np.mean([r["physics_score"] for r in lofo_results])

            if avg_physics_score < best_overall_score:
                best_overall_score = avg_physics_score
                best_model_name = model_name
                best_loss_name = loss_name
                best_params = data["best_params"]

    logger.info(f"\n✓ Best Model: {best_model_name}")
    logger.info(f"✓ Loss Function: {best_loss_name}")
    logger.info(f"✓ Physics Score: {best_overall_score:.4f}")
    logger.info("\n✓ Hyperparameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    return best_model_name, best_loss_name, best_params, best_overall_score


# ==================== Model Creation ====================


def create_model(
    model_name: str,
    input_size: int,
    output_size: int,
    params: Dict,
    device: torch.device,
) -> nn.Module:
    """Create model based on configuration."""
    model_map = {
        "MLP": "MLPModel",
        "GRU": "GRUModel",
        "LSTM": "LSTMModel",
        "CNN": "CNNModel",
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    model = instantiate_model(
        net_name=model_map[model_name],
        input_size=input_size,
        output_size=output_size,
        params={
            "hidden_size": params.get("hidden_size"),
            "num_layers": params.get("num_layers"),
            "dropout": params.get("dropout_rate", 0.1),
            "bidirectional": params.get("bidirectional", True),
            "width_factor": params.get("width_factor", 2),
            "num_filters": params.get("num_filters"),
            "kernel_sizes": (
                params.get("kernel_sizes")
                or (
                    [params.get("kernel_size")] * len(params.get("num_filters", []))
                    if params.get("kernel_size") is not None
                    else None
                )
            ),
        },
        device=device,
    )

    return model


def get_loss_fn(loss_name: str):
    """Get loss function by name."""
    if loss_name == "mse":
        return nn.MSELoss(reduction="none")  # Element-wise for distance weighting
    elif loss_name == "huber":
        return nn.HuberLoss(delta=1.0, reduction="none")
    elif loss_name == "log_cosh":

        def log_cosh_loss(y_pred, y_true):
            diff = y_pred - y_true
            return torch.log(
                torch.cosh(diff + 1e-12)
            )  # Element-wise for distance weighting

        return log_cosh_loss
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none")
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ==================== Data Loading with Advanced Scaling ====================


def analyze_feature_distributions(
    df: pd.DataFrame, feature_cols: List[str]
) -> Dict[str, str]:
    """
    Analyze feature distributions to recommend optimal scaler.

    Returns:
    --------
    dict : Recommended scaler for each feature type
    """
    recommendations = {}

    for col in feature_cols:
        data = df[col].values

        # Check for outliers using IQR method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outlier_count = np.sum((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr))
        outlier_ratio = outlier_count / len(data)

        # Check skewness
        skewness = stats.skew(data)

        if outlier_ratio > 0.1 or abs(skewness) > 2:
            recommendations[col] = "robust"
        else:
            recommendations[col] = "standard"

    return recommendations


def load_and_prepare_data(
    csv_file: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler_type: str = "auto",
) -> Tuple:
    """
    Load data and create train/val/test splits with intelligent scaling.

    Parameters:
    -----------
    csv_file : str
        Path to CSV file
    test_size : float
        Test set proportion
    val_size : float
        Validation set proportion
    scaler_type : str
        'standard', 'robust', 'minmax', or 'auto' (automatically selects best)

    Returns:
    --------
    Tuple of (train_loader, val_loader, test_loader, scalers, metadata, alpha_grid)
    """
    logger.info("=" * 80)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 80)

    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} samples from {csv_file}")

    # Feature columns (12 CST parameters: 5 lower + 5 upper + TE + leading edge)
    feature_cols = [
        col
        for col in df.columns
        if col.startswith(
            ("lower_weight_", "upper_weight_", "leading_edge_weight", "TE_thickness")
        )
    ]

    # Target columns (48 CL values)
    cl_cols = [f"CL_{i}" for i in range(48)]

    # Distance mask columns (48 continuous distance values, 0-1)
    dist_cols = [f"dist_{i}" for i in range(48)]

    # Fixed alpha grid (standard for this dataset)
    alpha_grid = np.linspace(-20, 20, 48)

    # Prepare arrays
    X = df[feature_cols].values.astype(np.float32)
    y = df[cl_cols].values.astype(np.float32)
    distances = df[dist_cols].values.astype(np.float32)
    airfoil_names = df["aerofoil_name"].values

    logger.info(
        f"Features: {X.shape[1]} CST parameters (12 = 5+5 weights + TE + leading edge)"
    )
    logger.info(f"Targets: {y.shape[1]} CL values")
    logger.info(
        f"Distance masks: {distances.shape[1]} continuous distance values (0-1)"
    )
    logger.info(f"Average distance: {distances.mean():.4f} (higher = better quality)")
    logger.info(
        f"Min distance: {distances.min():.4f}, Max distance: {distances.max():.4f}"
    )

    # Analyze feature distributions for optimal scaling
    if scaler_type == "auto":
        logger.info("\nAnalyzing feature distributions for optimal scaling...")
        recommendations = analyze_feature_distributions(df, feature_cols)

        robust_count = sum(1 for v in recommendations.values() if v == "robust")
        if robust_count > len(feature_cols) / 2:
            scaler_type = "robust"
            logger.info(
                f"Auto-selected RobustScaler ({robust_count}/{len(feature_cols)} features have outliers/skew)"
            )
        else:
            scaler_type = "standard"
            logger.info("Auto-selected StandardScaler (most features well-distributed)")

    # Train/temp split
    X_train, X_temp, y_train, y_temp, dist_train, dist_temp, names_train, names_temp = (
        train_test_split(
            X,
            y,
            distances,
            airfoil_names,
            test_size=(test_size + val_size),
            random_state=42,
        )
    )

    # Val/test split
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test, dist_val, dist_test, names_val, names_test = (
        train_test_split(
            X_temp,
            y_temp,
            dist_temp,
            names_temp,
            test_size=(1 - val_ratio),
            random_state=42,
        )
    )

    logger.info("\nDataset splits:")
    logger.info(f"  Train: {len(X_train)} samples ({len(X_train) / len(X) * 100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} samples ({len(X_val) / len(X) * 100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}%)")

    # Create scalers based on selected type
    logger.info(f"\nApplying {scaler_type.upper()} scaling strategy:")

    if scaler_type == "standard":
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
    elif scaler_type == "robust":
        X_scaler = RobustScaler()
        y_scaler = RobustScaler()
    elif scaler_type == "minmax":
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    # Scale features
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)

    # Scale targets
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    logger.info(f"Features (CST params): {scaler_type}")
    logger.info(
        f"    - Mean: {X_scaler.mean_[:3] if hasattr(X_scaler, 'mean_') else 'N/A'}"
    )
    logger.info(
        f"    - Scale: {X_scaler.scale_[:3] if hasattr(X_scaler, 'scale_') else 'N/A'}"
    )
    logger.info(f"Targets (CL values): {scaler_type}")
    logger.info(
        f"    - Mean: {y_scaler.mean_[:3] if hasattr(y_scaler, 'mean_') else 'N/A'}"
    )
    logger.info(
        f"    - Scale: {y_scaler.scale_[:3] if hasattr(y_scaler, 'scale_') else 'N/A'}"
    )
    logger.info("Distance masks: NOT SCALED (continuous 0-1 weights)")

    # Create dataloaders with distance masks
    batch_size = 32

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled),
        torch.FloatTensor(dist_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled),
        torch.FloatTensor(dist_val),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test_scaled),
        torch.FloatTensor(dist_test),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    scalers = {"X_scaler": X_scaler, "y_scaler": y_scaler, "scaler_type": scaler_type}

    metadata = {
        "feature_cols": feature_cols,
        "cl_cols": cl_cols,
        "dist_cols": dist_cols,
        "X_train": X_train,
        "y_train": y_train,
        "dist_train": dist_train,
        "X_val": X_val,
        "y_val": y_val,
        "dist_val": dist_val,
        "X_test": X_test,
        "y_test": y_test,
        "dist_test": dist_test,
        "names_test": names_test,
    }

    return train_loader, val_loader, test_loader, scalers, metadata, alpha_grid


# ==================== Training with Distance-Weighted Loss ====================


def compute_physics_score_for_batch(model, dataloader, device, y_scaler, alpha_grid):
    """Compute physics score on a batch (for validation monitoring)."""
    model.eval()
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for X_batch, y_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            y_pred_scaled = model(X_batch).cpu().numpy()
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            y_true = y_scaler.inverse_transform(y_batch.cpu().numpy())

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)

    # Compute physics metrics
    physics_errors = []
    for i in range(len(all_y_true)):
        errors, _, _ = compare_physical_metrics(
            alpha_grid, all_y_true[i], all_y_pred[i]
        )
        physics_errors.append(errors)

    # Calculate Physics Score (weighted combination)
    cl_max_error = np.mean([e["cl_max_error"] for e in physics_errors])
    alpha_stall_error = np.mean([e["alpha_stall_error"] for e in physics_errors])
    alpha_zero_error = np.mean([e["alpha_zero_lift_error"] for e in physics_errors])
    slope_error = np.mean([e["lift_curve_slope_error"] for e in physics_errors])

    physics_score = (
        1.0 * cl_max_error
        + 0.5 * alpha_stall_error
        + 0.3 * alpha_zero_error
        + 10.0 * slope_error
    )

    return physics_score


def train_epoch_distance_weighted(model, train_loader, optimizer, loss_fn, device):
    """Train one epoch using distance-weighted (continuous 0-1) loss.

    Each element-wise loss is weighted by the corresponding distance value and
    aggregated as a weighted mean.
    """
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for X_batch, y_batch, dist_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        dist_batch = dist_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_batch)

        # Compute element-wise loss
        loss_elements = loss_fn(y_pred, y_batch)

        weighted_loss = loss_elements * dist_batch

        batch_weight = dist_batch.sum()
        if batch_weight > 0:
            loss = weighted_loss.sum() / batch_weight
        else:
            loss = weighted_loss.mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_weight.item()
        total_weight += batch_weight.item()

    return total_loss / max(total_weight, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    loss_fn,
    device,
    weight_decay=1e-5,
    patience=30,
    y_scaler=None,
    alpha_grid=None,
):
    """Train model with early stopping and distance-weighted loss."""

    logger.info("=" * 80)
    logger.info("TRAINING MODEL (DISTANCE-WEIGHTED LOSS)")
    logger.info("=" * 80)

    if USE_IMPROVED_MODELS:
        optimizer = create_optimizer(model, "adamw", learning_rate, weight_decay)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    best_val_loss = float("inf")
    best_val_physics = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    train_losses = []
    val_losses = []
    train_loss_history = []
    val_physics_history = []

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch_distance_weighted(
            model, train_loader, optimizer, loss_fn, device
        )
        train_losses.append(train_loss)
        train_loss_history.append((epoch + 1, train_loss))

        # Validation
        model.eval()
        val_loss = 0.0
        total_weight = 0.0

        with torch.no_grad():
            for X_batch, y_batch, dist_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                dist_batch = dist_batch.to(device)

                y_pred = model(X_batch)
                loss_elements = loss_fn(y_pred, y_batch)
                weighted_loss = loss_elements * dist_batch

                batch_weight = dist_batch.sum()
                val_loss += weighted_loss.sum().item()
                total_weight += batch_weight.item()

        val_loss /= max(total_weight, 1)
        val_losses.append(val_loss)

        # Compute physics score on validation set (every 5 epochs to save time)
        if (
            y_scaler is not None
            and alpha_grid is not None
            and ((epoch + 1) % 5 == 0 or epoch == 0)
        ):
            val_physics = compute_physics_score_for_batch(
                model, val_loader, device, y_scaler, alpha_grid
            )
            val_physics_history.append((epoch + 1, val_physics))

        # Print progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_physics = (
                val_physics
                if ((epoch + 1) % 5 == 0 or epoch == 0)
                else best_val_physics
            )
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    logger.info("\n✓ Training complete!")
    logger.info(f"✓ Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")

    return (
        model,
        train_losses,
        val_losses,
        train_loss_history,
        val_physics_history,
        best_epoch,
    )


# ==================== Evaluation ====================


def evaluate_model(model, test_loader, scalers, alpha_grid, device):
    """Comprehensive evaluation with physics metrics."""

    logger.info("=" * 80)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 80)

    model.eval()

    all_predictions = []
    all_targets = []
    all_distances = []

    with torch.no_grad():
        for X_batch, y_batch, dist_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_true = y_batch.numpy()
            dist = dist_batch.numpy()

            all_predictions.append(y_pred)
            all_targets.append(y_true)
            all_distances.append(dist)

    y_pred_scaled = np.vstack(all_predictions)
    y_true_scaled = np.vstack(all_targets)
    distances = np.vstack(all_distances)

    # Inverse transform to original scale
    y_true = scalers["y_scaler"].inverse_transform(y_true_scaled)
    y_pred = scalers["y_scaler"].inverse_transform(y_pred_scaled)

    # Calculate MSE on original scale (overall and weighted)
    mse_total = np.mean((y_true - y_pred) ** 2)

    # Distance-weighted MSE
    mse_weighted = np.sum(((y_true - y_pred) ** 2) * distances) / np.sum(distances)

    # MSE on scaled data
    mse_scaled = np.mean((y_true_scaled - y_pred_scaled) ** 2)

    # R-squared score
    r2 = r2_score(y_true, y_pred)

    # Physics metrics
    physics_errors = []
    actual_metrics_list = []
    pred_metrics_list = []

    for i in range(len(y_true)):
        errors, actual_m, pred_m = compare_physical_metrics(
            alpha_grid, y_true[i], y_pred[i]
        )
        physics_errors.append(errors)
        actual_metrics_list.append(actual_m)
        pred_metrics_list.append(pred_m)

    # Aggregate
    physics_summary = {
        "mse_total": float(mse_total),
        "mse_weighted": float(mse_weighted),
        "mse_scaled": float(mse_scaled),
        "r2_score": float(r2),
        "avg_distance": float(distances.mean()),
        "min_distance": float(distances.min()),
        "max_distance": float(distances.max()),
        "cl_max_error_mean": float(
            np.mean([e["cl_max_error"] for e in physics_errors])
        ),
        "cl_max_error_std": float(np.std([e["cl_max_error"] for e in physics_errors])),
        "alpha_stall_error_mean": float(
            np.mean([e["alpha_stall_error"] for e in physics_errors])
        ),
        "alpha_stall_error_std": float(
            np.std([e["alpha_stall_error"] for e in physics_errors])
        ),
        "alpha_zero_lift_error_mean": float(
            np.mean([e["alpha_zero_lift_error"] for e in physics_errors])
        ),
        "alpha_zero_lift_error_std": float(
            np.std([e["alpha_zero_lift_error"] for e in physics_errors])
        ),
        "lift_curve_slope_error_mean": float(
            np.mean([e["lift_curve_slope_error"] for e in physics_errors])
        ),
        "lift_curve_slope_error_std": float(
            np.std([e["lift_curve_slope_error"] for e in physics_errors])
        ),
    }

    logger.info("Evaluation Metrics:")
    logger.info(f"  MSE (total):        {mse_total:.6f}")
    logger.info(
        f"  MSE (weighted):     {mse_weighted:.6f} ← Distance-weighted (most fair)"
    )
    logger.info(f"  MSE (scaled):       {mse_scaled:.6f}")
    logger.info(f"  R² Score:           {r2:.4f}")
    logger.info(f"  Average distance:   {distances.mean():.4f} (quality indicator)")
    logger.info(
        f"\n  ΔCL_max:            {physics_summary['cl_max_error_mean']:.4f} ± {physics_summary['cl_max_error_std']:.4f}"
    )
    logger.info(
        f"  Δα_stall:           {physics_summary['alpha_stall_error_mean']:.2f}° ± {physics_summary['alpha_stall_error_std']:.2f}°"
    )
    logger.info(
        f"  Δα_zero_lift:       {physics_summary['alpha_zero_lift_error_mean']:.2f}° ± {physics_summary['alpha_zero_lift_error_std']:.2f}°"
    )
    logger.info(
        f"  Δlift_curve_slope:  {physics_summary['lift_curve_slope_error_mean']:.4f} ± {physics_summary['lift_curve_slope_error_std']:.4f}"
    )

    results = {
        "predictions": {"y_true": y_true, "y_pred": y_pred, "distances": distances},
        "physics_errors": physics_errors,
        "actual_metrics": actual_metrics_list,
        "pred_metrics": pred_metrics_list,
        "summary": physics_summary,
    }

    return results


# ==================== Visualization ====================


def plot_training_curves(train_loss_history, val_physics_history, best_epoch, save_dir):
    """Plot training and validation curves with enhanced features."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract epochs and values
    train_epochs = [e for e, _ in train_loss_history]
    train_losses = [loss for _, loss in train_loss_history]
    val_epochs = [e for e, _ in val_physics_history]
    val_physics = [p for _, p in val_physics_history]

    # Plot training loss (weighted log-cosh)
    ax.plot(
        train_epochs,
        train_losses,
        "b-",
        linewidth=2,
        label="Weighted Log-Cosh (train)",
        alpha=0.7,
    )

    # Plot validation physics score
    ax.plot(
        val_epochs,
        val_physics,
        "r-",
        linewidth=2,
        label="Physics Score (val)",
        marker="o",
        markersize=4,
    )

    # Mark early stopping epoch
    ax.axvline(
        x=best_epoch,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Early Stop (epoch {best_epoch})",
        alpha=0.7,
    )

    # Set identical y-axis range for both curves
    all_values = train_losses + val_physics
    y_min = min(all_values) * 0.95
    y_max = max(all_values) * 1.05
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss / Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Training Progress: Weighted Log-Cosh vs Physics Score",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_curves.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: training_curves.png")
    plt.close()


def plot_lift_curves(results, names_test, alpha_grid, save_dir, n_samples=6):
    """Plot sample lift curves: CL vs alpha with distance quality coloring."""

    y_true = results["predictions"]["y_true"]
    y_pred = results["predictions"]["y_pred"]
    distances = results["predictions"]["distances"]

    # Select diverse samples
    indices = np.linspace(0, len(y_true) - 1, len(y_true), dtype=int)
    indices = [110, 235, 69, 41, 179, 221]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        dist = distances[idx]
        avg_dist = dist.mean()

        # Create colormap based on distance
        colors_true = plt.cm.Blues(dist)
        colors_pred = plt.cm.Oranges(dist)

        # Plot curves
        ax.plot(
            alpha_grid,
            y_true[idx],
            "o-",
            label="Ground Truth",
            linewidth=2,
            markersize=4,
            alpha=0.4,
            color="blue",
        )
        ax.plot(
            alpha_grid,
            y_pred[idx],
            "s-",
            label="Prediction",
            linewidth=2,
            markersize=4,
            alpha=0.4,
            color="orange",
        )

        # Overlay points colored by distance quality
        for i in range(len(alpha_grid)):
            ax.plot(
                alpha_grid[i],
                y_true[idx][i],
                "o",
                markersize=6,
                color=colors_true[i],
                alpha=0.8,
            )
            ax.plot(
                alpha_grid[i],
                y_pred[idx][i],
                "s",
                markersize=6,
                color=colors_pred[i],
                alpha=0.8,
            )

        # Mark key points
        actual_m = results["actual_metrics"][idx]
        pred_m = results["pred_metrics"][idx]

        ax.axhline(y=actual_m["cl_max"], color="blue", linestyle="--", alpha=0.3)
        ax.axvline(x=actual_m["alpha_stall"], color="blue", linestyle="--", alpha=0.3)
        ax.axhline(y=pred_m["cl_max"], color="orange", linestyle=":", alpha=0.3)
        ax.axvline(x=pred_m["alpha_stall"], color="orange", linestyle=":", alpha=0.3)

        ax.set_xlabel("Angle of Attack (°)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Lift Coefficient", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{names_test[idx]}\n(Avg Distance: {avg_dist:.3f}, darker = higher quality)",
            fontsize=10,
            fontweight="bold",
        )
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "lift_curves_sample.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: lift_curves_sample.png")
    plt.close()


def plot_error_distributions(results, save_dir):
    """Plot distribution of physics errors."""

    physics_errors = results["physics_errors"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("cl_max_error", "ΔCL_max", axes[0, 0]),
        ("alpha_stall_error", "Δα_stall (°)", axes[0, 1]),
        ("alpha_zero_lift_error", "Δα_zero_lift (°)", axes[1, 0]),
        ("lift_curve_slope_error", "Δlift_curve_slope", axes[1, 1]),
    ]

    for key, label, ax in metrics:
        errors = [e[key] for e in physics_errors]

        ax.hist(errors, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(
            np.mean(errors),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(errors):.4f}",
        )
        ax.axvline(
            np.median(errors),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(errors):.4f}",
        )

        ax.set_xlabel(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax.set_title(f"Distribution of {label}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / "error_distributions.png", dpi=300, bbox_inches="tight"
    )
    logger.info("Saved: error_distributions.png")
    plt.close()


def plot_prediction_scatter(results, save_dir):
    """Scatter plots: predicted vs actual for CL with full-set vs high-quality insets."""

    y_true = results["predictions"]["y_true"].flatten()
    y_pred = results["predictions"]["y_pred"].flatten()
    distances = results["predictions"]["distances"].flatten()

    # Calculate metrics for full set
    mse_full = np.mean((y_true - y_pred) ** 2)
    r2_full = r2_score(y_true, y_pred)

    # Calculate metrics for high-quality subset
    high_quality_mask = distances > 0.9
    y_true_hq = y_true[high_quality_mask]
    y_pred_hq = y_pred[high_quality_mask]
    mse_hq = np.mean((y_true_hq - y_pred_hq) ** 2) if len(y_true_hq) > 0 else 0
    r2_hq = r2_score(y_true_hq, y_pred_hq) if len(y_true_hq) > 0 else 0

    fig = plt.figure(figsize=(14, 6))

    # Main plot: Full dataset
    ax_main = plt.subplot(1, 2, 1)

    # Color by mask value
    scatter = ax_main.scatter(
        y_true, y_pred, c=distances, cmap="viridis", alpha=0.5, s=10, vmin=0, vmax=1
    )

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax_main.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
        alpha=0.7,
    )

    ax_main.set_xlabel("Actual CL", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Predicted CL", fontsize=12, fontweight="bold")
    ax_main.set_title(
        f"Full Dataset\nR²={r2_full:.4f}, MSE={mse_full:.6f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect("equal", adjustable="box")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label("Distance Quality", fontsize=10, fontweight="bold")

    # Inset plot: High-quality subset (distance > 0.9)
    ax_inset = plt.subplot(1, 2, 2)

    ax_inset.scatter(
        y_true_hq,
        y_pred_hq,
        c="green",
        alpha=0.6,
        s=15,
        label=f"High Quality (dist>0.9, n={len(y_true_hq)})",
    )

    # Perfect prediction line
    if len(y_true_hq) > 0:
        min_val_hq = min(y_true_hq.min(), y_pred_hq.min())
        max_val_hq = max(y_true_hq.max(), y_pred_hq.max())
        ax_inset.plot(
            [min_val_hq, max_val_hq],
            [min_val_hq, max_val_hq],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
            alpha=0.7,
        )

    ax_inset.set_xlabel("Actual CL", fontsize=12, fontweight="bold")
    ax_inset.set_ylabel("Predicted CL", fontsize=12, fontweight="bold")
    ax_inset.set_title(
        f"High-Quality Subset (dist>0.9)\nR²={r2_hq:.4f}, MSE={mse_hq:.6f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_inset.legend(loc="upper left", fontsize=9)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_aspect("equal", adjustable="box")

    # Overall title
    fig.suptitle(
        "CL Predictions: Full Dataset vs High-Quality Subset",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "prediction_scatter.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: prediction_scatter.png")
    plt.close()


def plot_physics_metrics_comparison(results, save_dir):
    """Compare physics metrics: true vs predicted."""

    actual_metrics = results["actual_metrics"]
    pred_metrics = results["pred_metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("cl_max", "CL_max", axes[0, 0]),
        ("alpha_stall", "α_stall (°)", axes[0, 1]),
        ("alpha_zero_lift", "α_zero_lift (°)", axes[1, 0]),
        ("lift_curve_slope", "Lift Curve Slope", axes[1, 1]),
    ]

    for key, label, ax in metrics:
        actual_vals = [m[key] for m in actual_metrics]
        pred_vals = [m[key] for m in pred_metrics]

        ax.scatter(actual_vals, pred_vals, alpha=0.5, s=30)

        min_val = min(min(actual_vals), min(pred_vals))
        max_val = max(max(actual_vals), max(pred_vals))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Match",
        )

        # Calculate R²
        correlation = np.corrcoef(actual_vals, pred_vals)[0, 1]
        r_squared = correlation**2

        ax.set_xlabel(f"Actual {label}", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Predicted {label}", fontsize=11, fontweight="bold")
        ax.set_title(f"{label} (R² = {r_squared:.4f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / "physics_metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    logger.info("Saved: physics_metrics_comparison.png")
    plt.close()


def plot_distance_quality_analysis(results, alpha_grid, save_dir):
    """Analyze and visualize distance quality distribution."""

    distances = results["predictions"]["distances"]

    # Average distance per alpha point
    dist_per_point = distances.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Distance vs alpha
    axes[0].plot(alpha_grid, dist_per_point, linewidth=2, marker="o", color="purple")
    axes[0].fill_between(alpha_grid, 0, dist_per_point, alpha=0.3, color="purple")
    axes[0].axhline(
        y=0.9, color="green", linestyle="--", alpha=0.5, label="High Quality (>0.9)"
    )
    axes[0].axhline(
        y=0.7, color="orange", linestyle="--", alpha=0.5, label="Medium Quality (>0.7)"
    )
    axes[0].set_xlabel("Angle of Attack (°)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Average Distance Quality", fontsize=12, fontweight="bold")
    axes[0].set_title("Data Quality Across Alpha Range", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Histogram of sample average distance
    sample_avg_dist = distances.mean(axis=1)
    axes[1].hist(sample_avg_dist, bins=30, alpha=0.7, color="purple", edgecolor="black")
    axes[1].axvline(
        sample_avg_dist.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {sample_avg_dist.mean():.3f}",
    )
    axes[1].axvline(
        np.median(sample_avg_dist),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(sample_avg_dist):.3f}",
    )
    axes[1].set_xlabel("Average Distance per Sample", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[1].set_title("Distribution of Sample Quality", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / "distance_quality_analysis.png", dpi=300, bbox_inches="tight"
    )
    logger.info("Saved: distance_quality_analysis.png")
    plt.close()


# ==================== Save Model ====================


def save_model_and_artifacts(
    model, scalers, model_name, loss_name, params, results, save_dir, scaler_type
):
    """Save trained model, scalers, config, and results."""

    logger.info("=" * 80)
    logger.info("SAVING MODEL AND ARTIFACTS")
    logger.info("=" * 80)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = save_path / "best_model.pt"
    torch.save(model.state_dict(), model_file)
    logger.info(f"Model saved: {model_file}")

    # Save scalers
    scalers_file = save_path / "scalers.pkl"
    with open(scalers_file, "wb") as f:
        pickle.dump(scalers, f)
    logger.info(f"Scalers saved: {scalers_file}")

    # Save configuration
    config = {
        "model_name": model_name,
        "loss_name": loss_name,
        "hyperparameters": params,
        "input_size": 12,
        "output_size": 48,
        "scaling_strategy": scaler_type,
        "dataset": "dataset_12CST_params.csv",
        "loss_type": "distance_weighted",
        "distance_weighting": "continuous (0-1)",
        "alpha_grid": "linspace(-20, 20, 48)",
        "timestamp": datetime.now().isoformat(),
    }

    config_file = save_path / "model_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved: {config_file}")

    # Save evaluation results
    results_file = save_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results["summary"], f, indent=2)
    logger.info(f"Results saved: {results_file}")

    logger.info(f"\nAll artifacts saved to: {save_dir}")


# ==================== Main Pipeline ====================


def main():
    """Main training and evaluation pipeline."""

    logger.info("=" * 80)
    logger.info("TRAIN AND EVALUATE BEST MODEL CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration (use organised/ root for published subset)
    RESULTS_DIR = "optuna_results/physics_aware"
    CSV_FILE = "data/csv/dataset_12CST_params.csv"
    SAVE_DIR = "trained_model"
    SCALER_TYPE = "auto"  # 'auto', 'standard', 'robust', or 'minmax'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Load best configuration
    model_name, loss_name, best_params, best_score = load_best_configuration(
        RESULTS_DIR
    )

    # 2. Load and prepare data with intelligent scaling
    train_loader, val_loader, test_loader, scalers, metadata, alpha_grid = (
        load_and_prepare_data(CSV_FILE, scaler_type=SCALER_TYPE)
    )

    # 3. Create model
    input_size = 12
    output_size = 48
    model = create_model(model_name, input_size, output_size, best_params, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel: {model_name}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # 4. Train model
    loss_fn = get_loss_fn(loss_name)
    (
        model,
        train_losses,
        val_losses,
        train_loss_history,
        val_physics_history,
        best_epoch,
    ) = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=best_params["num_epochs"],
        learning_rate=best_params["learning_rate"],
        loss_fn=loss_fn,
        device=device,
        weight_decay=best_params.get("weight_decay", 1e-5),
        y_scaler=scalers["y_scaler"],
        alpha_grid=alpha_grid,
    )

    # 5. Evaluate model
    results = evaluate_model(model, test_loader, scalers, alpha_grid, device)

    # 6. Create visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    viz_dir = Path(SAVE_DIR) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(train_loss_history, val_physics_history, best_epoch, viz_dir)
    plot_lift_curves(results, metadata["names_test"], alpha_grid, viz_dir, n_samples=6)
    plot_error_distributions(results, viz_dir)
    plot_prediction_scatter(results, viz_dir)
    plot_physics_metrics_comparison(results, viz_dir)
    plot_distance_quality_analysis(results, alpha_grid, viz_dir)

    logger.info(f"\nAll visualizations saved to: {viz_dir}")

    # 7. Save everything
    save_model_and_artifacts(
        model,
        scalers,
        model_name,
        loss_name,
        best_params,
        results,
        SAVE_DIR,
        scalers["scaler_type"],
    )

    # Final summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
