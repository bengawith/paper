"""
physics_aware_tuner.py

Physics-Aware Hyperparameter Tuning for Aerofoil Lift Prediction

This script uses Optuna to tune MLP, GRU, LSTM, and CNN models with multiple loss functions,
optimizing for physics-aware metrics (CL_max error, stall angle error, etc.) rather than
just MSE. The best models are saved with comprehensive evaluation metrics.
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
import optuna
import logging
import json
from datetime import datetime
from typing import List, Tuple

# Import models and utilities from `src`
from src.models import GRUModel, LSTMModel, MLPModel, CNNModel, create_optimizer

USE_IMPROVED_MODELS = True
logging.info("Using models from `src.models`")

# Try to import utility functions
try:
    from src.utils import set_seed
except ImportError:

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# Import shared utils from the organised package
try:
    from src.utils import (
        compare_physical_metrics,
        calculate_physics_score,
        get_aerofoil_family,
        create_lofo_splits,
    )
except Exception as e:
    logging.warning(
        f"Could not import organised.src.utils: {e}. Falling back to local implementations if present."
    )


# Set random seed for reproducibility
set_seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("physics_aware_tuning.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    csv_file: str,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], bool]:
    """
    Load CSV data and prepare it for training.

    Parameters:
    -----------
    csv_file : str
        Path to CSV file

    Returns:
    --------
    df : DataFrame with features, CL values, and family
    alpha_grid : standardized alpha grid
    feature_cols : list of feature column names (CST parameters only)
    has_mask : bool indicating if dataset has validity/distance masks
    """
    logger.info(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)

    # Add family column if not present
    if "family" not in df.columns:
        df["family"] = df["aerofoil_name"].apply(get_aerofoil_family)

    # Identify feature columns
    feature_cols = [
        col
        for col in df.columns
        if col.startswith(
            ("lower_weight_", "upper_weight_", "leading_edge_weight", "TE_thickness")
        )
    ]

    has_mask: bool = "dist_0" in df.columns

    # Create standardized alpha grid (assuming 48 CL points)
    alpha_grid = np.linspace(-20, 20, 48)

    logger.info(
        f"Loaded {len(df)} samples with {len(feature_cols)} CST parameter features"
    )
    logger.info(f"Found {df['family'].nunique()} aerofoil families")

    return df, alpha_grid, feature_cols, has_mask


# ==================== Model Definitions ====================

if not USE_IMPROVED_MODELS:
    # Fallback: Simple models if src2 is not available
    class SimpleMLP(nn.Module):
        """Multi-Layer Perceptron for regression."""

        def __init__(
            self,
            input_size: int,
            hidden_sizes: List[int],
            output_size: int,
            dropout_rate: float = 0.2,
        ):
            super(SimpleMLP, self).__init__()

            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, output_size))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

else:
    # Use improved models from src2
    SimpleMLP = None  # Will use MLPModel instead


if not USE_IMPROVED_MODELS:

    class SimpleGRU(nn.Module):
        """GRU Recurrent Neural Network for regression."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 2,
            dropout: float = 0.2,
        ):
            super(SimpleGRU, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.gru = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.gru(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

else:
    SimpleGRU = None  # Will use GRUModel instead


if not USE_IMPROVED_MODELS:

    class SimpleLSTM(nn.Module):
        """LSTM Recurrent Neural Network for regression."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 2,
            dropout: float = 0.2,
        ):
            super(SimpleLSTM, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

else:
    SimpleLSTM = None  # Will use LSTMModel instead
# ==================== Loss Functions ====================


def get_loss_fn(loss_name: str):
    """Get loss function by name."""
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "huber":
        return nn.HuberLoss(delta=1.0)
    elif loss_name == "log_cosh":

        def log_cosh_loss(y_pred, y_true):
            diff = y_pred - y_true
            return torch.log(
                torch.cosh(diff + 1e-12)
            )  # Element-wise for distance weighting

        return log_cosh_loss
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ==================== Training Functions ====================


def create_data_loaders(
    X, y, train_idx, test_idx, df, batch_size, use_masked_loss=False
):
    """
    Create train and test data loaders with optional masks.

    Parameters:
    -----------
    X : np.ndarray
        Feature array
    y : np.ndarray
        Target array (CL values)
    train_idx : np.ndarray
        Training indices
    test_idx : np.ndarray
        Test indices
    df : pd.DataFrame
        Original dataframe (for extracting masks)
    batch_size : int
        Batch size
    use_masked_loss : bool
        If True, includes masks in the data loaders

    Returns:
    --------
    train_loader : DataLoader
    test_loader : DataLoader
    """
    if use_masked_loss:
        if "dist_0" in df.columns:
            # Solution: Distance-based mask
            mask_cols = [f"dist_{i}" for i in range(48)]
            mask_train = df.loc[train_idx, mask_cols].values.astype(np.float32)
            mask_test = df.loc[test_idx, mask_cols].values.astype(np.float32)
        else:
            # Fallback: no mask available
            mask_train = np.ones_like(y[train_idx])
            mask_test = np.ones_like(y[test_idx])

        # Create datasets with masks
        train_dataset = TensorDataset(
            torch.FloatTensor(X[train_idx]),
            torch.FloatTensor(y[train_idx]),
            torch.FloatTensor(mask_train),
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X[test_idx]),
            torch.FloatTensor(y[test_idx]),
            torch.FloatTensor(mask_test),
        )
    else:
        # Standard datasets without masks
        train_dataset = TensorDataset(
            torch.FloatTensor(X[train_idx]), torch.FloatTensor(y[train_idx])
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X[test_idx]), torch.FloatTensor(y[test_idx])
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, loss_fn, device, use_masked_loss=False):
    """
    Train for one epoch.

    Parameters:
    -----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data (yields X, y, mask if use_masked_loss=True)
    optimizer : torch.optim.Optimizer
        Optimizer
    loss_fn : callable or nn.Module
        Loss function (must support reduction='none' if use_masked_loss=True)
    device : torch.device
        Device to train on
    use_masked_loss : bool, default=False
        If True, expects mask in train_loader and applies masked loss
    """
    model.train()
    total_loss = 0.0

    if use_masked_loss:
        # Element-wise loss for masking
        if isinstance(loss_fn, nn.MSELoss):
            criterion = nn.MSELoss(reduction="none")
        elif isinstance(loss_fn, nn.HuberLoss):
            criterion = nn.HuberLoss(reduction="none")
        elif isinstance(loss_fn, nn.L1Loss):
            criterion = nn.L1Loss(reduction="none")
        elif isinstance(loss_fn, nn.SmoothL1Loss):
            criterion = nn.SmoothL1Loss(reduction="none")
        elif callable(loss_fn):
            # Custom loss - assume it supports element-wise computation
            criterion = loss_fn
        else:
            criterion = loss_fn

        for batch_data in train_loader:
            if len(batch_data) == 3:
                X_batch, y_batch, mask_batch = batch_data
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
            else:
                # Fallback if mask not provided
                X_batch, y_batch = batch_data
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = torch.ones_like(y_batch).to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)

            if isinstance(
                criterion, (nn.MSELoss, nn.HuberLoss, nn.L1Loss, nn.SmoothL1Loss)
            ):
                loss_tensor = criterion(y_pred, y_batch)
            else:
                diff = y_pred - y_batch
                loss_tensor = torch.log(torch.cosh(diff + 1e-12))

            weighted_loss = loss_tensor * mask_batch

            batch_weight = mask_batch.sum()
            if batch_weight > 0:
                loss = weighted_loss.sum() / batch_weight
            else:
                loss = weighted_loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    else:
        # Standard training without masking
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)

            if callable(loss_fn):
                loss = loss_fn(y_pred, y_batch)
            else:
                loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, alpha_grid, device):
    """
    Evaluate model on test set with physics-aware metrics.

    Returns:
    --------
    dict : Evaluation metrics (MSE, physics errors)
    """
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                X_batch, y_batch, _ = batch_data
                X_batch, y_batch = batch_data

            X_batch = X_batch.to(device)
            y_pred = model(X_batch)

            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_predictions)

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)

    physics_score = calculate_physics_score(y_true, y_pred, alpha_grid)

    # Individual physics metrics
    physics_errors = {
        "cl_max_error": [],
        "alpha_stall_error": [],
        "alpha_zero_lift_error": [],
        "lift_curve_slope_error": [],
    }

    for i in range(len(y_true)):
        errors = compare_physical_metrics(alpha_grid, y_true[i], y_pred[i])
        for key in physics_errors:
            physics_errors[key].append(errors[key])

    results = {
        "mse": float(mse),
        "physics_score": float(physics_score),
        "cl_max_error_mean": float(np.mean(physics_errors["cl_max_error"])),
        "cl_max_error_std": float(np.std(physics_errors["cl_max_error"])),
        "alpha_stall_error_mean": float(np.mean(physics_errors["alpha_stall_error"])),
        "alpha_stall_error_std": float(np.std(physics_errors["alpha_stall_error"])),
        "alpha_zero_lift_error_mean": float(
            np.mean(physics_errors["alpha_zero_lift_error"])
        ),
        "alpha_zero_lift_error_std": float(
            np.std(physics_errors["alpha_zero_lift_error"])
        ),
        "lift_curve_slope_error_mean": float(
            np.mean(physics_errors["lift_curve_slope_error"])
        ),
        "lift_curve_slope_error_std": float(
            np.std(physics_errors["lift_curve_slope_error"])
        ),
    }

    return results


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    alpha_grid,
    num_epochs,
    learning_rate,
    loss_fn,
    device,
    patience=20,
    optimizer=None,
    use_masked_loss=False,
):
    """
    Train model and evaluate with early stopping.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer, optional
        Pre-configured optimizer. If None, creates Adam optimizer with learning_rate.
    use_masked_loss : bool, default=False
        If True, uses masked loss training

    Returns:
    --------
    dict : Final evaluation metrics
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_physics_score = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        _ = train_epoch(
            model, train_loader, optimizer, loss_fn, device, use_masked_loss
        )

        # Evaluate every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            eval_results = evaluate_model(model, test_loader, alpha_grid, device)
            physics_score = eval_results["physics_score"]

            # Early stopping based on physics score
            if physics_score < best_physics_score:
                best_physics_score = physics_score
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience // 10:  # Check every 10 epochs
                logger.debug(f"Early stopping at epoch {epoch + 1}")
                break

    # Final evaluation
    final_results = evaluate_model(model, test_loader, alpha_grid, device)

    return final_results


# ==================== Optuna Objective Functions ====================


def objective_mlp(
    trial,
    input_size,
    output_size,
    train_loader,
    test_loader,
    alpha_grid,
    device,
    loss_name,
    use_masked_loss=False,
):
    """Objective function for MLP hyperparameter tuning."""

    if USE_IMPROVED_MODELS:
        # Use improved MLP with wide-deep architecture and residual connections
        hidden_size = trial.suggest_int("hidden_size", 128, 512, step=64)
        num_layers = trial.suggest_int("num_layers", 3, 6)
        width_factor = trial.suggest_int("width_factor", 2, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 300)

        # Create improved model
        model = MLPModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            width_factor=width_factor,
            dropout=dropout_rate,
        ).to(device)

        # Get loss function
        loss_fn = get_loss_fn(loss_name)

        # Create optimizer with proper weight decay handling
        optimizer = create_optimizer(model, "adamw", learning_rate, weight_decay)

        # Train and evaluate with custom optimizer
        results = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            alpha_grid,
            num_epochs,
            learning_rate,
            loss_fn,
            device,
            optimizer=optimizer,
            use_masked_loss=use_masked_loss,
        )
    else:
        # Fallback to simple MLP
        num_layers = trial.suggest_int("num_layers", 2, 6)
        hidden_sizes = [
            trial.suggest_int(f"hidden_size_{i}", 64, 512, step=64)
            for i in range(num_layers)
        ]
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 300)

        # Create model
        model = SimpleMLP(input_size, hidden_sizes, output_size, dropout_rate).to(
            device
        )

        # Get loss function
        loss_fn = get_loss_fn(loss_name)

        # Train and evaluate
        results = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            alpha_grid,
            num_epochs,
            learning_rate,
            loss_fn,
            device,
            use_masked_loss=use_masked_loss,
        )

    # Return physics score (primary objective)
    return results["physics_score"]


def objective_gru(
    trial,
    input_size,
    output_size,
    train_loader,
    test_loader,
    alpha_grid,
    device,
    loss_name,
    use_masked_loss=False,
):
    """Objective function for GRU hyperparameter tuning."""

    if USE_IMPROVED_MODELS:
        # Use improved GRU with bidirectional processing and attention
        hidden_size = trial.suggest_int("hidden_size", 128, 512, step=64)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 300)

        # Create improved model
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        ).to(device)

        # Get loss function
        loss_fn = get_loss_fn(loss_name)

        # Create optimizer with proper weight decay handling
        optimizer = create_optimizer(model, "adamw", learning_rate, weight_decay)

        # Train and evaluate with custom optimizer
        results = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            alpha_grid,
            num_epochs,
            learning_rate,
            loss_fn,
            device,
            optimizer=optimizer,
            use_masked_loss=use_masked_loss,
        )
    else:
        # Fallback to simple GRU
        hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 300)

        # Create model
        model = SimpleGRU(
            input_size, hidden_size, output_size, num_layers, dropout_rate
        ).to(device)

        # Get loss function
        loss_fn = get_loss_fn(loss_name)

        # Train and evaluate
        results = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            alpha_grid,
            num_epochs,
            learning_rate,
            loss_fn,
            device,
            use_masked_loss=use_masked_loss,
        )

    return results["physics_score"]


def objective_lstm(
    trial,
    input_size,
    output_size,
    train_loader,
    test_loader,
    alpha_grid,
    device,
    loss_name,
    use_masked_loss=False,
):
    """Objective function for LSTM hyperparameter tuning."""

    if USE_IMPROVED_MODELS:
        # Use improved LSTM with bidirectional processing and attention
        hidden_size = trial.suggest_int("hidden_size", 128, 512, step=64)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 300)

        # Create improved model
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        ).to(device)

        # Get loss function
        loss_fn = get_loss_fn(loss_name)

        # Create optimizer with proper weight decay handling
        optimizer = create_optimizer(model, "adamw", learning_rate, weight_decay)

        # Train and evaluate with custom optimizer
        results = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            alpha_grid,
            num_epochs,
            learning_rate,
            loss_fn,
            device,
            optimizer=optimizer,
            use_masked_loss=use_masked_loss,
        )
    else:
        # Fallback to simple LSTM
        hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 300)

        # Create model
        model = SimpleLSTM(
            input_size, hidden_size, output_size, num_layers, dropout_rate
        ).to(device)

        # Get loss function
        loss_fn = get_loss_fn(loss_name)

        # Train and evaluate
        results = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            alpha_grid,
            num_epochs,
            learning_rate,
            loss_fn,
            device,
            use_masked_loss=use_masked_loss,
        )

    return results["physics_score"]


def objective_cnn(
    trial,
    input_size,
    output_size,
    train_loader,
    test_loader,
    alpha_grid,
    device,
    loss_name,
    use_masked_loss=False,
):
    """Objective function for 1D CNN hyperparameter tuning."""

    if USE_IMPROVED_MODELS:
        # Use improved 1D CNN with residual connections and SE blocks
        num_filter_blocks = trial.suggest_int("num_filter_blocks", 2, 4)
        base_filters = trial.suggest_int("base_filters", 32, 128, step=16)

        # Create filter progression (e.g., [32, 64, 128])
        num_filters = [base_filters * (2**i) for i in range(num_filter_blocks)]

        # Kernel sizes for each block
        kernel_sizes = [
            trial.suggest_int(f"kernel_size_{i}", 3, 7, step=2)
            for i in range(num_filter_blocks)
        ]

        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_epochs = trial.suggest_int("num_epochs", 100, 300)

        # Create improved 1D CNN model
        model = CNNModel(
            input_size=input_size,
            output_size=output_size,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout_rate,
        ).to(device)

        # Get loss function
        loss_fn = get_loss_fn(loss_name)

        # Create optimizer with proper weight decay handling
        optimizer = create_optimizer(model, "adamw", learning_rate, weight_decay)

        # Train and evaluate with custom optimizer
        results = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            alpha_grid,
            num_epochs,
            learning_rate,
            loss_fn,
            device,
            optimizer=optimizer,
            use_masked_loss=use_masked_loss,
        )
    else:
        # Skip CNN if improved models not available
        logger.warning("Improved models not available - skipping CNN")
        return float("inf")

    return results["physics_score"]


# ==================== Main Tuning Pipeline ====================


def main(n_trials: int = 250):
    """Main tuning pipeline."""

    # Configuration
    CSV_FILE = "data/csv/dataset_12CST_params.csv"
    RESULTS_DIR = "optuna_results/physics_aware"
    NUM_TRIALS = n_trials  # Trials per model-loss combination
    BATCH_SIZE = 32
    USE_DISTANCE_FEATURES = False  # Set to False for production-ready models

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    df, alpha_grid, feature_cols, has_mask = load_and_prepare_data(
        CSV_FILE, use_distance_features=USE_DISTANCE_FEATURES
    )

    # Determine if we should use masked loss training
    USE_MASKED_LOSS = has_mask
    if USE_MASKED_LOSS:
        logger.info(
            "✓ Masked loss training ENABLED (dataset has validity/distance masks)"
        )
    else:
        logger.info("  Masked loss training DISABLED (no masks in dataset)")

    # Get CL columns
    cl_cols = [f"CL_{i}" for i in range(48)]

    # Prepare features and targets
    X = df[feature_cols].values.astype(np.float32)
    y = df[cl_cols].values.astype(np.float32)

    input_size = len(feature_cols)
    output_size = len(cl_cols)

    logger.info(f"Input size: {input_size}, Output size: {output_size}")

    # Create LOFO splits for robust evaluation
    lofo_splits = create_lofo_splits(df, n_families=5, min_samples=50)

    # Model architectures
    model_types = {
        "MLP": objective_mlp,
        "GRU": objective_gru,
        "LSTM": objective_lstm,
        "CNN": objective_cnn,
    }

    # Loss functions
    loss_functions = ["mse", "huber", "log_cosh", "smooth_l1"]

    # Store all results
    all_results = {}

    # Main tuning loop
    for model_name, objective_fn in model_types.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Tuning {model_name}")
        logger.info(f"{'=' * 60}")

        all_results[model_name] = {}

        for loss_name in loss_functions:
            logger.info(f"\n  Loss function: {loss_name}")

            all_results[model_name][loss_name] = {
                "lofo_results": [],
                "best_params": None,
                "best_score": float("inf"),
            }

            # Use first LOFO fold for hyperparameter tuning
            family, train_idx, test_idx = lofo_splits[0]

            # Create data loaders with optional masks
            train_loader, test_loader = create_data_loaders(
                X, y, train_idx, test_idx, df, BATCH_SIZE, USE_MASKED_LOSS
            )

            # Create Optuna study
            study = optuna.create_study(
                direction="minimize",
                study_name=f"{model_name}_{loss_name}",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            )

            # Optimize
            logger.info(f"    Starting optimization ({NUM_TRIALS} trials)...")
            study.optimize(
                lambda trial: objective_fn(
                    trial,
                    input_size,
                    output_size,
                    train_loader,
                    test_loader,
                    alpha_grid,
                    device,
                    loss_name,
                    USE_MASKED_LOSS,
                ),
                n_trials=NUM_TRIALS,
                show_progress_bar=True,
            )

            best_params = study.best_params
            best_score = study.best_value

            logger.info(f"    Best physics score: {best_score:.6f}")
            logger.info(f"    Best params: {best_params}")

            all_results[model_name][loss_name]["best_params"] = best_params
            all_results[model_name][loss_name]["best_score"] = best_score

            # Evaluate on all LOFO folds with best hyperparameters
            logger.info(f"    Evaluating on all {len(lofo_splits)} LOFO folds...")

            for fold_idx, (family, train_idx, test_idx) in enumerate(lofo_splits):
                # Create data loaders for this fold with optional masks
                train_loader, test_loader = create_data_loaders(
                    X, y, train_idx, test_idx, df, BATCH_SIZE, USE_MASKED_LOSS
                )

                # Create model with best hyperparameters
                if USE_IMPROVED_MODELS:
                    # Use improved models
                    if model_name == "MLP":
                        model = MLPModel(
                            input_size=input_size,
                            hidden_size=best_params["hidden_size"],
                            output_size=output_size,
                            num_layers=best_params["num_layers"],
                            width_factor=best_params["width_factor"],
                            dropout=best_params["dropout_rate"],
                        ).to(device)
                    elif model_name == "GRU":
                        model = GRUModel(
                            input_size=input_size,
                            hidden_size=best_params["hidden_size"],
                            output_size=output_size,
                            num_layers=best_params["num_layers"],
                            bidirectional=best_params["bidirectional"],
                            dropout=best_params["dropout_rate"],
                        ).to(device)
                    elif model_name == "LSTM":
                        model = LSTMModel(
                            input_size=input_size,
                            hidden_size=best_params["hidden_size"],
                            output_size=output_size,
                            num_layers=best_params["num_layers"],
                            bidirectional=best_params["bidirectional"],
                            dropout=best_params["dropout_rate"],
                        ).to(device)
                    elif model_name == "CNN":
                        num_filter_blocks = best_params["num_filter_blocks"]
                        base_filters = best_params["base_filters"]
                        num_filters = [
                            base_filters * (2**i) for i in range(num_filter_blocks)
                        ]
                        kernel_sizes = [
                            best_params[f"kernel_size_{i}"]
                            for i in range(num_filter_blocks)
                        ]

                        model = CNNModel(
                            input_size=input_size,
                            output_size=output_size,
                            num_filters=num_filters,
                            kernel_sizes=kernel_sizes,
                            dropout=best_params["dropout_rate"],
                        ).to(device)
                else:
                    # Use simple models
                    if model_name == "MLP":
                        num_layers = best_params["num_layers"]
                        hidden_sizes = [
                            best_params[f"hidden_size_{i}"] for i in range(num_layers)
                        ]
                        model = SimpleMLP(
                            input_size,
                            hidden_sizes,
                            output_size,
                            best_params["dropout_rate"],
                        ).to(device)
                    elif model_name == "GRU":
                        model = SimpleGRU(
                            input_size,
                            best_params["hidden_size"],
                            output_size,
                            best_params["num_layers"],
                            best_params["dropout_rate"],
                        ).to(device)
                    elif model_name == "LSTM":
                        model = SimpleLSTM(
                            input_size,
                            best_params["hidden_size"],
                            output_size,
                            best_params["num_layers"],
                            best_params["dropout_rate"],
                        ).to(device)

                # Train and evaluate
                loss_fn = get_loss_fn(loss_name)

                # Create optimizer for improved models
                if USE_IMPROVED_MODELS:
                    optimizer = create_optimizer(
                        model,
                        "adamw",
                        best_params["learning_rate"],
                        best_params["weight_decay"],
                    )
                    results = train_and_evaluate(
                        model,
                        train_loader,
                        test_loader,
                        alpha_grid,
                        best_params["num_epochs"],
                        best_params["learning_rate"],
                        loss_fn,
                        device,
                        optimizer=optimizer,
                        use_masked_loss=USE_MASKED_LOSS,
                    )
                else:
                    results = train_and_evaluate(
                        model,
                        train_loader,
                        test_loader,
                        alpha_grid,
                        best_params["num_epochs"],
                        best_params["learning_rate"],
                        loss_fn,
                        device,
                        use_masked_loss=USE_MASKED_LOSS,
                    )

                results["family"] = family
                results["n_test"] = len(test_idx)

                all_results[model_name][loss_name]["lofo_results"].append(results)

                logger.info(
                    f"      Fold {fold_idx + 1} ({family}): Physics={results['physics_score']:.4f}, "
                    f"MSE={results['mse']:.6f}"
                )

            # Save results for this model-loss combination
            result_file = os.path.join(
                RESULTS_DIR, f"{model_name}_{loss_name}_results.json"
            )
            with open(result_file, "w") as f:
                json.dump(all_results[model_name][loss_name], f, indent=2)

            logger.info(f"    Results saved to {result_file}")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = os.path.join(
        RESULTS_DIR, f"comprehensive_results_{timestamp}.json"
    )

    with open(final_results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("Tuning complete! Comprehensive results saved to:")
    logger.info(f"{final_results_file}")
    logger.info(f"{'=' * 60}")

    # Summary report
    logger.info("%s", "\n" + "=" * 80)
    logger.info("PHYSICS-AWARE TUNING RESULTS SUMMARY")
    logger.info("%s", "=" * 80)

    for model_name in model_types.keys():
        logger.info("%s", f"\n{model_name}:")
        for loss_name in loss_functions:
            best_score = all_results[model_name][loss_name]["best_score"]

            # Calculate average LOFO performance
            lofo_results = all_results[model_name][loss_name]["lofo_results"]
            avg_physics = np.mean([r["physics_score"] for r in lofo_results])
            avg_mse = np.mean([r["mse"] for r in lofo_results])
            avg_cl_max_err = np.mean([r["cl_max_error_mean"] for r in lofo_results])
            avg_stall_err = np.mean([r["alpha_stall_error_mean"] for r in lofo_results])

            logger.info(
                "  %s: Physics=%.4f, MSE=%.6f, ΔCL_max=%.4f, Δα_stall=%.2f°",
                loss_name,
                avg_physics,
                avg_mse,
                avg_cl_max_err,
                avg_stall_err,
            )

    logger.info("%s", "\n" + "=" * 80)

    # Find overall best model
    best_overall = None
    best_overall_score = float("inf")

    for model_name in model_types.keys():
        for loss_name in loss_functions:
            lofo_results = all_results[model_name][loss_name]["lofo_results"]
            avg_physics = np.mean([r["physics_score"] for r in lofo_results])

            if avg_physics < best_overall_score:
                best_overall_score = avg_physics
                best_overall = (model_name, loss_name)

    logger.info("🏆 BEST MODEL: %s with %s loss", best_overall[0], best_overall[1])
    logger.info("Average Physics Score: %.4f", best_overall_score)
    logger.info("%s", "=" * 80 + "\n")


if __name__ == "__main__":
    main()
