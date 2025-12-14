from datetime import datetime, timedelta
import os
import logging
# Torch is optional for many utility functions; import lazily when needed.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TORCH_AVAILABLE = False

import numpy as np
import pandas as pd
try:
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency
    stats = None
from typing import Callable, Optional, Dict, Any, Tuple, List

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import AeroSandbox (optional)
try:
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
except Exception:
    logger.warning(
        "AeroSandbox not available; Kulfan extraction functions will be disabled."
    )
    get_kulfan_parameters = None


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True
        # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False
    else:  # pragma: no cover - non-torch environments
        logger.debug("Torch not available; skipping torch-specific seed setup.")


def test_time(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Tuple[Any, timedelta]:
    """Run `func` with supplied args/kwargs and return (result, elapsed).

    Uses the module logger to report elapsed time (wall-clock).
    """
    start = datetime.now()
    result = func(*args, **kwargs)
    elapsed: timedelta = datetime.now() - start

    logger.info("Time taken for %s: %s", func.__name__, elapsed)

    return result, elapsed


# ---------------------- Shared Aero & Physics Utilities ----------------------
def build_cst_feature_columns(n_weights_per_side: int) -> List[str]:
    """Return canonical CST (Kulfan) feature column names for given number of weights.

    Format: 'lower_weight_{i}', 'upper_weight_{i}', 'TE_thickness', 'leading_edge_weight'
    """
    cols = []
    for i in range(n_weights_per_side):
        cols.append(f"lower_weight_{i}")
    for i in range(n_weights_per_side):
        cols.append(f"upper_weight_{i}")
    cols.extend(["TE_thickness", "leading_edge_weight"])
    return cols


def load_airfoil_coordinates(
    airfoil_name: str, data_dir: str = "data/aerofoil_data"
) -> Optional[np.ndarray]:
    """Load an airfoil coordinate file (.dat) returning Nx2 numpy array or None.

    Tries a few filename variants and logs warnings on failure.
    """
    possible_names = [
        f"{airfoil_name}.dat",
        f"{airfoil_name.lower()}.dat",
        f"{airfoil_name.upper()}.dat",
    ]
    for filename in possible_names:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                coords = np.loadtxt(filepath)
                if coords.ndim != 2 or coords.shape[1] != 2:
                    logger.warning(
                        f"Invalid coordinate shape for {filepath}: {coords.shape}"
                    )
                    continue
                return coords
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
                continue
    logger.error(f"Could not find coordinates for {airfoil_name} in {data_dir}")
    return None


def extract_kulfan_parameters(
    coordinates: np.ndarray, n_weights_per_side: int = 4
) -> Optional[Dict[str, float]]:
    """Convenience wrapper around AeroSandbox `get_kulfan_parameters`.

    Returns a flat dict of parameter names -> float.
    """
    try:
        kulfan_params = get_kulfan_parameters(
            coordinates=coordinates, n_weights_per_side=n_weights_per_side
        )
    except Exception as e:
        logger.error(f"Error extracting Kulfan parameters: {e}")
        return None

    params: Dict[str, float] = {}
    if "lower_weights" in kulfan_params:
        for i, w in enumerate(kulfan_params["lower_weights"]):
            params[f"lower_weight_{i}"] = float(w)
    if "upper_weights" in kulfan_params:
        for i, w in enumerate(kulfan_params["upper_weights"]):
            params[f"upper_weight_{i}"] = float(w)
    if "leading_edge_weight" in kulfan_params:
        params["leading_edge_weight"] = float(kulfan_params["leading_edge_weight"])
    te_key = (
        "TE_thickness"
        if "TE_thickness" in kulfan_params
        else (
            "trailing_edge_thickness"
            if "trailing_edge_thickness" in kulfan_params
            else None
        )
    )
    if te_key:
        params["TE_thickness"] = float(kulfan_params[te_key])

    return params


def extract_cst_params_from_coordinates(
    df: pd.DataFrame,
    n_weights_per_side: int,
    data_dir: str = "data/aerofoil_data",
    airfoil_name_col: str = "aerofoil_name",
) -> pd.DataFrame:
    """Extract CST parameters for each airfoil listed in `df[airfoil_name_col]`.

    Returns a new DataFrame with the original columns plus the extracted CST params.
    """
    rows = []
    failed = 0
    for _, row in df.iterrows():
        name = row[airfoil_name_col]
        coords = load_airfoil_coordinates(name, data_dir)
        if coords is None:
            failed += 1
            logger.warning(f"Skipping {name}: coordinates not found")
            continue
        params = extract_kulfan_parameters(coords, n_weights_per_side)
        if params is None:
            failed += 1
            logger.warning(f"Skipping {name}: could not extract Kulfan params")
            continue
        new_row = row.to_dict()
        new_row.update(params)
        rows.append(new_row)

    if failed > 0:
        logger.warning(f"Failed to extract CST params for {failed}/{len(df)} airfoils")

    return pd.DataFrame(rows)


def get_aerofoil_family(name: str) -> str:
    """Heuristic extraction of an airfoil family token from a name (LOFO grouping)."""
    n = str(name).lower().strip()
    if "naca" in n:
        return "NACA"
    if n.startswith("fx") or n[:2] == "fx":
        return "FX"
    if n.startswith("ag") and len(n) <= 5:
        return "AG"
    if n.startswith("e") and len(n) >= 3 and n[1:4].isdigit():
        return "Eppler"
    if n.startswith("s") and len(n) >= 4 and n[1:5].isdigit():
        return "Selig"
    if "clark" in n:
        return "Clark"
    if n.startswith("goe") or n.startswith("g0"):
        return "Gottingen"
    if n.startswith("s8") or "nrel" in n:
        return "NREL"
    if n.startswith("mh"):
        return "MH"
    if n.startswith("sd"):
        return "SD"
    if len(n) >= 3:
        return n[:3].upper()
    return "Misc"


def create_lofo_splits(
    df: pd.DataFrame, n_families: int = 5, min_samples: int = 50
) -> List[Tuple[str, List[int], List[int]]]:
    """Create Leave-One-Family-Out splits based on `family` column in df."""
    family_counts = df["family"].value_counts()
    valid_families = family_counts[family_counts >= min_samples].index.tolist()[
        :n_families
    ]
    splits: List[Tuple[str, List[int], List[int]]] = []
    for fam in valid_families:
        test_mask = df["family"] == fam
        test_idx = df[test_mask].index.tolist()
        train_idx = df[~test_mask].index.tolist()
        splits.append((fam, train_idx, test_idx))
    return splits


def extract_known_alpha_ranges(df_base: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Return mapping aerofoil_name -> (min_alpha, max_alpha) using alpha_0..alpha_47 columns."""
    alpha_cols = [c for c in df_base.columns if c.startswith("alpha_")]
    ranges: Dict[str, Tuple[float, float]] = {}
    for _, row in df_base.iterrows():
        name = row["aerofoil_name"]
        vals = row[alpha_cols].values.astype(float)
        ranges[name] = (float(np.min(vals)), float(np.max(vals)))
    return ranges


def calculate_physical_metrics(
    alpha_curve: np.ndarray, cl_curve: np.ndarray
) -> Dict[str, float]:
    """Compute cl_max, alpha_stall, alpha_zero_lift, lift curve slope and linear_r2."""
    metrics: Dict[str, float] = {}
    max_idx = int(np.argmax(cl_curve))
    metrics["cl_max"] = float(cl_curve[max_idx])
    metrics["alpha_stall"] = float(alpha_curve[max_idx])

    # Zero-lift angle
    sign_changes = np.where(np.diff(np.sign(cl_curve)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        a1, a2 = alpha_curve[idx], alpha_curve[idx + 1]
        c1, c2 = cl_curve[idx], cl_curve[idx + 1]
        if c2 != c1:
            metrics["alpha_zero_lift"] = float(a1 - c1 * (a2 - a1) / (c2 - c1))
        else:
            metrics["alpha_zero_lift"] = float(a1)
    else:
        metrics["alpha_zero_lift"] = float(alpha_curve[np.argmin(np.abs(cl_curve))])

    # Lift curve slope (linear fit around zero-lift)
    alpha_zero = metrics["alpha_zero_lift"]
    linear_mask = (alpha_curve >= alpha_zero - 5) & (alpha_curve <= alpha_zero + 5)
    pre_stall_mask = np.arange(len(alpha_curve)) < max_idx * 0.5
    mask = linear_mask & pre_stall_mask

    if np.sum(mask) >= 3:
        alpha_linear = alpha_curve[mask]
        cl_linear = cl_curve[mask]
        if len(np.unique(alpha_linear)) >= 2:
            try:
                slope, _, r_value, _, _ = stats.linregress(alpha_linear, cl_linear)
                metrics["lift_curve_slope"] = float(slope)
                metrics["linear_r2"] = float(r_value**2)
            except Exception:
                A = np.vstack([alpha_linear, np.ones(len(alpha_linear))]).T
                slope, _ = np.linalg.lstsq(A, cl_linear, rcond=None)[0]
                metrics["lift_curve_slope"] = float(slope)
                metrics["linear_r2"] = 0.0
        else:
            mid_idx = len(alpha_curve) // 4
            metrics["lift_curve_slope"] = float(
                np.gradient(cl_curve, alpha_curve)[mid_idx]
            )
            metrics["linear_r2"] = 0.0
    else:
        mid_idx = len(alpha_curve) // 3
        if mid_idx > 0 and mid_idx < len(alpha_curve) - 1:
            metrics["lift_curve_slope"] = float(
                (cl_curve[mid_idx + 1] - cl_curve[mid_idx - 1])
                / (alpha_curve[mid_idx + 1] - alpha_curve[mid_idx - 1])
            )
        else:
            metrics["lift_curve_slope"] = 0.0
        metrics["linear_r2"] = 0.0

    return metrics


def compare_physical_metrics(
    alpha: np.ndarray,
    cl_actual: np.ndarray,
    cl_pred: np.ndarray,
    alpha_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Return absolute errors between actual and predicted physical metrics."""
    if alpha_range is not None:
        min_a, max_a = alpha_range
        mask = (alpha >= min_a) & (alpha <= max_a)
        alpha_f, actual_f, pred_f = alpha[mask], cl_actual[mask], cl_pred[mask]
    else:
        alpha_f, actual_f, pred_f = alpha, cl_actual, cl_pred

    actual = calculate_physical_metrics(alpha_f, actual_f)
    pred = calculate_physical_metrics(alpha_f, pred_f)
    return {
        "cl_max_error": abs(actual["cl_max"] - pred["cl_max"]),
        "alpha_stall_error": abs(actual["alpha_stall"] - pred["alpha_stall"]),
        "alpha_zero_lift_error": abs(
            actual["alpha_zero_lift"] - pred["alpha_zero_lift"]
        ),
        "lift_curve_slope_error": abs(
            actual["lift_curve_slope"] - pred["lift_curve_slope"]
        ),
    }


def calculate_physics_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha_grid: np.ndarray,
    alpha_ranges: Optional[List[Tuple[float, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted physics score (lower is better) across samples."""
    if weights is None:
        weights = {
            "cl_max": 1.0,
            "alpha_stall": 0.3,
            "alpha_zero_lift": 0.5,
            "lift_curve_slope": 10.0,
        }

    errors = []
    for i in range(len(y_true)):
        alpha_range = alpha_ranges[i] if alpha_ranges is not None else None
        err = compare_physical_metrics(alpha_grid, y_true[i], y_pred[i], alpha_range)
        weighted = (
            weights["cl_max"] * err["cl_max_error"]
            + weights["alpha_stall"] * err["alpha_stall_error"]
            + weights["alpha_zero_lift"] * err["alpha_zero_lift_error"]
            + weights["lift_curve_slope"] * err["lift_curve_slope_error"]
        )
        errors.append(weighted)
    return float(np.mean(errors))


if TORCH_AVAILABLE:
    class QuickMLP(nn.Module):
        """Small MLP used in dataset optimization scripts.

        Kept lightweight and reproducible for quick comparisons.
        """

        def __init__(self, input_size: int, output_size: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
else:  # pragma: no cover - non-torch environments
    class QuickMLP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed; QuickMLP is unavailable.")


def train_quick_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: "torch.device",
    epochs: int = 30,
    use_masked_loss: bool = False,
) -> nn.Module:
    if not TORCH_AVAILABLE:  # pragma: no cover - non-torch environments
        raise RuntimeError("PyTorch must be installed to train models using train_quick_model.")
    """Train a `QuickMLP` (or similar) with optional element-wise masked MSE loss.

    The mask is expected to be provided as a third tensor in each batch when
    `use_masked_loss` is True.
    """
    criterion = nn.MSELoss(reduction="none") if use_masked_loss else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            # Support both (X, y) and (X, y, mask) batch formats
            if use_masked_loss:
                if len(batch) == 3:
                    X, y, mask = batch
                else:
                    X, y = batch
                    mask = torch.ones_like(y)
            else:
                X, y = batch
                mask = None

            X, y = X.to(device), y.to(device)
            if mask is not None:
                mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(X)

            if mask is not None:
                loss_tensor = criterion(outputs, y)
                masked = loss_tensor * mask
                mask_sum = torch.sum(mask)
                loss = (
                    (torch.sum(masked) / mask_sum)
                    if mask_sum > 0
                    else torch.sum(masked)
                )
            else:
                loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
    return model


def evaluate_quick_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: np.ndarray,
    device: "torch.device",
    alpha_ranges: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    if not TORCH_AVAILABLE:  # pragma: no cover - non-torch environments
        raise RuntimeError("PyTorch must be installed to evaluate models using evaluate_quick_model.")
    """Evaluate model using MSE and physics-aware metrics."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_t).cpu().numpy()

    mse = float(np.mean((y_test - y_pred) ** 2))
    physics_score = calculate_physics_score(y_test, y_pred, alpha_grid, alpha_ranges)

    phys_err = {
        "cl_max_error": [],
        "alpha_stall_error": [],
        "alpha_zero_lift_error": [],
        "lift_curve_slope_error": [],
    }
    for i in range(len(y_test)):
        a_range = alpha_ranges[i] if alpha_ranges is not None else None
        errs = compare_physical_metrics(alpha_grid, y_test[i], y_pred[i], a_range)
        for k in phys_err:
            phys_err[k].append(errs[k])

    return {
        "mse": mse,
        "physics_score": physics_score,
        "cl_max_error_mean": float(np.mean(phys_err["cl_max_error"])),
        "alpha_stall_error_mean": float(np.mean(phys_err["alpha_stall_error"])),
        "alpha_zero_lift_error_mean": float(np.mean(phys_err["alpha_zero_lift_error"])),
        "lift_curve_slope_error_mean": float(
            np.mean(phys_err["lift_curve_slope_error"])
        ),
    }
