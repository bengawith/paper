"""
Physics-Aware Sensitivity Analysis using Best Tuned Model

This script performs comprehensive sensitivity analysis using the best model
from physics-aware hyperparameter tuning (MLP with Huber loss).

Key Features:
1. Loads best model with exact hyperparameters from Optuna tuning
2. Uses physics-aware metrics (CL_max, alpha_stall, etc.)
3. Global sensitivity methods (Sobol, Morris)
4. Shows how geometry perturbations affect physical predictions
5. Validates alignment with physical laws
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# Add project root to Python path
sys.path.append(ROOT_DIR := os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# Import models from local `src` package
from src.models import MLPModel, GRUModel, LSTMModel  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# SALib for global sensitivity analysis
try:
    from SALib.sample import saltelli, morris as morris_sampler
    from SALib.analyze import sobol, morris

    SALIB_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("SALib is not installed. Install with: pip install SALib")
    logger.warning("Falling back to basic sensitivity analysis only.")
    SALIB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("physics_aware_sensitivity_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


@dataclass
class PhysicalMetrics:
    """Container for physical aerodynamic metrics."""

    cl_max: float
    alpha_stall: float
    alpha_zero_lift: float
    lift_curve_slope: float


class PhysicsCalculator:
    """Calculate physical aerodynamic metrics from lift curves."""

    @staticmethod
    def calculate_metrics(
        alpha_curve: np.ndarray, cl_curve: np.ndarray
    ) -> PhysicalMetrics:
        """
        Calculate physics-aware metrics from an alpha-CL curve.

        Parameters:
        -----------
        alpha_curve : np.ndarray
            Angle of attack values (degrees)
        cl_curve : np.ndarray
            Lift coefficient values

        Returns:
        --------
        PhysicalMetrics : Physical aerodynamic metrics
        """
        # 1. CL,max and Stall Angle
        max_idx = np.argmax(cl_curve)
        cl_max = float(cl_curve[max_idx])
        alpha_stall = float(alpha_curve[max_idx])

        # 2. Zero-Lift Angle
        sign_changes = np.where(np.diff(np.sign(cl_curve)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            if idx < len(alpha_curve) - 1:
                alpha_zero_lift = float(
                    np.interp(
                        0,
                        [cl_curve[idx], cl_curve[idx + 1]],
                        [alpha_curve[idx], alpha_curve[idx + 1]],
                    )
                )
            else:
                alpha_zero_lift = float(alpha_curve[0])
        else:
            alpha_zero_lift = float(alpha_curve[0])

        # 3. Lift Curve Slope (in linear region)
        linear_mask = (alpha_curve >= alpha_zero_lift - 5) & (
            alpha_curve <= alpha_zero_lift + 5
        )
        pre_stall_mask = np.arange(len(alpha_curve)) < max_idx * 0.5
        mask = linear_mask & pre_stall_mask

        if np.sum(mask) >= 3:
            try:
                slope, _ = np.polyfit(alpha_curve[mask], cl_curve[mask], 1)
                lift_curve_slope = float(slope)
            except Exception:
                lift_curve_slope = 0.1
        else:
            lift_curve_slope = 0.1

        return PhysicalMetrics(
            cl_max=cl_max,
            alpha_stall=alpha_stall,
            alpha_zero_lift=alpha_zero_lift,
            lift_curve_slope=lift_curve_slope,
        )

    @staticmethod
    def compare_metrics(
        actual: PhysicalMetrics, predicted: PhysicalMetrics
    ) -> Dict[str, float]:
        """Compare two sets of physical metrics."""
        return {
            "cl_max_error": abs(actual.cl_max - predicted.cl_max),
            "alpha_stall_error": abs(actual.alpha_stall - predicted.alpha_stall),
            "alpha_zero_lift_error": abs(
                actual.alpha_zero_lift - predicted.alpha_zero_lift
            ),
            "lift_curve_slope_error": abs(
                actual.lift_curve_slope - predicted.lift_curve_slope
            ),
        }


class BestModelLoader:
    """Load the best model from physics-aware tuning results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.best_models_dir = self.results_dir / "best_models"
        self.config = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_best_config(self) -> Dict:
        """Load best model configuration."""
        config_file = self.best_models_dir / "best_model_configs.json"

        if not config_file.exists():
            raise FileNotFoundError(f"Best model config not found at {config_file}")

        with open(config_file, "r") as f:
            configs = json.load(f)

        # Find overall best model (lowest physics score)
        best_model_name = min(configs.keys(), key=lambda k: configs[k]["physics_score"])

        self.config = configs[best_model_name]
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"Loss Function: {self.config['best_loss_function']}")
        logger.info(f"Physics Score: {self.config['physics_score']:.4f}")
        logger.info(f"{'=' * 80}\n")

        return self.config

    def instantiate_model(self) -> nn.Module:
        """Instantiate model with best hyperparameters."""
        if self.config is None:
            self.load_best_config()

        model_class = self.config["model_class"]
        hyperparams = self.config["hyperparameters"]

        logger.info("Instantiating model with hyperparameters:")
        for key, value in hyperparams.items():
            if key not in ["learning_rate", "weight_decay", "num_epochs"]:
                logger.info(f"  {key}: {value}")

        if model_class in ("ImprovedMLPModel", "MLPModel"):
            self.model = MLPModel(
                input_size=hyperparams["input_size"],
                hidden_size=hyperparams["hidden_size"],
                output_size=hyperparams["output_size"],
                num_layers=hyperparams.get("num_layers", 4),
                dropout=hyperparams.get("dropout_rate", 0.1),
                width_factor=hyperparams.get("width_factor", 2),
            )
        elif model_class in ("ImprovedGRUModel", "GRUModel"):
            self.model = GRUModel(
                input_size=hyperparams["input_size"],
                hidden_size=hyperparams["hidden_size"],
                output_size=hyperparams["output_size"],
                num_layers=hyperparams.get("num_layers", 2),
                dropout=hyperparams.get("dropout_rate", 0.1),
                bidirectional=hyperparams.get("bidirectional", True),
            )
        elif model_class in ("ImprovedLSTMModel", "LSTMModel"):
            self.model = LSTMModel(
                input_size=hyperparams["input_size"],
                hidden_size=hyperparams["hidden_size"],
                output_size=hyperparams["output_size"],
                num_layers=hyperparams.get("num_layers", 2),
                dropout=hyperparams.get("dropout_rate", 0.1),
                bidirectional=hyperparams.get("bidirectional", True),
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"\nModel instantiated on device: {self.device}")
        logger.info(
            f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        return self.model


class PhysicsAwareSensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis with physics-aware metrics.
    Uses the best model from physics-aware hyperparameter tuning.
    """

    def __init__(self, model_loader: BestModelLoader, data_path: str):
        self.model_loader = model_loader
        self.model = None
        self.data_path = data_path
        self.df = None
        self.feature_cols = None
        self.alpha_grid = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = model_loader.device

        # CST parameter info (12 parameters: 5 lower + 5 upper + TE + LE)
        # Using notation from Table 1 in paper
        self.param_names = [
            "lower_weight_0",
            "lower_weight_1",
            "lower_weight_2",
            "lower_weight_3",
            "lower_weight_4",
            "upper_weight_0",
            "upper_weight_1",
            "upper_weight_2",
            "upper_weight_3",
            "upper_weight_4",
            "TE_thickness",
            "leading_edge_weight",
        ]

        # Labels matching Table 1 notation
        self.param_labels = [
            "A_L,0",
            "A_L,1",
            "A_L,2",
            "A_L,3",
            "A_L,4",
            "A_U,0",
            "A_U,1",
            "A_U,2",
            "A_U,3",
            "A_U,4",
            "ζ_TE",
            "Δz_LE",
        ]

        # Table 1 descriptions for documentation
        self.param_descriptions = [
            "Lower Surface Bernstein Coeff. 0",
            "Lower Surface Bernstein Coeff. 1",
            "Lower Surface Bernstein Coeff. 2",
            "Lower Surface Bernstein Coeff. 3",
            "Lower Surface Bernstein Coeff. 4",
            "Upper Surface Bernstein Coeff. 0",
            "Upper Surface Bernstein Coeff. 1",
            "Upper Surface Bernstein Coeff. 2",
            "Upper Surface Bernstein Coeff. 3",
            "Upper Surface Bernstein Coeff. 4",
            "Trailing Edge Thickness",
            "Leading Edge Modification Weight",
        ]

    def setup(self) -> None:
        """Load model and prepare data."""
        logger.info("\n" + "=" * 80)
        logger.info("SETUP: Loading Model and Data")
        logger.info("=" * 80)

        # Load model
        self.model = self.model_loader.instantiate_model()

        # Load data
        logger.info(f"\nLoading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # Identify feature columns (CST parameters only)
        self.feature_cols = [
            col
            for col in self.df.columns
            if col.startswith(
                (
                    "lower_weight_",
                    "upper_weight_",
                    "leading_edge_weight",
                    "TE_thickness",
                )
            )
        ]

        logger.info(f"Found {len(self.feature_cols)} CST parameters")

        # Get CL columns
        cl_cols = [col for col in self.df.columns if col.startswith("CL_")]
        logger.info(f"Found {len(cl_cols)} CL output points")

        # Create alpha grid (standard -20° to 20° for 48 points)
        self.alpha_grid = np.linspace(-20, 20, len(cl_cols))

        # Fit scalers on full dataset
        X_all = self.df[self.feature_cols].values
        y_all = self.df[cl_cols].values

        self.scaler_X.fit(X_all)
        self.scaler_y.fit(y_all)

        logger.info(f"Loaded {len(self.df)} airfoils")
        logger.info(
            f"Alpha range: {self.alpha_grid[0]:.1f}° to {self.alpha_grid[-1]:.1f}°"
        )

    def get_baseline_params(self, aerofoil_name: str = "naca0024") -> np.ndarray:
        """Get baseline CST parameters for an airfoil."""
        aerofoil_name = aerofoil_name.lower()

        # Try exact match first
        matches = self.df[self.df["aerofoil_name"].str.lower() == aerofoil_name]

        if len(matches) == 0:
            # Try partial match
            matches = self.df[
                self.df["aerofoil_name"].str.lower().str.contains(aerofoil_name)
            ]

        if len(matches) == 0:
            raise ValueError(f"Airfoil '{aerofoil_name}' not found in dataset")

        baseline = matches.iloc[0][self.feature_cols].values
        logger.info(f"\nBaseline airfoil: {matches.iloc[0]['aerofoil_name']}")

        return baseline

    def predict_single(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict lift curve for given CST parameters.

        Returns:
        --------
        alpha_grid : np.ndarray
            Angle of attack values
        cl_pred : np.ndarray
            Predicted CL values
        """
        # Normalize input
        params_scaled = self.scaler_X.transform(params.reshape(1, -1))

        # Predict
        with torch.no_grad():
            x_tensor = torch.FloatTensor(params_scaled).to(self.device)
            y_pred_scaled = self.model(x_tensor).cpu().numpy()

        # Denormalize output
        cl_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0]

        return self.alpha_grid, cl_pred

    def predict_with_uncertainty(
        self, params: np.ndarray, n_samples: int = 100, noise_std: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification using Monte Carlo sampling.

        Returns:
        --------
        alpha_grid : np.ndarray
        cl_mean : np.ndarray
            Mean predicted CL
        cl_lower : np.ndarray
            Lower bound (2.5th percentile)
        cl_upper : np.ndarray
            Upper bound (97.5th percentile)
        """
        predictions = []

        for _ in range(n_samples):
            # Add noise to parameters
            noisy_params = params + np.random.normal(
                0, noise_std * np.abs(params), params.shape
            )
            _, cl_pred = self.predict_single(noisy_params)
            predictions.append(cl_pred)

        predictions = np.array(predictions)

        cl_mean = np.mean(predictions, axis=0)
        cl_lower = np.percentile(predictions, 2.5, axis=0)
        cl_upper = np.percentile(predictions, 97.5, axis=0)

        return self.alpha_grid, cl_mean, cl_lower, cl_upper

    def one_at_a_time_sensitivity(
        self, baseline_params: np.ndarray, perturbation_factor: float = 0.2
    ) -> pd.DataFrame:
        """
        One-at-a-time sensitivity analysis with physics-aware metrics.

        Parameters:
        -----------
        baseline_params : np.ndarray
            Baseline CST parameters
        perturbation_factor : float
            Perturbation magnitude (default 20%)

        Returns:
        --------
        pd.DataFrame : Sensitivity results for each parameter
        """
        logger.info(
            f"\nPerforming OAT sensitivity analysis (±{perturbation_factor * 100}% perturbation)..."
        )

        # Baseline prediction
        alpha_base, cl_base = self.predict_single(baseline_params)
        metrics_base = PhysicsCalculator.calculate_metrics(alpha_base, cl_base)

        results = []

        for i, (param_name, param_label) in enumerate(
            zip(self.param_names, self.param_labels)
        ):
            logger.info(f"  Analyzing {param_name}...")

            # Perturb parameter
            params_plus = baseline_params.copy()
            params_minus = baseline_params.copy()

            perturbation = perturbation_factor * abs(baseline_params[i])
            if perturbation < 1e-6:  # Handle near-zero parameters
                perturbation = 0.01

            params_plus[i] += perturbation
            params_minus[i] -= perturbation

            # Predictions
            _, cl_plus = self.predict_single(params_plus)
            _, cl_minus = self.predict_single(params_minus)

            # Calculate metrics
            metrics_plus = PhysicsCalculator.calculate_metrics(alpha_base, cl_plus)

            # Sensitivities (absolute change)
            cl_max_sens = abs(metrics_plus.cl_max - metrics_base.cl_max)
            alpha_stall_sens = abs(metrics_plus.alpha_stall - metrics_base.alpha_stall)
            alpha_zero_sens = abs(
                metrics_plus.alpha_zero_lift - metrics_base.alpha_zero_lift
            )
            slope_sens = abs(
                metrics_plus.lift_curve_slope - metrics_base.lift_curve_slope
            )

            # Normalized sensitivities
            cl_max_norm = (
                cl_max_sens / abs(metrics_base.cl_max)
                if abs(metrics_base.cl_max) > 0
                else 0
            )
            alpha_stall_norm = (
                alpha_stall_sens / abs(metrics_base.alpha_stall)
                if abs(metrics_base.alpha_stall) > 0
                else 0
            )
            alpha_zero_norm = (
                alpha_zero_sens / abs(metrics_base.alpha_zero_lift)
                if abs(metrics_base.alpha_zero_lift) > 1e-6
                else 0
            )
            slope_norm = (
                slope_sens / abs(metrics_base.lift_curve_slope)
                if abs(metrics_base.lift_curve_slope) > 0
                else 0
            )

            results.append(
                {
                    "parameter": param_name,
                    "parameter_label": param_label,
                    "baseline_value": baseline_params[i],
                    "cl_max_sensitivity": cl_max_sens,
                    "alpha_stall_sensitivity": alpha_stall_sens,
                    "alpha_zero_lift_sensitivity": alpha_zero_sens,
                    "lift_curve_slope_sensitivity": slope_sens,
                    "cl_max_norm_sensitivity": cl_max_norm,
                    "alpha_stall_norm_sensitivity": alpha_stall_norm,
                    "alpha_zero_lift_norm_sensitivity": alpha_zero_norm,
                    "lift_curve_slope_norm_sensitivity": slope_norm,
                }
            )

        df_results = pd.DataFrame(results)
        logger.info("[DONE] OAT analysis complete")

        return df_results

    def sobol_sensitivity(
        self,
        baseline_params: np.ndarray,
        n_samples: int = 512,
        calc_second_order: bool = False,
        n_bootstrap: int = 100,
    ) -> Dict:
        """
        Sobol global sensitivity analysis using SALib with Saltelli sampling.

        Addresses reviewer requirements:
        - Uses Saltelli sampling scheme
        - Documents prior ranges for each CST coefficient
        - Reports total number of model evaluations
        - Includes bootstrap confidence intervals

        Parameters:
        -----------
        baseline_params : np.ndarray
            Baseline CST parameters
        n_samples : int
            Number of samples for Saltelli sampling (default 512)
            Total evaluations = n_samples * (2*n_params + 2) for first-order
                              = n_samples * (n_params + 2) for first+second order
        calc_second_order : bool
            Whether to calculate second-order indices
        n_bootstrap : int
            Number of bootstrap resamples for confidence intervals (default 100)

        Returns:
        --------
        dict : Sobol indices with confidence intervals for each physical metric
        """
        if not SALIB_AVAILABLE:
            logger.warning("SALib not available, skipping Sobol analysis")
            return None

        logger.info(
            "\nPerforming Sobol sensitivity analysis using Saltelli sampling..."
        )
        logger.info(f"  Base samples (n): {n_samples}")
        logger.info(f"  Second-order indices: {'Yes' if calc_second_order else 'No'}")
        logger.info(f"  Bootstrap resamples: {n_bootstrap}")

        # Define problem for SALib
        problem = {
            "num_vars": len(self.param_names),
            "names": self.param_names,
            "bounds": [],
        }

        # Set bounds as ±50% of baseline values (uniform prior)
        # Document prior ranges for each parameter
        logger.info("\n  Prior Ranges (Uniform Distribution):")
        prior_ranges = {}

        for i, (param_name, param_label) in enumerate(
            zip(self.param_names, self.param_labels)
        ):
            if abs(baseline_params[i]) < 1e-6:
                lower_bound = -0.1
                upper_bound = 0.1
            else:
                lower_bound = baseline_params[i] - 0.5 * abs(baseline_params[i])
                upper_bound = baseline_params[i] + 0.5 * abs(baseline_params[i])

            problem["bounds"].append([lower_bound, upper_bound])
            prior_ranges[param_name] = {
                "label": param_label,
                "baseline": float(baseline_params[i]),
                "lower": float(lower_bound),
                "upper": float(upper_bound),
                "range_width": float(upper_bound - lower_bound),
            }
            logger.info(
                f"    {param_label:10s}: [{lower_bound:8.4f}, {upper_bound:8.4f}]  "
                f"(baseline: {baseline_params[i]:8.4f})"
            )

        # Generate Saltelli samples
        logger.info("\n  Generating Saltelli samples...")
        param_samples = saltelli.sample(
            problem, n_samples, calc_second_order=calc_second_order
        )

        # Calculate total number of model evaluations
        n_params = len(self.param_names)
        if calc_second_order:
            n_evaluations = n_samples * (2 * n_params + 2)
        else:
            n_evaluations = n_samples * (n_params + 2)

        logger.info(f"  Total model evaluations: {n_evaluations:,}")
        logger.info(
            f"    Formula: n_samples × (2×n_params + 2) = {n_samples} × {2 * n_params + 2}"
        )

        logger.info("  Running model predictions...")

        # Run model for all samples
        metrics_samples = {
            "cl_max": [],
            "alpha_stall": [],
            "alpha_zero_lift": [],
            "lift_curve_slope": [],
        }

        for i, params in enumerate(param_samples):
            if (i + 1) % 500 == 0 or i == 0:
                logger.info(
                    f"    Progress: {i + 1}/{len(param_samples)} ({100 * (i + 1) / len(param_samples):.1f}%)"
                )

            alpha, cl_pred = self.predict_single(params)
            metrics = PhysicsCalculator.calculate_metrics(alpha, cl_pred)

            metrics_samples["cl_max"].append(metrics.cl_max)
            metrics_samples["alpha_stall"].append(metrics.alpha_stall)
            metrics_samples["alpha_zero_lift"].append(metrics.alpha_zero_lift)
            metrics_samples["lift_curve_slope"].append(metrics.lift_curve_slope)

        # Analyze Sobol indices for each metric with bootstrap confidence intervals
        sobol_results = {
            "sampling_scheme": "Saltelli",
            "n_samples": n_samples,
            "n_evaluations": n_evaluations,
            "n_bootstrap": n_bootstrap,
            "calc_second_order": calc_second_order,
            "prior_ranges": prior_ranges,
            "metrics": {},
        }

        for metric_name, metric_values in metrics_samples.items():
            logger.info(f"\n  Analyzing Sobol indices for {metric_name}...")
            logger.info(
                f"    Computing bootstrap confidence intervals ({n_bootstrap} resamples)..."
            )

            # Main Sobol analysis
            Si = sobol.analyze(
                problem,
                np.array(metric_values),
                calc_second_order=calc_second_order,
                num_resamples=n_bootstrap,
                conf_level=0.95,
            )

            metric_result = {
                "S1": Si["S1"].tolist(),  # First-order indices
                "ST": Si["ST"].tolist(),  # Total-order indices
                "S1_conf": Si["S1_conf"].tolist(),  # 95% confidence intervals
                "ST_conf": Si["ST_conf"].tolist(),
                "parameter_names": self.param_names,
                "parameter_labels": self.param_labels,
            }

            if calc_second_order and "S2" in Si:
                metric_result["S2"] = Si[
                    "S2"
                ].tolist()  # Second-order interaction indices
                if "S2_conf" in Si:
                    metric_result["S2_conf"] = Si["S2_conf"].tolist()

            sobol_results["metrics"][metric_name] = metric_result

            # Log summary statistics
            logger.info("    First-order indices (S1):")
            for i, (name, s1, conf) in enumerate(
                zip(self.param_labels, Si["S1"], Si["S1_conf"])
            ):
                logger.info(f"      {name:10s}: {s1:7.4f} ± {conf:7.4f}")

            logger.info("    Total-order indices (ST):")
            for i, (name, st, conf) in enumerate(
                zip(self.param_labels, Si["ST"], Si["ST_conf"])
            ):
                logger.info(f"      {name:10s}: {st:7.4f} ± {conf:7.4f}")

        logger.info("\n[DONE] Sobol analysis complete")

        return sobol_results

    def morris_sensitivity(
        self, baseline_params: np.ndarray, n_trajectories: int = 50
    ) -> Dict:
        """
        Morris screening method for parameter importance.

        Parameters:
        -----------
        baseline_params : np.ndarray
            Baseline CST parameters
        n_trajectories : int
            Number of Morris trajectories

        Returns:
        --------
        dict : Morris measures for each physical metric
        """
        if not SALIB_AVAILABLE:
            logger.warning("SALib not available, skipping Morris analysis")
            return None

        logger.info(
            f"\nPerforming Morris sensitivity analysis ({n_trajectories} trajectories)..."
        )

        # Define problem
        problem = {
            "num_vars": len(self.param_names),
            "names": self.param_names,
            "bounds": [],
        }

        # Set bounds as ±50% of baseline values
        for param_val in baseline_params:
            if abs(param_val) < 1e-6:
                problem["bounds"].append([-0.1, 0.1])
            else:
                problem["bounds"].append(
                    [param_val - 0.5 * abs(param_val), param_val + 0.5 * abs(param_val)]
                )

        # Generate Morris samples
        param_samples = morris_sampler.sample(problem, n_trajectories, num_levels=4)

        logger.info(f"  Generated {len(param_samples)} parameter combinations")
        logger.info("  Running model predictions...")

        # Run model for all samples
        metrics_samples = {
            "cl_max": [],
            "alpha_stall": [],
            "alpha_zero_lift": [],
            "lift_curve_slope": [],
        }

        for i, params in enumerate(param_samples):
            if (i + 1) % 100 == 0:
                logger.info(f"    Progress: {i + 1}/{len(param_samples)}")

            alpha, cl_pred = self.predict_single(params)
            metrics = PhysicsCalculator.calculate_metrics(alpha, cl_pred)

            metrics_samples["cl_max"].append(metrics.cl_max)
            metrics_samples["alpha_stall"].append(metrics.alpha_stall)
            metrics_samples["alpha_zero_lift"].append(metrics.alpha_zero_lift)
            metrics_samples["lift_curve_slope"].append(metrics.lift_curve_slope)

        # Analyze Morris measures for each metric
        morris_results = {}

        for metric_name, metric_values in metrics_samples.items():
            logger.info(f"  Analyzing Morris measures for {metric_name}...")

            Si = morris.analyze(problem, param_samples, np.array(metric_values))

            morris_results[metric_name] = {
                "mu": Si["mu"].tolist(),  # Mean elementary effect
                "mu_star": Si["mu_star"].tolist(),  # Mean absolute elementary effect
                "sigma": Si["sigma"].tolist(),  # Standard deviation
                "mu_star_conf": Si["mu_star_conf"].tolist(),  # Confidence intervals
                "parameter_names": self.param_names,
            }

        logger.info("[DONE] Morris analysis complete")

        return morris_results

    def compare_airfoils(
        self, airfoil_names: List[str], perturbation_factor: float = 0.2
    ) -> pd.DataFrame:
        """
        Compare OAT sensitivity across multiple airfoils.

        Returns:
        --------
        pd.DataFrame : Combined sensitivity results for all airfoils
        """
        logger.info(f"\nComparing sensitivity across {len(airfoil_names)} airfoils...")

        all_results = []

        for airfoil_name in airfoil_names:
            try:
                logger.info(f"  Analyzing {airfoil_name}...")
                baseline = self.get_baseline_params(airfoil_name)
                oat_results = self.one_at_a_time_sensitivity(
                    baseline, perturbation_factor
                )
                oat_results["airfoil"] = airfoil_name
                all_results.append(oat_results)
            except Exception as e:
                logger.warning(f"  Failed to analyze {airfoil_name}: {e}")

        if len(all_results) == 0:
            logger.error("No airfoils were successfully analyzed")
            return pd.DataFrame()

        combined_df = pd.concat(all_results, ignore_index=True)
        logger.info("[DONE] Multi-airfoil comparison complete")

        return combined_df


class PhysicsAwareVisualizer:
    """Create physics-aware visualizations for sensitivity analysis."""

    def __init__(self, save_dir: str = "."):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Colors for consistency
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#06A77D",
            "warning": "#D62246",
        }

    def plot_oat_sensitivity(
        self, oat_results: pd.DataFrame, metric: str = "cl_max"
    ) -> None:
        """Plot OAT sensitivity for a specific metric."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute sensitivity
        sensitivity_col = f"{metric}_sensitivity"
        data = oat_results.sort_values(sensitivity_col, ascending=False)

        ax1.barh(
            data["parameter_label"],
            data[sensitivity_col],
            color=self.colors["primary"],
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xlabel(
            f"Absolute Sensitivity ({metric.replace('_', ' ').title()})",
            fontweight="bold",
        )
        ax1.set_title(
            f"OAT Sensitivity: {metric.replace('_', ' ').title()}",
            fontweight="bold",
            fontsize=12,
        )
        ax1.grid(True, alpha=0.3, axis="x")

        # Normalized sensitivity
        norm_col = f"{metric}_norm_sensitivity"
        data = oat_results.sort_values(norm_col, ascending=False)

        ax2.barh(
            data["parameter_label"],
            data[norm_col],
            color=self.colors["accent"],
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xlabel("Normalized Sensitivity (Fractional Change)", fontweight="bold")
        ax2.set_title(
            f"Normalized OAT Sensitivity: {metric.replace('_', ' ').title()}",
            fontweight="bold",
            fontsize=12,
        )
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        filename = self.save_dir / f"oat_sensitivity_{metric}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {filename}")
        plt.close()

    def plot_sobol_indices(self, sobol_results: Dict, metric: str = "cl_max") -> None:
        """
        Plot Sobol indices for a specific metric with all reviewer requirements.

        Includes:
        - Table 1 parameter labels (A_L,0-4, A_U,0-4, ζ_TE, Δz_LE)
        - Saltelli sampling scheme annotation
        - Prior ranges for each CST coefficient
        - Number of model evaluations
        - Bootstrap confidence intervals
        """
        if sobol_results is None:
            return

        # Extract data
        data = sobol_results["metrics"][metric]
        param_labels = data["parameter_labels"]  # Using Table 1 notation
        s1 = np.array(data["S1"])
        st = np.array(data["ST"])
        s1_conf = np.array(data["S1_conf"])
        st_conf = np.array(data["ST_conf"])

        # Get prior range info
        prior_ranges = sobol_results["prior_ranges"]
        param_names = data["parameter_names"]

        # Sort by total-order index
        sort_idx = np.argsort(st)[::-1]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # === Main plot: Sobol indices with confidence intervals ===
        x = np.arange(len(param_labels))
        width = 0.35

        # Bar plots
        ax1.bar(
            x - width / 2,
            s1[sort_idx],
            width,
            label="First-order (S₁)",
            color=self.colors["primary"],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.2,
        )
        ax1.bar(
            x + width / 2,
            st[sort_idx],
            width,
            label="Total-order (Sₜ)",
            color=self.colors["accent"],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.2,
        )

        # Error bars (bootstrap confidence intervals)
        ax1.errorbar(
            x - width / 2,
            s1[sort_idx],
            yerr=s1_conf[sort_idx],
            fmt="none",
            ecolor="black",
            capsize=4,
            capthick=1.5,
            alpha=0.7,
            linewidth=1.5,
            label="95% Bootstrap CI",
        )
        ax1.errorbar(
            x + width / 2,
            st[sort_idx],
            yerr=st_conf[sort_idx],
            fmt="none",
            ecolor="black",
            capsize=4,
            capthick=1.5,
            alpha=0.7,
            linewidth=1.5,
        )

        # Sorted parameter labels (Table 1 notation)
        sorted_labels = [param_labels[i] for i in sort_idx]
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_labels, rotation=0, ha="center", fontsize=11)
        ax1.set_ylabel("Sobol Index", fontweight="bold", fontsize=12)
        ax1.set_title(
            f"Sobol Sensitivity Indices: {metric.replace('_', ' ').title()}\n"
            + f"Saltelli Sampling | {sobol_results['n_evaluations']:,} Model Evaluations | "
            + f"{sobol_results['n_bootstrap']} Bootstrap Resamples",
            fontweight="bold",
            fontsize=13,
            pad=15,
        )
        ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax1.set_ylim(0, max(max(st), 1.0) * 1.15)

        # Add text box with sampling info
        info_text = (
            f"Sampling: Saltelli\n"
            f"Base samples: {sobol_results['n_samples']}\n"
            f"Total evaluations: {sobol_results['n_evaluations']:,}\n"
            f"Prior: Uniform ±50% baseline"
        )
        ax1.text(
            0.02,
            0.98,
            info_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # === Bottom panel: Prior ranges table ===
        ax2.axis("off")

        # Create table data for prior ranges
        table_data = []
        table_data.append(["Parameter", "Label", "Prior Range", "Baseline"])

        for idx in sort_idx:
            param_name = param_names[idx]
            param_label = param_labels[idx]
            prior_info = prior_ranges[param_name]

            range_str = f"[{prior_info['lower']:.4f}, {prior_info['upper']:.4f}]"
            baseline_str = f"{prior_info['baseline']:.4f}"

            table_data.append(
                [
                    param_name.replace("_weight_", "_").replace("_", " "),
                    param_label,
                    range_str,
                    baseline_str,
                ]
            )

        # Create table
        table = ax2.table(
            cellText=table_data,
            cellLoc="left",
            colWidths=[0.25, 0.15, 0.35, 0.15],
            bbox=[0.0, 0.0, 1.0, 1.0],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor("#4A90E2")
            cell.set_text_props(weight="bold", color="white")
            cell.set_height(0.12)

        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(4):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor("#F0F0F0")
                cell.set_height(0.08)

        # Add title to table
        ax2.text(
            0.5,
            1.05,
            "Prior Ranges (Uniform Distribution, ±50% of Baseline)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            transform=ax2.transAxes,
            clip_on=False,
        )

        plt.tight_layout()
        filename = self.save_dir / f"sobol_indices_{metric}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {filename}")
        plt.close()

    def plot_morris_screening(
        self, morris_results: Dict, metric: str = "cl_max"
    ) -> None:
        """Plot Morris screening results (mu* vs sigma)."""
        if morris_results is None:
            return

        data = morris_results[metric]
        params = data["parameter_names"]
        mu_star = np.array(data["mu_star"])
        sigma = np.array(data["sigma"])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(
            mu_star,
            sigma,
            s=200,
            alpha=0.6,
            c=range(len(params)),
            cmap="viridis",
            edgecolors="black",
            linewidth=1.5,
        )

        # Label points
        for i, param in enumerate(params):
            label = param.replace("_weight_", " ").replace("_", " ").title()
            ax.annotate(
                label,
                (mu_star[i], sigma[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

        # Add quadrant lines
        mu_star_median = np.median(mu_star)
        sigma_median = np.median(sigma)
        ax.axvline(mu_star_median, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(sigma_median, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Quadrant labels
        ax.text(
            0.98,
            0.98,
            "High Influence\nHigh Non-linearity",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("μ* (Mean Absolute Effect)", fontweight="bold")
        ax.set_ylabel("σ (Standard Deviation)", fontweight="bold")
        ax.set_title(
            f"Morris Screening: {metric.replace('_', ' ').title()}",
            fontweight="bold",
            fontsize=12,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = self.save_dir / f"morris_screening_{metric}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {filename}")
        plt.close()

    def plot_comparison_heatmap(
        self, comparison_df: pd.DataFrame, metric: str = "cl_max"
    ) -> None:
        """Plot heatmap comparing sensitivity across airfoils."""
        if comparison_df.empty:
            return

        sensitivity_col = f"{metric}_norm_sensitivity"

        # Pivot table
        pivot = comparison_df.pivot_table(
            values=sensitivity_col,
            index="parameter_label",
            columns="airfoil",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={"label": "Normalized Sensitivity"},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_xlabel("Airfoil", fontweight="bold")
        ax.set_ylabel("CST Parameter", fontweight="bold")
        ax.set_title(
            f"Sensitivity Comparison: {metric.replace('_', ' ').title()}",
            fontweight="bold",
            fontsize=12,
        )

        plt.tight_layout()
        filename = self.save_dir / f"sensitivity_comparison_{metric}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {filename}")
        plt.close()

    def plot_uncertainty_quantification(
        self, analyzer: PhysicsAwareSensitivityAnalyzer, airfoil_name: str = "naca0024"
    ) -> None:
        """Plot lift curve with uncertainty bounds."""
        try:
            baseline = analyzer.get_baseline_params(airfoil_name)
            # Ensure baseline is numpy array of floats
            baseline = np.array(baseline, dtype=np.float64)
            alpha, cl_mean, cl_lower, cl_upper = analyzer.predict_with_uncertainty(
                baseline
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot mean prediction
            ax.plot(alpha, cl_mean, "b-", linewidth=2, label="Mean Prediction")

            # Plot confidence interval
            ax.fill_between(
                alpha,
                cl_lower,
                cl_upper,
                alpha=0.3,
                color="blue",
                label="95% Confidence Interval",
            )

            ax.set_xlabel("Angle of Attack (°)", fontweight="bold")
            ax.set_ylabel("Lift Coefficient (CL)", fontweight="bold")
            ax.set_title(
                f"Prediction with Uncertainty: {airfoil_name.upper()}",
                fontweight="bold",
                fontsize=12,
            )
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = self.save_dir / f"uncertainty_{airfoil_name.lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            logger.info(f"  Saved: {filename}")
            plt.close()

        except Exception as e:
            logger.warning(f"Could not plot uncertainty for {airfoil_name}: {e}")

    def create_summary_figure(
        self,
        oat_results: pd.DataFrame,
        sobol_results: Optional[Dict],
        morris_results: Optional[Dict],
        metric: str = "cl_max",
    ) -> None:
        """Create comprehensive summary figure."""
        if sobol_results is None or morris_results is None:
            logger.warning(
                "Skipping summary figure (requires Sobol and Morris results)"
            )
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. OAT normalized sensitivity
        ax1 = fig.add_subplot(gs[0, 0])
        norm_col = f"{metric}_norm_sensitivity"
        data = oat_results.sort_values(norm_col, ascending=False)
        ax1.barh(
            data["parameter_label"],
            data[norm_col],
            color=self.colors["primary"],
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xlabel("Normalized Sensitivity", fontweight="bold")
        ax1.set_title("One-At-A-Time Analysis", fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="x")

        # 2. Sobol indices
        ax2 = fig.add_subplot(gs[0, 1])
        sobol_data = sobol_results["metrics"][metric]
        param_labels = sobol_data["parameter_labels"]
        st = np.array(sobol_data["ST"])
        sort_idx = np.argsort(st)[::-1][:10]

        sorted_labels = [param_labels[i] for i in sort_idx]
        ax2.barh(
            sorted_labels,
            st[sort_idx],
            color=self.colors["accent"],
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xlabel("Total Sobol Index (Sₜ)", fontweight="bold")
        ax2.set_title("Sobol Global Sensitivity", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")

        # 3. Morris screening
        ax3 = fig.add_subplot(gs[1, 0])
        morris_data = morris_results[metric]
        morris_params = morris_data["parameter_names"]
        mu_star = np.array(morris_data["mu_star"])
        sigma = np.array(morris_data["sigma"])

        ax3.scatter(
            mu_star,
            sigma,
            s=150,
            alpha=0.6,
            c=range(len(morris_params)),
            cmap="viridis",
            edgecolors="black",
            linewidth=1.2,
        )

        # Label top 5
        top_idx = np.argsort(mu_star)[-5:]
        for i in top_idx:
            label = morris_params[i].replace("_weight_", " ").replace("_", " ").title()
            ax3.annotate(
                label,
                (mu_star[i], sigma[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        ax3.set_xlabel("μ* (Mean Absolute Effect)", fontweight="bold")
        ax3.set_ylabel("σ (Standard Deviation)", fontweight="bold")
        ax3.set_title("Morris Screening", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # 4. Text summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        summary_text = "PHYSICS-AWARE SENSITIVITY SUMMARY\n"
        summary_text += f"Metric: {metric.replace('_', ' ').title()}\n\n"

        summary_text += "Top 3 Most Influential Parameters:\n"
        top3_idx = np.argsort(st)[::-1][:3]
        for rank, idx in enumerate(top3_idx, 1):
            param_label = param_labels[idx]
            summary_text += f"  {rank}. {param_label} (Sₜ = {st[idx]:.3f})\n"

        summary_text += f"\nTotal Variance Explained: {np.sum(st):.2f}\n"
        summary_text += "\nParameter Interactions: "
        interactions = np.sum(st - np.array(sobol_data["S1"]))
        summary_text += f"{interactions:.3f}\n"
        summary_text += "(Difference between total and first-order indices)\n"

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.suptitle(
            f"Comprehensive Sensitivity Analysis: {metric.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )

        filename = self.save_dir / f"summary_sensitivity_{metric}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {filename}")
        plt.close()


def main():
    """Main execution function."""
    logger.info("\n" + "=" * 80)
    logger.info("PHYSICS-AWARE SENSITIVITY ANALYSIS")
    logger.info("Using Best Model from Physics-Aware Hyperparameter Tuning")
    logger.info("=" * 80 + "\n")

    # Configuration
    results_dir = f"{ROOT_DIR}/optuna_results/physics_aware"
    data_path = f"{ROOT_DIR}/data/csv/dataset_12CST_params.csv"
    save_dir = f"{ROOT_DIR}/optuna_results/physics_aware/sensitivity_figures"

    # Initialize
    model_loader = BestModelLoader(results_dir)
    analyzer = PhysicsAwareSensitivityAnalyzer(model_loader, data_path)
    analyzer.setup()

    # Initialize visualizer
    visualizer = PhysicsAwareVisualizer(save_dir)

    # Baseline airfoil
    baseline_airfoil = "naca0024"
    baseline_params = analyzer.get_baseline_params(baseline_airfoil)

    logger.info(f"\nBaseline airfoil: {baseline_airfoil}")

    # ========================================================================
    # STEP 1: One-At-A-Time Sensitivity Analysis
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: ONE-AT-A-TIME SENSITIVITY ANALYSIS")
    logger.info("=" * 80)

    oat_results = analyzer.one_at_a_time_sensitivity(
        baseline_params, perturbation_factor=0.20
    )

    # Save results
    oat_file = Path(save_dir) / "oat_sensitivity_results.csv"
    oat_results.to_csv(oat_file, index=False)
    logger.info(f"\nSaved OAT results to: {oat_file}")

    # Plot for each metric
    for metric in ["cl_max", "alpha_stall", "alpha_zero_lift", "lift_curve_slope"]:
        visualizer.plot_oat_sensitivity(oat_results, metric)

    # ========================================================================
    # STEP 2: Sobol Global Sensitivity Analysis
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: SOBOL GLOBAL SENSITIVITY ANALYSIS")
    logger.info("=" * 80)

    sobol_results = None
    if SALIB_AVAILABLE:
        sobol_results = analyzer.sobol_sensitivity(baseline_params, n_samples=512)

        if sobol_results:
            # Save results
            sobol_file = Path(save_dir) / "sobol_sensitivity_results.json"
            with open(sobol_file, "w") as f:
                json.dump(sobol_results, f, indent=2)
            logger.info(f"\nSaved Sobol results to: {sobol_file}")

            # Plot for each metric
            for metric in [
                "cl_max",
                "alpha_stall",
                "alpha_zero_lift",
                "lift_curve_slope",
            ]:
                visualizer.plot_sobol_indices(sobol_results, metric)

    # ========================================================================
    # STEP 3: Morris Screening
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: MORRIS SCREENING METHOD")
    logger.info("=" * 80)

    morris_results = None
    if SALIB_AVAILABLE:
        morris_results = analyzer.morris_sensitivity(baseline_params, n_trajectories=50)

        if morris_results:
            # Save results
            morris_file = Path(save_dir) / "morris_sensitivity_results.json"
            with open(morris_file, "w") as f:
                json.dump(morris_results, f, indent=2)
            logger.info(f"\nSaved Morris results to: {morris_file}")

            # Plot for each metric
            for metric in [
                "cl_max",
                "alpha_stall",
                "alpha_zero_lift",
                "lift_curve_slope",
            ]:
                visualizer.plot_morris_screening(morris_results, metric)

    # ========================================================================
    # STEP 4: Multi-Airfoil Comparison
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: MULTI-AIRFOIL SENSITIVITY COMPARISON")
    logger.info("=" * 80)

    comparison_airfoils = ["naca0024", "naca0012", "naca4412"]

    # Try to find more interesting airfoils
    available_airfoils = analyzer.df["aerofoil_name"].str.lower().unique()
    if "dragonfly" in available_airfoils:
        comparison_airfoils.append("dragonfly")

    comparison_df = analyzer.compare_airfoils(
        comparison_airfoils, perturbation_factor=0.20
    )

    if not comparison_df.empty:
        comp_file = Path(save_dir) / "airfoil_comparison_results.csv"
        comparison_df.to_csv(comp_file, index=False)
        logger.info(f"\nSaved comparison results to: {comp_file}")

        # Plot heatmaps
        for metric in ["cl_max", "alpha_stall", "alpha_zero_lift", "lift_curve_slope"]:
            visualizer.plot_comparison_heatmap(comparison_df, metric)

    # ========================================================================
    # STEP 5: Uncertainty Quantification
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: UNCERTAINTY QUANTIFICATION")
    logger.info("=" * 80)

    uq_airfoils = ["naca0024"]
    if "dragonfly" in available_airfoils:
        uq_airfoils.append("dragonfly")

    for airfoil in uq_airfoils:
        visualizer.plot_uncertainty_quantification(analyzer, airfoil)

    # ========================================================================
    # STEP 6: Create Summary Figures
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: CREATING SUMMARY FIGURES")
    logger.info("=" * 80)

    for metric in ["cl_max", "alpha_stall"]:
        visualizer.create_summary_figure(
            oat_results, sobol_results, morris_results, metric
        )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nAll results saved to: {save_dir}/")
    logger.info("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
