"""
gather_model_details.py

Comprehensive script to gather model complexity, training details, and benchmarking
information for the paper. This addresses the reviewer's requests for:
1. Parameter counts for all models
2. Non-neural baseline comparison (XGBoost)
3. Hardware and software specifications
4. Validation strategy details
5. Inference timing benchmarks
"""

import os
import sys

sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
import platform
from datetime import datetime
from typing import Dict, List, Tuple
import psutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import model classes from local `src` package
from src.models import GRUModel, LSTMModel, MLPModel, CNNModel  # noqa: E402

IMPROVED_MODELS_AVAILABLE = True

# Try importing XGBoost for baseline
try:
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")


# ==================== Parameter Counting ====================


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and non-trainable parameters in a PyTorch model.

    Returns:
    --------
    dict: Dictionary with total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


def analyze_model_architecture(
    model: nn.Module, model_name: str, hyperparams: Dict
) -> Dict:
    """
    Comprehensive analysis of a model's architecture.
    """
    param_counts = count_parameters(model)
    model_size = get_model_size_mb(model)

    analysis = {
        "model_name": model_name,
        "hyperparameters": hyperparams,
        "total_parameters": param_counts["total_parameters"],
        "trainable_parameters": param_counts["trainable_parameters"],
        "non_trainable_parameters": param_counts["non_trainable_parameters"],
        "model_size_mb": round(model_size, 4),
        "parameter_breakdown": {},
    }

    # Layer-by-layer breakdown
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                analysis["parameter_breakdown"][name] = module_params

    return analysis


# ==================== Hardware & Software Info ====================


def get_system_info() -> Dict:
    """
    Gather comprehensive system and software information.
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "os": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # GPU information
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                f"gpu_{i}_name": torch.cuda.get_device_name(i),
                f"gpu_{i}_memory_total_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                ),
                f"gpu_{i}_compute_capability": f"{torch.cuda.get_device_properties(i).major}."
                f"{torch.cuda.get_device_properties(i).minor}",
            }
            info.update(gpu_info)

    # Software versions
    try:
        import scipy

        info["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    try:
        import pandas

        info["pandas_version"] = pandas.__version__
    except ImportError:
        pass

    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    if XGBOOST_AVAILABLE:
        info["xgboost_version"] = xgb.__version__

    return info


# ==================== Inference Benchmarking ====================


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    n_runs: int = 1000,
    warmup_runs: int = 100,
    batch_sizes: List[int] = [1, 8, 32, 64],
) -> Dict:
    """
    Benchmark model inference time across different batch sizes.

    Parameters:
    -----------
    model : nn.Module
        Model to benchmark
    input_shape : tuple
        Shape of a single input sample (without batch dimension)
    device : torch.device
        Device to run on
    n_runs : int
        Number of inference runs per batch size
    warmup_runs : int
        Number of warmup runs to exclude from timing
    batch_sizes : list
        Batch sizes to test

    Returns:
    --------
    dict : Timing statistics for each batch size
    """
    model.eval()
    model.to(device)

    results = {}

    for batch_size in batch_sizes:
        # Create dummy input
        if len(input_shape) == 1:
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
        else:
            dummy_input = torch.randn(batch_size, *input_shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        # Synchronize for accurate timing on GPU
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy_input)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)

        # Calculate per-sample time
        per_sample_times = times / batch_size

        results[f"batch_{batch_size}"] = {
            "mean_total_ms": float(np.mean(times)),
            "std_total_ms": float(np.std(times)),
            "min_total_ms": float(np.min(times)),
            "max_total_ms": float(np.max(times)),
            "median_total_ms": float(np.median(times)),
            "mean_per_sample_ms": float(np.mean(per_sample_times)),
            "std_per_sample_ms": float(np.std(per_sample_times)),
            "throughput_samples_per_sec": float(batch_size * 1000 / np.mean(times)),
        }

    return results


def benchmark_all_precision_modes(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    batch_size: int = 32,
) -> Dict:
    """
    Benchmark model in FP32, FP16, and mixed precision modes.
    """
    results = {}

    # FP32 (default)
    logger.info("Benchmarking FP32...")
    results["fp32"] = benchmark_inference(
        model, input_shape, device, n_runs=500, batch_sizes=[batch_size]
    )

    # FP16 (if CUDA available)
    if device.type == "cuda":
        logger.info("Benchmarking FP16...")
        model_fp16 = model.half()
        results["fp16"] = benchmark_inference(
            model_fp16, input_shape, device, n_runs=500, batch_sizes=[batch_size]
        )
        model.float()  # Restore to FP32

    return results


# ==================== XGBoost Baseline ====================


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: np.ndarray,
) -> Dict:
    """
    Train XGBoost baseline model for comparison using MultiOutputRegressor.

    This approach trains a single XGBoost model wrapped in MultiOutputRegressor
    to handle all 48 CL output points simultaneously, which is more efficient
    than training 48 separate models.

    Returns comprehensive metrics for fair comparison with neural models.
    """
    if not XGBOOST_AVAILABLE:
        return {"error": "XGBoost not available"}

    logger.info("Training XGBoost Multi-Output baseline...")

    # Hyperparameters (can be tuned with Optuna as well)
    base_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_jobs": -1,
        "verbosity": 0,
    }

    logger.info("Creating MultiOutputRegressor with XGBoost base estimator...")
    logger.info("Base parameters: max_depth=%s, n_estimators=%s", base_params['max_depth'], base_params['n_estimators'])

    # Create base XGBoost regressor
    base_estimator = xgb.XGBRegressor(**base_params)

    # Wrap in MultiOutputRegressor for handling multiple outputs
    multi_output_model = MultiOutputRegressor(
        estimator=base_estimator,
        n_jobs=1,  # XGBoost handles parallelization internally
    )

    # Training with timing
    logger.info("Training on %d samples, %d outputs...", y_train.shape[0], y_train.shape[1])
    start_time = time.time()

    multi_output_model.fit(X_train, y_train)

    training_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds", training_time)

    # Predictions
    logger.info("Making predictions...")
    y_pred_train = multi_output_model.predict(X_train)
    y_pred_test = multi_output_model.predict(X_test)

    # Calculate metrics
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test - y_pred_test) ** 2)

    # Per-output MSE for analysis
    per_output_mse = np.mean((y_test - y_pred_test) ** 2, axis=0)

    # Physics-aware metrics (import from organised.src.utils)
    test_physics_score = None
    try:
        from organised.src.utils import calculate_physics_score

        test_physics_score = calculate_physics_score(y_test, y_pred_test, alpha_grid)
        logger.info("Physics Score: %.4f", test_physics_score)
    except Exception as e:
        logger.warning("Could not compute physics metrics: %s", e)

    # Count total parameters across all estimators
    total_trees = 0
    total_nodes = 0
    for estimator in multi_output_model.estimators_:
        booster = estimator.get_booster()
        trees_df = booster.trees_to_dataframe()
        total_trees += len(trees_df["Tree"].unique())
        total_nodes += len(trees_df)

    logger.info("Total trees: %d", total_trees)
    logger.info("Total nodes: %d", total_nodes)

    # Inference timing (batch of 1 for fair comparison)
    # Decreased number due to complexity leading to longer times
    n_runs = 100
    warmup_runs = 10

    # Warmup
    for _ in range(warmup_runs):
        _ = multi_output_model.predict(X_test[:1])

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = multi_output_model.predict(X_test[:1])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = np.array(times)

    # Batch inference timing (batch of 32)
    batch_32_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = multi_output_model.predict(X_test[:32])
        end = time.perf_counter()
        batch_32_times.append((end - start) * 1000)  # ms

    batch_32_times = np.array(batch_32_times)

    results = {
        "model_type": "XGBoost MultiOutputRegressor",
        "n_outputs": y_train.shape[1],
        "n_estimators_per_output": base_params["n_estimators"],
        "total_trees": total_trees,
        "total_nodes": total_nodes,
        "total_parameters": total_nodes,  # Approximate: each node is a parameter
        "hyperparameters": base_params,
        "training_time_sec": float(training_time),
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "per_output_mse_mean": float(per_output_mse.mean()),
        "per_output_mse_std": float(per_output_mse.std()),
        "per_output_mse_min": float(per_output_mse.min()),
        "per_output_mse_max": float(per_output_mse.max()),
        "test_physics_score": float(test_physics_score) if test_physics_score else None,
        "inference_time_single_mean_ms": float(np.mean(times)),
        "inference_time_single_std_ms": float(np.std(times)),
        "inference_time_single_median_ms": float(np.median(times)),
        "inference_time_batch32_mean_ms": float(np.mean(batch_32_times)),
        "inference_time_batch32_per_sample_ms": float(np.mean(batch_32_times) / 32),
        "throughput_batch32_samples_per_sec": float(
            32 * 1000 / np.mean(batch_32_times)
        ),
    }

    logger.info("XGBoost Summary:")
    logger.info("Test MSE: %.6f", results['test_mse'])
    logger.info("Training Time: %.1f sec", results['training_time_sec'])
    logger.info("Inference (single): %.4f ms", results['inference_time_single_mean_ms'])
    print(
        f"    Inference (batch 32): {results['inference_time_batch32_per_sample_ms']:.4f} ms/sample"
    )

    return results


# ==================== Validation Strategy Documentation ====================


def document_validation_strategy() -> Dict:
    """
    Document the validation strategy used in the Optuna study.
    This information should be extracted from the tuning script.
    """
    strategy = {
        "cross_validation": "Leave-One-Family-Out (LOFO)",
        "n_folds": 5,
        "early_stopping": {
            "enabled": True,
            "patience": 20,
            "metric": "physics_score",
            "check_frequency_epochs": 10,
            "description": (
                "Early stopping monitors the physics-aware score on the validation set "
                "every 10 epochs. Training stops if no improvement is seen for 20 consecutive "
                "checks (200 epochs)."
            ),
        },
        "hyperparameter_tuning": {
            "method": "Optuna TPE (Tree-structured Parzen Estimator)",
            "pruning": "MedianPruner with 10 warmup steps",
            "objective": "Minimize physics-aware score",
            "tuning_fold": "First LOFO fold used for hyperparameter search",
            "evaluation_folds": "All LOFO folds used for final evaluation",
        },
        "train_val_test_split": {
            "description": (
                "For each LOFO fold, one aerofoil family is held out as test set. "
                "Remaining families form the training set. Within Optuna trials, "
                "training set is further split 80-20 for training and validation "
                "to enable early stopping."
            ),
            "lofo_families": [
                "NACA (largest family)",
                "Gottingen",
                "Clark",
                "Eppler",
                "Selig",
            ],
        },
    }

    return strategy


# ==================== Main Analysis Pipeline ====================


def main():
    """
    Main pipeline to gather all model details for the paper.
    """
    print("=" * 80)
    logger.info("MODEL COMPLEXITY AND TRAINING DETAILS ANALYSIS")
    logger.info("For Paper Submission - Addressing Reviewer Comments")
    logger.info("%s", "=" * 80)

    OUTPUT_DIR = "organised/optuna_results/physics_aware/model_details"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. System Information
    logger.info("[1/4] Gathering System and Software Information...")
    system_info = get_system_info()

    logger.info("System: %s %s", system_info["os"], system_info["os_version"])
    logger.info("Processor: %s", system_info["processor"])
    logger.info(
        "CPU Cores: %d physical, %d logical",
        system_info["cpu_count"],
        system_info["cpu_count_logical"],
    )
    logger.info("RAM: %s GB", system_info["total_ram_gb"])
    logger.info("Python: %s", system_info["python_version"])
    logger.info("PyTorch: %s", system_info["pytorch_version"])

    if system_info["cuda_available"]:
        logger.info("CUDA: Available (%d device(s))", system_info["device_count"])
        for i in range(system_info["device_count"]):
            logger.info(
                "    GPU %d: %s (%s GB)",
                i,
                system_info[f"gpu_{i}_name"],
                system_info[f"gpu_{i}_memory_total_gb"],
            )
    else:
        logger.info("CUDA: Not available")

    # Save system info
    with open(os.path.join(OUTPUT_DIR, "system_info.json"), "w") as f:
        json.dump(system_info, f, indent=2)

    # 2. Load Best Models from Optuna Results
    logger.info("[2/4] Analyzing Best Model Architectures...")

    if not IMPROVED_MODELS_AVAILABLE:
        logger.error("Improved models not available. Cannot proceed.")
        return

    # Load best hyperparameters from Optuna results
    RESULTS_DIR = "optuna_results/physics_aware"
    BEST_MODELS_FILE = os.path.join(
        RESULTS_DIR, "best_models", "best_model_configs.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define input/output sizes (adjust based on your data)
    input_size = 12  # 5 CST params * 2 + LE + TE = 12 features
    output_size = 48  # 48 CL points

    # Try to load actual best model configurations
    model_configs = None
    if os.path.exists(BEST_MODELS_FILE):
        logger.info("Loading best model configurations from: %s", BEST_MODELS_FILE)
        with open(BEST_MODELS_FILE, "r") as f:
            model_configs = json.load(f)
    else:
        logger.warning("%s not found. Using example hyperparameters. Run extract_best_models.py first!", BEST_MODELS_FILE)

    # Convert loaded configs or use example hyperparameters
    if model_configs:
        example_models = {}
        for model_name, config in model_configs.items():
            hyperparams = config["hyperparameters"]

            if model_name == "MLP":
                model_class = MLPModel
            elif model_name == "GRU":
                model_class = GRUModel
            elif model_name == "LSTM":
                model_class = LSTMModel
            elif model_name == "CNN":
                model_class = CNNModel
            else:
                continue

            example_models[model_name] = {
                "hyperparams": hyperparams,
                "model_class": model_class,
                "best_loss": config.get("best_loss_function", "unknown"),
                "physics_score": config.get("physics_score", 0.0),
            }
    else:
        # Fallback: Example hyperparameters
        example_models = {
            "MLP": {
                "hyperparams": {
                    "hidden_size": 384,
                    "num_layers": 5,
                    "width_factor": 3,
                    "dropout_rate": 0.25,
                    "learning_rate": 0.0005,
                    "weight_decay": 0.0001,
                },
                "model_class": MLPModel,
                "best_loss": "example",
                "physics_score": 0.0,
            },
            "GRU": {
                "hyperparams": {
                    "hidden_size": 320,
                    "num_layers": 3,
                    "bidirectional": True,
                    "dropout_rate": 0.2,
                    "learning_rate": 0.0003,
                    "weight_decay": 0.00005,
                },
                "model_class": GRUModel,
                "best_loss": "example",
                "physics_score": 0.0,
            },
            "LSTM": {
                "hyperparams": {
                    "hidden_size": 384,
                    "num_layers": 3,
                    "bidirectional": True,
                    "dropout_rate": 0.25,
                    "learning_rate": 0.0002,
                    "weight_decay": 0.00008,
                },
                "model_class": LSTMModel,
                "best_loss": "example",
                "physics_score": 0.0,
            },
        }

    model_analyses = {}

    for model_name, config in example_models.items():
        logger.info("Analyzing %s", model_name)

        # Create model instance
        if model_name == "MLP":
            model = config["model_class"](
                input_size=input_size,
                output_size=output_size,
                hidden_size=config["hyperparams"]["hidden_size"],
                num_layers=config["hyperparams"]["num_layers"],
                width_factor=config["hyperparams"]["width_factor"],
                dropout=config["hyperparams"]["dropout_rate"],
            )
        elif model_name == "CNN":
            model = config["model_class"](
                input_size=input_size,
                output_size=output_size,
                num_filters=config["hyperparams"]["num_filters"],
                kernel_sizes=config["hyperparams"]["kernel_sizes"],
                dropout=config["hyperparams"]["dropout_rate"],
            )
        else:  # GRU or LSTM
            model = config["model_class"](
                input_size=input_size,
                hidden_size=config["hyperparams"]["hidden_size"],
                output_size=output_size,
                num_layers=config["hyperparams"]["num_layers"],
                bidirectional=config["hyperparams"]["bidirectional"],
                dropout=config["hyperparams"]["dropout_rate"],
            )

        # Analyze architecture
        analysis = analyze_model_architecture(model, model_name, config["hyperparams"])

        model_analyses[model_name] = analysis

        logger.info("Total Parameters: %d", analysis["total_parameters"])
        logger.info("Model Size: %.2f MB", analysis["model_size_mb"])

    # Save model analyses
    with open(os.path.join(OUTPUT_DIR, "model_architectures.json"), "w") as f:
        json.dump(model_analyses, f, indent=2)

    # 3. Benchmark Inference Times
    logger.info("[3/4] Benchmarking Inference Times...")

    inference_results = {}
    inference_results_cpu = {}

    for model_name, config in example_models.items():
        logger.info("Benchmarking %s", model_name)

        # Recreate model
        if model_name == "MLP":
            model = config["model_class"](
                input_size=input_size,
                output_size=output_size,
                hidden_size=config["hyperparams"]["hidden_size"],
                num_layers=config["hyperparams"]["num_layers"],
                width_factor=config["hyperparams"]["width_factor"],
                dropout=config["hyperparams"]["dropout_rate"],
            )
            input_shape = (input_size,)
        elif model_name == "CNN":
            model = config["model_class"](
                input_size=input_size,
                output_size=output_size,
                num_filters=config["hyperparams"]["num_filters"],
                kernel_sizes=config["hyperparams"]["kernel_sizes"],
                dropout=config["hyperparams"]["dropout_rate"],
            )
            input_shape = (input_size,)
        else:  # GRU or LSTM
            model = config["model_class"](
                input_size=input_size,
                hidden_size=config["hyperparams"]["hidden_size"],
                output_size=output_size,
                num_layers=config["hyperparams"]["num_layers"],
                bidirectional=config["hyperparams"]["bidirectional"],
                dropout=config["hyperparams"]["dropout_rate"],
            )
            input_shape = (input_size,)

        # Benchmark on GPU (if available)
        if device.type == "cuda":
            logger.info("GPU Benchmarking...")
            timing_results = benchmark_inference(
                model,
                input_shape,
                device,
                n_runs=1000,
                warmup_runs=100,
                batch_sizes=[1, 8, 32, 64, 128],
            )

            inference_results[model_name] = timing_results

            # Print GPU summary
            single_sample = timing_results["batch_1"]["mean_per_sample_ms"]
            batch_32 = timing_results["batch_32"]["mean_per_sample_ms"]
            throughput = timing_results["batch_32"]["throughput_samples_per_sec"]

            print(
                f"      GPU Single sample: {single_sample:.4f} ± "
                f"{timing_results['batch_1']['std_per_sample_ms']:.4f} ms"
            )
            print(
                f"      GPU Batch 32: {batch_32:.4f} ms/sample "
                f"({throughput:.1f} samples/sec)"
            )

        # Benchmark on CPU
        logger.info("CPU Benchmarking...")
        cpu_device = torch.device("cpu")
        timing_results_cpu = benchmark_inference(
            model,
            input_shape,
            cpu_device,
            n_runs=1000,
            warmup_runs=100,
            batch_sizes=[1, 8, 32, 64, 128],
        )

        inference_results_cpu[model_name] = timing_results_cpu

        # Print CPU summary
        single_sample_cpu = timing_results_cpu["batch_1"]["mean_per_sample_ms"]
        batch_32_cpu = timing_results_cpu["batch_32"]["mean_per_sample_ms"]
        throughput_cpu = timing_results_cpu["batch_32"]["throughput_samples_per_sec"]

        print(
            f"      CPU Single sample: {single_sample_cpu:.4f} ± "
            f"{timing_results_cpu['batch_1']['std_per_sample_ms']:.4f} ms"
        )
        print(
            f"      CPU Batch 32: {batch_32_cpu:.4f} ms/sample "
            f"({throughput_cpu:.1f} samples/sec)"
        )

        # Print speedup if GPU available
        if device.type == "cuda":
            speedup_single = single_sample_cpu / single_sample
            speedup_batch = batch_32_cpu / batch_32
            logger.info("GPU Speedup (batch 1): %.2f x", speedup_single)
            logger.info("GPU Speedup (batch 32): %.2f x", speedup_batch)

    # Save inference results
    if device.type == "cuda":
        with open(os.path.join(OUTPUT_DIR, "inference_benchmarks_gpu.json"), "w") as f:
            json.dump(inference_results, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "inference_benchmarks_cpu.json"), "w") as f:
        json.dump(inference_results_cpu, f, indent=2)

    # Save combined comparison
    combined_results = {
        "gpu": inference_results if device.type == "cuda" else {},
        "cpu": inference_results_cpu,
        "speedup_comparison": {},
    }

    if device.type == "cuda":
        for model_name in inference_results.keys():
            combined_results["speedup_comparison"][model_name] = {
                "batch_1_speedup": inference_results_cpu[model_name]["batch_1"][
                    "mean_per_sample_ms"
                ]
                / inference_results[model_name]["batch_1"]["mean_per_sample_ms"],
                "batch_32_speedup": inference_results_cpu[model_name]["batch_32"][
                    "mean_per_sample_ms"
                ]
                / inference_results[model_name]["batch_32"]["mean_per_sample_ms"],
            }

    with open(
        os.path.join(OUTPUT_DIR, "inference_benchmarks_comparison.json"), "w"
    ) as f:
        json.dump(combined_results, f, indent=2)

    # 4. XGBoost Baseline (if data available)
    logger.info("[4/4] Training XGBoost Baseline...")

    # Try to load data
    CSV_FILE = "data/csv/dataset_12CST_params.csv"

    xgboost_results = None
    if os.path.exists(CSV_FILE) and XGBOOST_AVAILABLE:
        try:
            df = pd.read_csv(CSV_FILE)

            # Prepare data
            feature_cols = [
                col
                for col in df.columns
                if col.startswith(
                    (
                        "lower_weight_",
                        "upper_weight_",
                        "leading_edge_weight",
                        "TE_thickness",
                    )
                )
            ]
            cl_cols = [f"CL_{i}" for i in range(48)]

            X = df[feature_cols].values.astype(np.float32)
            y = df[cl_cols].values.astype(np.float32)

            # Simple train-test split (80-20)
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Alpha grid for physics metrics
            alpha_grid = np.linspace(-20, 20, 48)

            # Train XGBoost
            xgboost_results = train_xgboost_baseline(
                X_train, y_train, X_test, y_test, alpha_grid
            )

            logger.info("XGBoost Results:")
            logger.info("Test MSE: %.6f", xgboost_results['test_mse'])
            if xgboost_results["test_physics_score"]:
                logger.info("Physics Score: %.4f", xgboost_results['test_physics_score'])
            logger.info(
                "Inference Time: %.4f ms", xgboost_results['inference_time_mean_ms']
            )
            logger.info(
                "Training Time: %.1f sec", xgboost_results['training_time_total_sec']
            )

            # Save results
            with open(os.path.join(OUTPUT_DIR, "xgboost_baseline.json"), "w") as f:
                json.dump(xgboost_results, f, indent=2)

        except Exception as e:
            logger.error("Error training XGBoost: %s", e)
    else:
        if not XGBOOST_AVAILABLE:
            logger.info("Skipped: XGBoost not installed")
        else:
            logger.info("Skipped: Data file not found: %s", CSV_FILE)

    logger.info("%s", "\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("%s", "=" * 80)
    logger.info("All results saved to: %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
