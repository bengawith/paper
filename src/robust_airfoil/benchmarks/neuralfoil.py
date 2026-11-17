from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeuralFoilBenchmarkResult:
    status: str
    reason: str
    cl: np.ndarray | None = None
    cd: np.ndarray | None = None
    cm: np.ndarray | None = None
    analysis_confidence: np.ndarray | None = None


def benchmark_neuralfoil(kulfan_parameters: dict[str, object], alpha_deg: np.ndarray, reynolds_number: float = 1e6) -> NeuralFoilBenchmarkResult:
    try:
        import neuralfoil as nf
    except ImportError:
        return NeuralFoilBenchmarkResult("unavailable", "neuralfoil optional dependency is not installed")
    try:
        result = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan_parameters,
            alpha=alpha_deg,
            Re=reynolds_number,
            model_size="xlarge",
        )
        return NeuralFoilBenchmarkResult(
            "ok",
            "diagnostic only",
            np.asarray(result["CL"]).reshape(-1),
            np.asarray(result["CD"]).reshape(-1),
            np.asarray(result["CM"]).reshape(-1),
            np.asarray(result.get("analysis_confidence", np.full_like(alpha_deg, np.nan))).reshape(-1),
        )
    except Exception as exc:
        return NeuralFoilBenchmarkResult("error", repr(exc))


def benchmark_neuralfoil_from_coordinates(
    coordinates: np.ndarray,
    alpha_deg: np.ndarray,
    reynolds_number: float = 1e6,
) -> NeuralFoilBenchmarkResult:
    """Refit arbitrary CST geometry to NeuralFoil's fixed eight-weight Kulfan form."""
    try:
        import aerosandbox as asb
    except ImportError:
        return NeuralFoilBenchmarkResult("unavailable", "aerosandbox benchmark dependency is not installed")
    try:
        airfoil = asb.Airfoil(name="robust_airfoil_reference", coordinates=np.asarray(coordinates, dtype=float))
        kulfan = airfoil.to_kulfan_airfoil(n_weights_per_side=8)
        return benchmark_neuralfoil(kulfan.kulfan_parameters, alpha_deg, reynolds_number)
    except Exception as exc:
        return NeuralFoilBenchmarkResult("error", repr(exc))
