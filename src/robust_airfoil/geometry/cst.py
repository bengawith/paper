from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class CSTFit:
    parameters: np.ndarray
    reconstructed_upper: np.ndarray
    reconstructed_lower: np.ndarray
    rms_error: float
    max_error: float


def _basis(x: np.ndarray, n: int = 4) -> np.ndarray:
    bernstein = np.column_stack([comb(n, i) * x**i * (1 - x) ** (n - i) for i in range(n + 1)])
    class_function = np.sqrt(np.clip(x, 0, 1)) * (1 - x)
    return class_function[:, None] * bernstein


def _leading_edge_basis(x: np.ndarray) -> np.ndarray:
    """A local leading-edge modifier outside the degree-4 Bernstein span."""
    return np.sqrt(np.clip(x, 0, 1)) * (1 - x) * np.exp(-20 * x)


def reconstruct_cst(parameters: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(parameters) != 12:
        raise ValueError("CST vector must have 12 parameters")
    lower_weights = parameters[:5]
    upper_weights = parameters[5:10]
    leading_edge_weight, te_thickness = parameters[10], parameters[11]
    basis = _basis(x)
    le_basis = _leading_edge_basis(x)
    upper = basis @ upper_weights + leading_edge_weight * le_basis + 0.5 * te_thickness * x
    lower = basis @ lower_weights - leading_edge_weight * le_basis - 0.5 * te_thickness * x
    return upper, lower


def fit_cst(upper: np.ndarray, lower: np.ndarray) -> CSTFit:
    if upper.shape != lower.shape or not np.allclose(upper[:, 0], lower[:, 0]):
        raise ValueError("Upper/lower surfaces must share x grid")
    x = upper[:, 0]
    basis = _basis(x)
    le = _leading_edge_basis(x)
    zeros = np.zeros_like(basis)
    design_upper = np.column_stack([zeros, basis, le, 0.5 * x])
    design_lower = np.column_stack([basis, zeros, -le, -0.5 * x])
    design = np.vstack([design_lower, design_upper])
    target = np.concatenate([lower[:, 1], upper[:, 1]])
    parameters, *_ = np.linalg.lstsq(design, target, rcond=None)
    reconstructed_upper, reconstructed_lower = reconstruct_cst(parameters, x)
    errors = np.concatenate([reconstructed_upper - upper[:, 1], reconstructed_lower - lower[:, 1]])
    return CSTFit(parameters, np.column_stack([x, reconstructed_upper]), np.column_stack([x, reconstructed_lower]), float(np.sqrt(np.mean(errors**2))), float(np.max(np.abs(errors))))
