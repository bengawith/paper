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
    """Fit with a nonnegative trailing-edge thickness, re-solving at the bound."""
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
    if parameters[-1] < 0:
        # Exact active-set solution with the sole TE >= 0 constraint.
        other_parameters, *_ = np.linalg.lstsq(design[:, :-1], target, rcond=None)
        parameters = np.concatenate([other_parameters, [0.0]])
    reconstructed_upper, reconstructed_lower = reconstruct_cst(parameters, x)
    errors = np.concatenate([reconstructed_upper - upper[:, 1], reconstructed_lower - lower[:, 1]])
    return CSTFit(parameters, np.column_stack([x, reconstructed_upper]), np.column_stack([x, reconstructed_lower]), float(np.sqrt(np.mean(errors**2))), float(np.max(np.abs(errors))))


@dataclass(frozen=True)
class BatchCSTFit:
    parameters: np.ndarray
    reconstructed_upper: np.ndarray
    reconstructed_lower: np.ndarray
    rms_error: np.ndarray
    max_error: np.ndarray


class BatchCSTFitter:
    """Reuse exact fixed-grid linear operators; never reuse candidate objectives."""
    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x,dtype=float).copy()
        if self.x.ndim != 1 or len(self.x) < 7 or not np.isfinite(self.x).all():
            raise ValueError('A finite one-dimensional CST grid is required')
        if np.any(np.diff(self.x) <= 0) or self.x[0] < 0 or self.x[-1] > 1:
            raise ValueError('CST grid must increase within [0,1]')
        basis,le = _basis(self.x),_leading_edge_basis(self.x)
        zeros = np.zeros_like(basis)
        self.design = np.vstack([
            np.column_stack([basis,zeros,-le,-.5*self.x]),
            np.column_stack([zeros,basis,le,.5*self.x]),
        ])
        # Same SVD rank convention as fit_cst, applied once to the identity.
        identity = np.eye(len(self.design))
        self.inverse = np.linalg.lstsq(self.design,identity,rcond=None)[0]
        self.closed_inverse = np.linalg.lstsq(self.design[:,:-1],identity,rcond=None)[0]

    def fit_many(self, upper: np.ndarray, lower: np.ndarray) -> BatchCSTFit:
        upper,lower = np.asarray(upper,float),np.asarray(lower,float)
        if upper.ndim != 2 or upper.shape != lower.shape or upper.shape[1] != len(self.x):
            raise ValueError('Batch CST surfaces must have shape (samples,grid_points)')
        if not np.isfinite(upper).all() or not np.isfinite(lower).all():
            raise ValueError('Batch CST surfaces must be finite')
        target = np.concatenate([lower,upper],axis=1)
        parameters = target @ self.inverse.T
        active = parameters[:,-1] < 0
        if active.any():
            parameters[active,:-1] = target[active] @ self.closed_inverse.T
            parameters[active,-1] = 0.0
        reconstructed = parameters @ self.design.T
        reconstructed_lower,reconstructed_upper = np.split(reconstructed,2,axis=1)
        errors = reconstructed-target
        return BatchCSTFit(parameters,reconstructed_upper,reconstructed_lower,
                           np.sqrt(np.mean(errors**2,axis=1)),np.max(abs(errors),axis=1))
