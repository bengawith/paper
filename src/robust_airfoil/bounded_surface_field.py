"""Bounded correlated cubic normal-surface field for uncertainty experiments.

This describes a prescribed numerical perturbation law, not measured
manufacturing variability. Every candidate needs its own geometry checks after
perturbation; this module never certifies geometry, solver trust, or feasibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import qmc


@dataclass(frozen=True)
class FieldConfig:
    controls: int = 8
    control_correlation_length: float = 0.15
    upper_lower_correlation: float = 0.0
    taper_fraction: float = 0.03

    def validate(self) -> None:
        if self.controls < 4:
            raise ValueError("At least four controls are required for a cubic spline")
        if not np.isfinite(self.control_correlation_length) or self.control_correlation_length <= 0:
            raise ValueError("Correlation length must be finite and positive")
        if not np.isfinite(self.upper_lower_correlation) or abs(self.upper_lower_correlation) > 1:
            raise ValueError("Correlation must lie in [-1, 1]")
        if not np.isfinite(self.taper_fraction) or not 0 < self.taper_fraction < 0.5:
            raise ValueError("Taper fraction must lie strictly between zero and one half")


def latent_samples(config: FieldConfig, count: int, seed: int, method: str = "sobol") -> np.ndarray:
    """Return bounded latent rows; Sobol rows are quadrature points, not IID."""
    config.validate()
    if count < 1:
        raise ValueError("A positive sample count is required")
    dimension = 2 * config.controls
    if method == "sobol":
        if count & (count - 1):
            raise ValueError("Use a power-of-two Sobol sample count")
        values = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(count.bit_length() - 1)
    elif method == "iid":
        values = np.random.default_rng(seed).uniform(size=(count, dimension))
    else:
        raise ValueError("Sampling method must be 'sobol' or 'iid'")
    return 2.0 * values - 1.0


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values**3 * (10.0 - 15.0 * values + 6.0 * values**2)


def operator(x: np.ndarray, config: FieldConfig) -> np.ndarray:
    """Return a candidate-independent globally bounded field operator."""
    config.validate()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("x must be a finite strictly increasing vector")
    if not np.isclose(x[0], 0.0, atol=1e-12, rtol=0) or not np.isclose(
        x[-1], 1.0, atol=1e-12, rtol=0
    ):
        raise ValueError("x must span the normalised nominal chord [0, 1]")

    count = config.controls
    internal = np.linspace(0.0, 1.0, count - 2)[1:-1]
    knots = np.concatenate((np.zeros(4), internal, np.ones(4)))
    basis = BSpline(knots, np.eye(count), 3, extrapolate=False)(x)
    control_x = np.linspace(0.0, 1.0, count)
    separation = control_x[:, None] - control_x[None, :]
    covariance = np.exp(-0.5 * (separation / config.control_correlation_length) ** 2)
    factor = np.linalg.cholesky(covariance + 1e-12 * np.eye(count))
    denominator = np.max(np.abs(factor).sum(axis=1)) * np.sqrt(2.0)
    taper = _smoothstep(x / config.taper_fraction) * _smoothstep(
        (1.0 - x) / config.taper_fraction
    )
    return taper[:, None] * (basis @ factor) / denominator


def displacement_fields(
    x: np.ndarray,
    latents: np.ndarray,
    epsilon: float,
    config: FieldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed upper/lower normal deviations in chord units."""
    latent_values = np.asarray(latents, dtype=np.float64)
    if latent_values.ndim != 2 or latent_values.shape[1] != 2 * config.controls or not len(latent_values):
        raise ValueError("Latent matrix has the wrong shape")
    if not np.isfinite(latent_values).all() or np.max(np.abs(latent_values)) > 1.0 + 1e-12:
        raise ValueError("Latent values must be finite and bounded by one")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")

    matrix = operator(x, config)
    top_latents = latent_values[:, : config.controls]
    independent = latent_values[:, config.controls :]
    rho = config.upper_lower_correlation
    bottom_latents = rho * top_latents + np.sqrt(max(0.0, 1.0 - rho * rho)) * independent
    upper = epsilon * (top_latents @ matrix.T)
    lower = epsilon * (bottom_latents @ matrix.T)
    if max(float(np.max(np.abs(upper))), float(np.max(np.abs(lower)))) > epsilon + 1e-12:
        raise AssertionError("Analytical displacement bound violated")
    return upper, lower


def surface_normals(surface: np.ndarray, upper: bool) -> np.ndarray:
    surface = np.asarray(surface, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1] != 2 or len(surface) < 4 or not np.isfinite(surface).all():
        raise ValueError("A finite N x 2 surface is required")
    if np.any(np.diff(surface[:, 0]) <= 0):
        raise ValueError("Nominal surfaces must be strictly ordered LE -> TE")
    tangent = np.gradient(surface, axis=0, edge_order=2)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    norms = np.linalg.norm(normal, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Degenerate tangent")
    normal /= norms[:, None]
    return normal if upper else -normal


def perturb_candidate(
    upper: np.ndarray,
    lower: np.ndarray,
    latents: np.ndarray,
    epsilon: float,
    config: FieldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce actual candidate-specific displaced coordinates on a common x grid."""
    upper_values = np.asarray(upper, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    if upper_values.shape != lower_values.shape or not np.allclose(
        upper_values[:, 0], lower_values[:, 0], atol=1e-12, rtol=0
    ):
        raise ValueError("Nominal upper/lower surfaces must use a common x grid")
    top_delta, bottom_delta = displacement_fields(upper_values[:, 0], latents, epsilon, config)
    return (
        upper_values[None, :, :]
        + top_delta[:, :, None] * surface_normals(upper_values, True)[None, :, :],
        lower_values[None, :, :]
        + bottom_delta[:, :, None] * surface_normals(lower_values, False)[None, :, :],
    )


def field_context_hash(config: FieldConfig, x: np.ndarray, latents: np.ndarray, epsilon: float) -> str:
    metadata = {
        "schema": "bounded_cubic_normal_field_v1",
        "config": asdict(config),
        "epsilon": epsilon,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for values in (x, latents):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
