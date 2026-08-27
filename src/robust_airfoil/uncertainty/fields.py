from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robust_airfoil.geometry.validity import validate_geometry


@dataclass(frozen=True)
class PerturbedGeometry:
    upper: np.ndarray
    lower: np.ndarray
    upper_displacement: np.ndarray
    lower_displacement: np.ndarray
    valid: bool
    invalid_reasons: tuple[str, ...]


def surface_normals(surface: np.ndarray, upper: bool) -> np.ndarray:
    tangent = np.gradient(surface, axis=0)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(length, 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    desired_sign = 1.0 if upper else -1.0
    flip = np.sign(np.nanmean(normal[:, 1])) != desired_sign
    return -normal if flip else normal


def correlated_control_field(control_x: np.ndarray, latent: np.ndarray, x: np.ndarray, correlation_length: float) -> np.ndarray:
    covariance = np.exp(-0.5 * ((control_x[:, None] - control_x[None, :]) / correlation_length) ** 2)
    factor = np.linalg.cholesky(covariance + 1e-9 * np.eye(len(control_x)))
    controls = factor @ latent
    field = np.interp(x, control_x, controls)
    maximum = float(np.max(np.abs(field)))
    return field / maximum if maximum > 1e-12 else np.zeros_like(field)


def smooth_normal_perturbation(
    upper: np.ndarray,
    lower: np.ndarray,
    latent_upper: np.ndarray,
    latent_lower: np.ndarray,
    amplitude: float,
    correlation_length: float = 0.15,
    leading_edge_taper: bool = True,
    trailing_edge_zero_displacement: bool = True,
) -> PerturbedGeometry:
    if len(latent_upper) != len(latent_lower):
        raise ValueError("Upper and lower control dimensions differ")
    x = upper[:, 0]
    controls = (1 - np.cos(np.linspace(0, np.pi, len(latent_upper)))) / 2
    upper_field = correlated_control_field(controls, latent_upper, x, correlation_length)
    lower_field = correlated_control_field(controls, latent_lower, x, correlation_length)
    taper = np.ones_like(x)
    if leading_edge_taper:
        leading_edge_coordinate = np.clip(x / 0.10, 0.0, 1.0)
        taper *= leading_edge_coordinate**2 * (3 - 2 * leading_edge_coordinate)
    if trailing_edge_zero_displacement:
        taper *= 1 - np.clip(x, 0.0, 1.0)
    upper_displacement = amplitude * upper_field * taper
    lower_displacement = amplitude * lower_field * taper
    perturbed_upper = upper + surface_normals(upper, True) * upper_displacement[:, None]
    perturbed_lower = lower + surface_normals(lower, False) * lower_displacement[:, None]
    validity = validate_geometry(perturbed_upper, perturbed_lower)
    return PerturbedGeometry(perturbed_upper, perturbed_lower, upper_displacement, lower_displacement, validity.valid, validity.reasons)
