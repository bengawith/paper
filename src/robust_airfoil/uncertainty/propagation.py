from __future__ import annotations

import numpy as np

from robust_airfoil.uncertainty.fields import PerturbedGeometry, smooth_normal_perturbation
from robust_airfoil.uncertainty.sampling import sobol_normal_samples


def generate_perturbations(
    upper: np.ndarray,
    lower: np.ndarray,
    count: int,
    controls_per_surface: int,
    amplitude: float,
    seed: int = 20260824,
    upper_lower_correlation: float = 0.0,
    correlation_length: float = 0.15,
    leading_edge_taper: bool = True,
    trailing_edge_zero_displacement: bool = True,
) -> list[PerturbedGeometry]:
    latent = sobol_normal_samples(count, controls_per_surface * 2, seed, antithetic=True)
    if not -1 <= upper_lower_correlation <= 1:
        raise ValueError("upper/lower correlation must be in [-1, 1]")
    independent_scale = float(np.sqrt(max(0.0, 1 - upper_lower_correlation**2)))
    return [
        smooth_normal_perturbation(
            upper,
            lower,
            row[:controls_per_surface],
            upper_lower_correlation * row[:controls_per_surface] + independent_scale * row[controls_per_surface:],
            amplitude,
            correlation_length,
            leading_edge_taper,
            trailing_edge_zero_displacement,
        )
        for row in latent
    ]
