"""Candidate-specific correlated surface-normal manufacturing scenarios.

The same latent Sobol draw is shared across designs. Normal displacement
is applied to each *candidate's own* coordinates, then refitted for surrogate
input. No reference-CST increments are transferred to different designs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robust_airfoil.config import UncertaintyConfig
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.normalise import resample_surfaces_to_common_x
from robust_airfoil.uncertainty.fields import smooth_normal_perturbation


@dataclass(frozen=True)
class CandidateScenarios:
    parameters: np.ndarray
    invalid: np.ndarray
    invalid_reasons: tuple[str, ...]


def candidate_surface_scenarios(
    parameters: np.ndarray,
    latent: np.ndarray,
    config: UncertaintyConfig,
) -> CandidateScenarios:
    """Refit genuine displaced geometry for each shared latent scenario."""
    original = np.asarray(parameters, dtype=float)
    draws = np.asarray(latent, dtype=float)
    controls = config.basis_control_points_per_surface
    if original.shape != (12,) or draws.ndim != 2 or draws.shape[1] != controls * 2:
        raise ValueError("Expected twelve CST parameters and (samples, 2*controls) draws")
    if not np.isfinite(original).all() or not np.isfinite(draws).all():
        raise ValueError("CST parameters and uncertainty draws must be finite")
    x = (1 - np.cos(np.linspace(0, np.pi, config.cosine_points_per_surface))) / 2
    uy, ly = reconstruct_cst(original, x)
    upper, lower = np.column_stack([x, uy]), np.column_stack([x, ly])
    out = np.tile(original, (len(draws), 1))
    invalid = np.zeros(len(draws), dtype=bool)
    reasons: list[str] = []
    for index, row in enumerate(draws):
        reason = ""
        try:
            correlation = float(config.upper_lower_correlation)
            if not -1.0 <= correlation <= 1.0:
                raise ValueError("upper_lower_correlation must be in [-1,1]")
            upper_modes = row[:controls]
            lower_modes = (
                correlation * upper_modes + np.sqrt(max(0.0, 1.0 - correlation**2)) * row[controls:]
            )
            pert = smooth_normal_perturbation(
                upper,
                lower,
                upper_modes,
                lower_modes,
                max(config.amplitudes_fraction_chord),
                config.correlation_length_chord,
                config.leading_edge_taper,
                config.trailing_edge_zero_displacement,
            )
            if not pert.valid:
                reason = ";".join(pert.invalid_reasons) or "invalid_geometry"
            else:
                upper_fit, lower_fit = resample_surfaces_to_common_x(pert.upper, pert.lower, x)
                fit = fit_cst(upper_fit, lower_fit)
                if not np.isfinite(fit.parameters).all():
                    reason = "nonfinite_refit"
                elif fit.max_error > config.maximum_cst_refit_error_fraction_chord:
                    reason = "excessive_refit_error"
                else:
                    out[index] = fit.parameters
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            reason = f"refit_error:{type(exc).__name__}"
        invalid[index] = bool(reason)
        reasons.append(reason)
    # Invalid placeholders carry the nominal parameters solely to allow
    # batched matrix shape; callers must penalise every invalid row, and
    # report invalid counts. Never use those nominal predictions as outcomes.
    return CandidateScenarios(out, invalid, tuple(reasons))
