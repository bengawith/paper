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
from robust_airfoil.geometry.validity import validate_geometry
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
                    refit_upper, refit_lower = reconstruct_cst(fit.parameters, x)
                    refit_validity = validate_geometry(
                        np.column_stack([x, refit_upper]),
                        np.column_stack([x, refit_lower]),
                    )
                    if not refit_validity.valid:
                        reason = "invalid_refitted_geometry:" + ";".join(refit_validity.reasons)
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


class PreparedCandidateScenarios:
    """Exact shared-field preparation with candidate-specific displaced coordinates.

    The scalar function above remains the independently testable reference.
    Only latent fields and fixed-grid linear maps are cached, never reference
    shape increments. Dense post-refit checks are additional fail-closed gates.
    """
    def __init__(self, latent: np.ndarray, config: UncertaintyConfig, dense_verification_points: int = 2001):
        from robust_airfoil.geometry.cst import BatchCSTFitter
        from robust_airfoil.uncertainty.fields import correlated_control_field

        self.config = config
        self.latent = np.asarray(latent,float).copy()
        controls = config.basis_control_points_per_surface
        if self.latent.ndim != 2 or self.latent.shape[1] != 2*controls or not len(self.latent):
            raise ValueError('Prepared scenarios require nonempty common latent draws')
        if not np.isfinite(self.latent).all() or dense_verification_points < config.cosine_points_per_surface:
            raise ValueError('Finite draws and a sufficiently dense validation grid are required')
        rho = float(config.upper_lower_correlation)
        if not -1 <= rho <= 1:
            raise ValueError('upper_lower_correlation must be in [-1,1]')
        self.x = (1-np.cos(np.linspace(0,np.pi,config.cosine_points_per_surface)))/2
        self.dense_x = (1-np.cos(np.linspace(0,np.pi,dense_verification_points)))/2
        self.fitter = BatchCSTFitter(self.x)
        # Columns map coefficients to (upper,lower) ordinates at the denser grid.
        self.dense_upper = np.column_stack([reconstruct_cst(unit,self.dense_x)[0] for unit in np.eye(12)])
        self.dense_lower = np.column_stack([reconstruct_cst(unit,self.dense_x)[1] for unit in np.eye(12)])
        control_x = (1-np.cos(np.linspace(0,np.pi,controls)))/2
        taper = np.ones_like(self.x)
        if config.leading_edge_taper:
            s = np.clip(self.x/.10,0.,1.)
            taper *= s*s*(3-2*s)
        if config.trailing_edge_zero_displacement:
            taper *= 1-np.clip(self.x,0.,1.)
        amplitude = max(config.amplitudes_fraction_chord)
        top,bottom = [],[]
        for row in self.latent:
            lower_modes = rho*row[:controls]+np.sqrt(max(0.,1-rho*rho))*row[controls:]
            top.append(amplitude*correlated_control_field(control_x,row[:controls],self.x,config.correlation_length_chord)*taper)
            bottom.append(amplitude*correlated_control_field(control_x,lower_modes,self.x,config.correlation_length_chord)*taper)
        self.upper_displacement = np.asarray(top)
        self.lower_displacement = np.asarray(bottom)

    def displaced_surfaces(self, parameters: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        from robust_airfoil.uncertainty.fields import surface_normals

        candidate = np.asarray(parameters,float)
        if candidate.shape != (12,) or not np.isfinite(candidate).all():
            raise ValueError('Expected twelve finite candidate CST coefficients')
        uy,ly = reconstruct_cst(candidate,self.x)
        upper,lower = np.column_stack([self.x,uy]),np.column_stack([self.x,ly])
        return (
            upper[None,:,:]+self.upper_displacement[:,:,None]*surface_normals(upper,True)[None,:,:],
            lower[None,:,:]+self.lower_displacement[:,:,None]*surface_normals(lower,False)[None,:,:],
        )

    def _resample_valid(self, surface: np.ndarray) -> np.ndarray:
        rounded_x = np.round(surface[:,0],12)
        if np.all(np.diff(rounded_x)>0):
            if self.x[0] < rounded_x[0]-1e-10 or self.x[-1] > rounded_x[-1]+1e-10:
                raise ValueError('Common grid falls outside displaced surface extent')
            return np.interp(self.x,rounded_x,surface[:,1])
        # Preserve the scalar implementation's rare duplicate-rounding policy.
        return resample_surfaces_to_common_x(surface,surface,self.x)[0][:,1]

    def evaluate(self, parameters: np.ndarray) -> CandidateScenarios:
        candidate = np.asarray(parameters, float)
        upper, lower = self.displaced_surfaces(candidate)
        result = np.tile(candidate, (len(self.latent), 1))
        valid = validate_surface_batches(upper, lower)
        reasons = ["" if value else "invalid_displaced_geometry" for value in valid]
        extent_ok = np.round(upper[:, 0, 0], 12) <= self.x[0] + 1e-10
        extent_ok &= np.round(lower[:, 0, 0], 12) <= self.x[0] + 1e-10
        extent_ok &= np.round(upper[:, -1, 0], 12) >= self.x[-1] - 1e-10
        extent_ok &= np.round(lower[:, -1, 0], 12) >= self.x[-1] - 1e-10
        for i in np.flatnonzero(valid & ~extent_ok):
            reasons[i] = "refit_error:ValueError"
        valid &= extent_ok
        retained = np.flatnonzero(valid)
        if not len(retained):
            return CandidateScenarios(result, ~valid, tuple(reasons))
        # Round the physical knots exactly as the scalar resampling routine.
        # Only the rare collapsed rounded knots take its duplicate-averaging path.
        top = upper[retained].copy()
        bottom = lower[retained].copy()
        top[:, :, 0] = np.round(top[:, :, 0], 12)
        bottom[:, :, 0] = np.round(bottom[:, :, 0], 12)
        regular = np.all(np.diff(top[:, :, 0], axis=1) > 0, axis=1)
        regular &= np.all(np.diff(bottom[:, :, 0], axis=1) > 0, axis=1)
        ups = np.empty((len(retained), len(self.x)))
        lows = np.empty_like(ups)
        if regular.any():
            ups[regular] = interpolate_surface_batches(self.x, top[regular])
            lows[regular] = interpolate_surface_batches(self.x, bottom[regular])
        for collapsed_index in np.flatnonzero(~regular):
            ups[collapsed_index] = self._resample_valid(upper[retained[collapsed_index]])
            lows[collapsed_index] = self._resample_valid(lower[retained[collapsed_index]])
        fit = self.fitter.fit_many(ups, lows)
        dense_upper = fit.parameters @ self.dense_upper.T
        dense_lower = fit.parameters @ self.dense_lower.T
        thickness = dense_upper - dense_lower
        fit_finite = np.isfinite(fit.parameters).all(axis=1)
        fit_error_ok = fit.max_error <= self.config.maximum_cst_refit_error_fraction_chord
        dense_valid = (np.min(thickness, axis=1) >= -1e-8)
        dense_valid &= np.trapezoid(thickness, self.dense_x, axis=1) > 0
        dense_error_ok = np.ones(len(retained), bool)
        if len(self.dense_x) > len(self.x):
            top_error = abs(dense_upper - interpolate_surface_batches(self.dense_x, upper[retained]))
            bottom_error = abs(dense_lower - interpolate_surface_batches(self.dense_x, lower[retained]))
            dense_error = np.maximum(np.max(top_error, axis=1), np.max(bottom_error, axis=1))
            dense_error_ok = dense_error <= self.config.maximum_cst_refit_error_fraction_chord
        admitted = fit_finite & fit_error_ok & dense_valid & dense_error_ok
        result[retained[admitted]] = fit.parameters[admitted]
        for j, row_index in enumerate(retained.tolist()):
            if not fit_finite[j]:
                reasons[row_index] = "nonfinite_refit"
            elif not fit_error_ok[j]:
                reasons[row_index] = "excessive_refit_error"
            elif not dense_valid[j]:
                reasons[row_index] = "invalid_refitted_geometry:surface_crossing_or_nonpositive_area"
            elif not dense_error_ok[j]:
                reasons[row_index] = "excessive_dense_refit_error"
        invalid = np.asarray([bool(reason) for reason in reasons])
        return CandidateScenarios(result, invalid, tuple(reasons))


def interpolate_surface_batches(queries: np.ndarray, surfaces: np.ndarray) -> np.ndarray:
    """Apply NumPy's actual piecewise-linear interpolation to aligned polylines.

    Validation is vectorised once; each interpolation uses the same compiled
    routine as the scalar reference, avoiding shifted-coordinate roundoff.
    """
    surfaces = np.asarray(surfaces, float)
    queries = np.asarray(queries, float)
    if surfaces.ndim != 3 or surfaces.shape[2] != 2 or surfaces.shape[1] < 2:
        raise ValueError("Expected (samples, points, 2) surfaces")
    if not np.isfinite(surfaces).all() or np.any(np.diff(surfaces[:, :, 0], axis=1) <= 0):
        raise ValueError("Surface knots must be finite and strictly increasing")
    if queries.ndim == 1:
        queries = np.broadcast_to(queries, (len(surfaces), len(queries)))
    if queries.ndim != 2 or len(queries) != len(surfaces) or not np.isfinite(queries).all():
        raise ValueError("Query grids must be finite and aligned with the surface rows")
    if not len(surfaces):
        return np.empty(queries.shape)
    return np.asarray([np.interp(q, s[:, 0], s[:, 1]) for q, s in zip(queries, surfaces, strict=True)])


def validate_surface_batches(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Same shared-physical-x polyline validity without constructing union grids.

    The difference of two linear polylines reaches its extrema at a knot of
    either surface. Its integral equals the difference of their integrals.
    Borderline cancellation uses the original union-knot routine explicitly.
    """
    upper, lower = np.asarray(upper, float), np.asarray(lower, float)
    if upper.ndim != 3 or upper.shape != lower.shape or upper.shape[2] != 2 or upper.shape[1] < 2:
        raise ValueError("Expected equally shaped batches of two-dimensional surfaces")
    valid = np.isfinite(upper).all(axis=(1, 2)) & np.isfinite(lower).all(axis=(1, 2))
    valid &= np.all(np.diff(upper[:, :, 0], axis=1) > 0, axis=1)
    valid &= np.all(np.diff(lower[:, :, 0], axis=1) > 0, axis=1)
    valid &= np.all(np.isclose(upper[:, [0, -1], 0], lower[:, [0, -1], 0], atol=1e-10, rtol=0), axis=1)
    selected = np.flatnonzero(valid)
    if len(selected):
        top, bottom = upper[selected], lower[selected]
        on_top = top[:, :, 1] - interpolate_surface_batches(top[:, :, 0], bottom)
        on_bottom = interpolate_surface_batches(bottom[:, :, 0], top) - bottom[:, :, 1]
        minimum = np.minimum(np.min(on_top, axis=1), np.min(on_bottom, axis=1))
        area = np.trapezoid(top[:, :, 1], top[:, :, 0], axis=1) - np.trapezoid(bottom[:, :, 1], bottom[:, :, 0], axis=1)
        accepted = (minimum >= -1e-8) & (area > 0)
        borderline = (abs(minimum + 1e-8) < 1e-11) | (abs(area) < 1e-11)
        borderline |= np.any(top[:, [0, -1], 0] != bottom[:, [0, -1], 0], axis=1)
        for j in np.flatnonzero(borderline):
            accepted[j] = validate_geometry(top[j], bottom[j]).valid
        valid[selected] = accepted
    return valid
