"""Connect tested batched polyline checks to the actual prepared-scenario evaluator."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
p=ROOT/'src/robust_airfoil/optimisation/candidate_uncertainty.py'
t=p.read_text(encoding='utf-8')
assert 'def interpolate_surface_batches(' not in t
(Path(__file__).parent/'candidate_uncertainty_before_batch_geometry.py.txt').write_text(t,encoding='utf-8')
start=t.index('    def evaluate(self, parameters: np.ndarray) -> CandidateScenarios:',t.index('class PreparedCandidateScenarios:'))
t=t[:start]+'''    def evaluate(self, parameters: np.ndarray) -> CandidateScenarios:
        candidate = np.asarray(parameters, float)
        upper, lower = self.displaced_surfaces(candidate)
        result = np.tile(candidate, (len(self.latent), 1))
        valid = validate_surface_batches(upper, lower)
        reasons = ["" if value else "invalid_displaced_geometry" for value in valid]
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
        for j in np.flatnonzero(~regular):
            ups[j] = self._resample_valid(upper[retained[j]])
            lows[j] = self._resample_valid(lower[retained[j]])
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
        for j, i in enumerate(retained):
            if not fit_finite[j]:
                reasons[i] = "nonfinite_refit"
            elif not fit_error_ok[j]:
                reasons[i] = "excessive_refit_error"
            elif not dense_valid[j]:
                reasons[i] = "invalid_refitted_geometry:surface_crossing_or_nonpositive_area"
            elif not dense_error_ok[j]:
                reasons[i] = "excessive_dense_refit_error"
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
        for j in np.flatnonzero(borderline):
            accepted[j] = validate_geometry(top[j], bottom[j]).valid
        valid[selected] = accepted
    return valid
'''
p.write_text(t,encoding='utf-8')
print('BATCHED_GEOMETRY_CONNECTED')
