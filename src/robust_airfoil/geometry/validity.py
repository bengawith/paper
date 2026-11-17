from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryValidity:
    valid: bool
    reasons: tuple[str, ...]
    minimum_thickness: float
    maximum_thickness: float
    section_area: float


def validate_geometry(
    upper: np.ndarray,
    lower: np.ndarray,
    minimum_local_thickness: float = 0.0,
) -> GeometryValidity:
    """Validate single-valued surface polylines at common physical x locations.

    Normal displacement moves upper and lower x coordinates differently.
    Subtracting corresponding y indices can therefore miss an intersection.
    The union of both knot sets captures extrema of their piecewise-linear
    thickness exactly, rather than certifying only the nominal index pairing.
    """
    upper, lower = np.asarray(upper), np.asarray(lower)
    unavailable = float('nan')
    if (
        upper.ndim != 2
        or lower.ndim != 2
        or upper.shape != lower.shape
        or upper.shape[1] != 2
        or len(upper) < 2
    ):
        return GeometryValidity(
            False, ('surface_shape_mismatch',), unavailable, unavailable, unavailable
        )
    if not np.isfinite(upper).all() or not np.isfinite(lower).all():
        return GeometryValidity(False, ('non_finite',), unavailable, unavailable, unavailable)
    if np.any(np.diff(upper[:, 0]) <= 0) or np.any(np.diff(lower[:, 0]) <= 0):
        return GeometryValidity(
            False, ('non_monotonic_x',), unavailable, unavailable, unavailable
        )
    if not np.allclose(upper[[0, -1], 0], lower[[0, -1], 0], atol=1e-10, rtol=0):
        return GeometryValidity(
            False, ('surface_extent_mismatch',), unavailable, unavailable, unavailable
        )
    x = np.union1d(upper[:, 0], lower[:, 0])
    thickness = np.interp(x, upper[:, 0], upper[:, 1]) - np.interp(
        x, lower[:, 0], lower[:, 1]
    )
    reasons: list[str] = []
    if float(np.min(thickness)) < minimum_local_thickness - 1e-8:
        reasons.append('surface_crossing')
    area = float(np.trapezoid(thickness, x))
    if area <= 0:
        reasons.append('non_positive_area')
    return GeometryValidity(
        not reasons, tuple(reasons), float(np.min(thickness)), float(np.max(thickness)), area
    )
