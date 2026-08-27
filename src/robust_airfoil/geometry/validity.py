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


def validate_geometry(upper: np.ndarray, lower: np.ndarray, minimum_local_thickness: float = 0.0) -> GeometryValidity:
    if upper.shape != lower.shape or upper.shape[1] != 2:
        return GeometryValidity(False, ("surface_shape_mismatch",), float("nan"), float("nan"), float("nan"))
    thickness = upper[:, 1] - lower[:, 1]
    reasons: list[str] = []
    if not np.isfinite(upper).all() or not np.isfinite(lower).all():
        reasons.append("non_finite")
    if np.any(np.diff(upper[:, 0]) < 0) or np.any(np.diff(lower[:, 0]) < 0):
        reasons.append("non_monotonic_x")
    if float(np.min(thickness)) < minimum_local_thickness - 1e-8:
        reasons.append("surface_crossing")
    area = float(np.trapezoid(thickness, upper[:, 0]))
    if area <= 0:
        reasons.append("non_positive_area")
    return GeometryValidity(not reasons, tuple(reasons), float(np.min(thickness)), float(np.max(thickness)), area)
