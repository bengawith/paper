from __future__ import annotations

import numpy as np

from robust_airfoil.modelling.evaluate import cvar


def required_lift_drag(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray, cl_required: float) -> float:
    """Interpolate only on the initial strictly rising branch in alpha order.

    The caller must supply a supported operating interval. A gap represented
    by NaN, conflicting duplicate alpha, later re-rising branch, nonpositive
    drag, or ambiguous rising crossings is infeasible, not extrapolated.
    This numerical check is not a physical proof that flow is attached.
    """
    alpha, cl, cd = (np.asarray(v, dtype=float) for v in (alpha, cl, cd))
    if any(v.ndim != 1 for v in (alpha, cl, cd)) or not (len(alpha) == len(cl) == len(cd)):
        raise ValueError("alpha, cl and cd must be equal-length one-dimensional arrays")
    if len(alpha) < 2 or not np.isfinite(cl_required):
        return float("inf")
    if not all(np.isfinite(v).all() for v in (alpha, cl, cd)) or np.any(cd <= 0):
        return float("inf")
    order = np.argsort(alpha, kind="stable")
    alpha, cl, cd = alpha[order], cl[order], cd[order]
    unique, starts, counts = np.unique(alpha, return_index=True, return_counts=True)
    for start, count in zip(starts, counts, strict=True):
        if count > 1 and (
            not np.allclose(cl[start:start + count], cl[start], rtol=0, atol=1e-12)
            or not np.allclose(cd[start:start + count], cd[start], rtol=0, atol=1e-12)
        ):
            return float("inf")
    alpha, cl, cd = unique, cl[starts], cd[starts]
    if len(alpha) < 2:
        return float("inf")
    changes = np.diff(cl)
    non_rising = np.flatnonzero(changes <= 1e-12)
    initial_end = int(non_rising[0]) if len(non_rising) else len(alpha) - 1
    crossings: list[tuple[float, float, int]] = []
    for i in range(len(alpha) - 1):
        if changes[i] <= 1e-12 or not (cl[i] <= cl_required <= cl[i + 1]):
            continue
        fraction = (cl_required - cl[i]) / changes[i]
        crossing_alpha = float(alpha[i] + fraction * (alpha[i + 1] - alpha[i]))
        crossing_cd = float(cd[i] + fraction * (cd[i + 1] - cd[i]))
        if not crossings or not np.isclose(crossing_alpha, crossings[-1][0], rtol=0, atol=1e-10):
            crossings.append((crossing_alpha, crossing_cd, i))
    if len(crossings) != 1 or crossings[0][2] >= initial_end:
        return float("inf")
    return crossings[0][1]


def weighted_required_lift_drag(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray, required: np.ndarray, weights: np.ndarray) -> float:
    required, weights = np.asarray(required, dtype=float), np.asarray(weights, dtype=float)
    if required.ndim != 1 or required.shape != weights.shape or not len(required):
        raise ValueError("Required lift and weights must be equal nonempty vectors")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and have positive sum")
    values = np.asarray([required_lift_drag(alpha, cl, cd, target) for target in required])
    return float(np.sum(values * weights / weights.sum())) if np.isfinite(values).all() else float("inf")


def robust_drag_objectives(sample_weighted_drags: np.ndarray) -> dict[str, float]:
    values = np.asarray(sample_weighted_drags, dtype=float)
    if values.ndim != 1:
        raise ValueError("Risk samples must be a one-dimensional vector")
    # Failure-aware constraint handling belongs to the caller. Never condition
    # reported aerodynamic risk silently on the successful subset.
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        return {"expected_weighted_cd": float("inf"), "cvar_95_weighted_cd": float("inf")}
    return {"expected_weighted_cd": float(values.mean()), "cvar_95_weighted_cd": cvar(values, 0.95)}


def ava_lift_sensitivity(mean_lift: np.ndarray, lift_std: np.ndarray, weights: np.ndarray, sensitivity_lambda: float, *, lift_scale: float = 1.0, sensitivity_scale: float = 1.0) -> float:
    """Mean-lift/sensitivity objective with candidate-independent fixed scales.

    CL is dimensionless, so the default unit scales are meaningful. A study
    may supply other positive scales, frozen from its reference/development
    set before search. Never scale by each candidate's own mean lift.
    """
    mean_lift, lift_std, weights = (np.asarray(v, dtype=float) for v in (mean_lift, lift_std, weights))
    if mean_lift.ndim != 1 or mean_lift.shape != lift_std.shape or mean_lift.shape != weights.shape or not len(mean_lift):
        raise ValueError("Lift, deviation and weights must be equal nonempty vectors")
    if not all(np.isfinite(v).all() for v in (mean_lift, lift_std, weights)):
        return float("inf")
    if np.any(lift_std < 0) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid standard deviations or weights")
    if not all(np.isfinite(v) for v in (sensitivity_lambda, lift_scale, sensitivity_scale)) or sensitivity_lambda < 0 or min(lift_scale, sensitivity_scale) <= 0:
        raise ValueError("Objective scales must be positive and lambda nonnegative")
    weights = weights / weights.sum()
    return float(-np.sum(weights * mean_lift) / lift_scale + sensitivity_lambda * np.sum(weights * lift_std) / sensitivity_scale)
