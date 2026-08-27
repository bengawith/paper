from __future__ import annotations

import numpy as np

from robust_airfoil.modelling.evaluate import cvar


def required_lift_drag(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray, cl_required: float) -> float:
    finite = np.isfinite(alpha) & np.isfinite(cl) & np.isfinite(cd)
    if int(finite.sum()) < 2:
        return float("inf")
    order = np.argsort(alpha[finite])
    ordered_alpha = alpha[finite][order]
    ordered_cl = cl[finite][order]
    ordered_cd = cd[finite][order]
    unique_alpha, unique_index = np.unique(ordered_alpha, return_index=True)
    ordered_cl = ordered_cl[unique_index]
    ordered_cd = ordered_cd[unique_index]
    crossings: list[tuple[float, float]] = []
    for index in range(len(unique_alpha) - 1):
        cl_left = ordered_cl[index]
        cl_right = ordered_cl[index + 1]
        if cl_right <= cl_left or not (cl_left <= cl_required <= cl_right):
            continue
        fraction = (cl_required - cl_left) / (cl_right - cl_left)
        crossing_alpha = unique_alpha[index] + fraction * (
            unique_alpha[index + 1] - unique_alpha[index]
        )
        crossing_cd = ordered_cd[index] + fraction * (ordered_cd[index + 1] - ordered_cd[index])
        if not crossings or not np.isclose(crossing_alpha, crossings[-1][0], atol=1e-10):
            crossings.append((float(crossing_alpha), float(crossing_cd)))
    if len(crossings) != 1:
        return float("inf")
    return crossings[0][1]


def weighted_required_lift_drag(alpha: np.ndarray, cl: np.ndarray, cd: np.ndarray, required: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray([required_lift_drag(alpha, cl, cd, target) for target in required])
    return float(np.sum(values * weights)) if np.isfinite(values).all() else float("inf")


def robust_drag_objectives(sample_weighted_drags: np.ndarray) -> dict[str, float]:
    finite = np.asarray(sample_weighted_drags, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"expected_weighted_cd": float("inf"), "cvar_95_weighted_cd": float("inf")}
    return {"expected_weighted_cd": float(finite.mean()), "cvar_95_weighted_cd": cvar(finite, 0.95)}


def ava_lift_sensitivity(mean_lift: np.ndarray, lift_std: np.ndarray, weights: np.ndarray, sensitivity_lambda: float) -> float:
    scale = max(float(np.mean(np.abs(mean_lift))), 1e-9)
    return float(-np.sum(weights * mean_lift) / scale + sensitivity_lambda * np.sum(weights * lift_std) / scale)
