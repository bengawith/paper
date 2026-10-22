"""Order-invariant agreement diagnostics for feasible nondominated fronts.

Inputs must already have been evaluated under one frozen objective, sample, and
trust context. This utility measures agreement; it does not certify convergence
or global optimality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist


def front_agreement(
    fronts: list[np.ndarray],
    objective_scales: np.ndarray,
    threshold: float = 0.20,
    minimum_feasible_fraction: float = 0.80,
) -> dict[str, Any]:
    """Compute symmetric nearest-front agreement without discarding failed runs."""
    scales = np.asarray(objective_scales, dtype=float)
    if (
        scales.ndim != 1
        or not len(scales)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("Objective scales must be predeclared, finite, and strictly positive")
    if (
        not fronts
        or not 0 <= minimum_feasible_fraction <= 1
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("Invalid agreement parameters")

    valid: list[np.ndarray] = []
    for front in fronts:
        values = np.asarray(front, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(scales):
            raise ValueError("Front shape is inconsistent with objective scales")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite front points must be explicitly classified upstream")
        if len(values):
            valid.append(np.unique(values, axis=0) / scales)

    distances: list[float] = []
    hausdorff: list[float] = []
    for index, left in enumerate(valid):
        for right in valid[index + 1 :]:
            pair = cdist(left, right)
            distances.append(float(0.5 * (pair.min(axis=1).mean() + pair.min(axis=0).mean())))
            hausdorff.append(float(max(pair.min(axis=1).max(), pair.min(axis=0).max())))

    mean_distance = float(np.mean(distances)) if distances else None
    feasible_fraction = len(valid) / len(fronts)
    return {
        "method": "symmetric_mean_nearest_front_distance_v1",
        "run_count": len(fronts),
        "feasible_run_count": len(valid),
        "feasible_fraction": feasible_fraction,
        "objective_scales": scales.tolist(),
        "pairwise_symmetric_distances": distances,
        "maximum_symmetric_hausdorff": max(hausdorff) if hausdorff else None,
        "mean_symmetric_distance": mean_distance,
        "threshold": threshold,
        "minimum_feasible_fraction": minimum_feasible_fraction,
        "passed": (
            mean_distance is not None
            and feasible_fraction >= minimum_feasible_fraction
            and mean_distance <= threshold
        ),
    }
