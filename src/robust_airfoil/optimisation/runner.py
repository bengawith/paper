from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import differential_evolution

from robust_airfoil.optimisation.problems import DesignBounds


def deterministic_smoke_optimisation(objective: Callable[[np.ndarray], float], bounds: DesignBounds, seed: int = 20260824, maxiter: int = 10) -> dict[str, object]:
    result = differential_evolution(objective, list(zip(bounds.lower, bounds.upper, strict=True)), seed=seed, maxiter=maxiter, popsize=5, polish=False, workers=1, updating="immediate")
    return {"success": bool(result.success), "message": str(result.message), "objective": float(result.fun), "parameters": result.x.tolist(), "evaluations": int(result.nfev)}
