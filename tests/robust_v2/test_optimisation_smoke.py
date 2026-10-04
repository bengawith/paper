import numpy as np

from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.optimisation.runner import deterministic_smoke_optimisation


def test_differential_evolution_smoke_is_deterministic():
    bounds = DesignBounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    first = deterministic_smoke_optimisation(objective, bounds, seed=8, maxiter=2)
    second = deterministic_smoke_optimisation(objective, bounds, seed=8, maxiter=2)
    assert first["parameters"] == second["parameters"]
    assert first["objective"] < 0.2
