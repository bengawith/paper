from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc


def sobol_normal_samples(count: int, dimension: int, seed: int = 20260824, antithetic: bool = False) -> np.ndarray:
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol sample count must be a positive power of two")
    if antithetic:
        if count < 2:
            raise ValueError("Antithetic Sobol sampling requires count >= 2")
        base_count = count // 2
    else:
        base_count = count
    exponent = int(np.log2(base_count))
    uniform = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(exponent)
    uniform = np.clip(uniform, 1e-9, 1 - 1e-9)
    normal = norm.ppf(uniform)
    return np.vstack([normal, -normal]) if antithetic else normal
