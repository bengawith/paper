from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesignBounds:
    lower: np.ndarray
    upper: np.ndarray


def development_bounds(parameters: np.ndarray, padding_fraction: float = 0.05) -> DesignBounds:
    low = np.quantile(parameters, 0.01, axis=0)
    high = np.quantile(parameters, 0.99, axis=0)
    span = np.maximum(high - low, 1e-6)
    return DesignBounds(low - padding_fraction * span, high + padding_fraction * span)
