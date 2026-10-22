from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(frame: pd.DataFrame, prediction_columns: dict[str, str]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for target, prediction in prediction_columns.items():
        mask = frame[f"mask_{target}"].astype(bool) & frame[target].notna() & frame[prediction].notna()
        actual = frame.loc[mask, target].to_numpy(float)
        predicted = frame.loc[mask, prediction].to_numpy(float)
        if not len(actual):
            results[target] = {"count": 0, "mae": float("nan"), "mse": float("nan"), "r2": float("nan"), "macro_mse": float("nan")}
            continue
        per_airfoil = frame.loc[mask].assign(squared_error=(actual - predicted) ** 2).groupby("airfoil_id")["squared_error"].mean()
        results[target] = {
            "count": int(len(actual)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "mse": float(mean_squared_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
            "macro_mse": float(per_airfoil.mean()),
        }
    return results


def cvar(values: np.ndarray, quantile: float = 0.95) -> float:
    """Exact upper-tail CVaR of an equally weighted empirical distribution.

    Fractional boundary probability is retained. Averaging all values >= the
    interpolated sample quantile is not equivalent for atoms or small samples.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("CVaR requires a nonempty finite one-dimensional sample")
    if not np.isfinite(quantile) or not 0 <= quantile < 1:
        raise ValueError("CVaR quantile must be in [0, 1)")
    tail_mass = len(values) * (1.0 - quantile)
    descending = np.sort(values)[::-1]
    whole = int(np.floor(tail_mass))
    fractional = tail_mass - whole
    tail_sum = float(descending[:whole].sum())
    if fractional > 0 and whole < len(values):
        tail_sum += fractional * float(descending[whole])
    return tail_sum / tail_mass
