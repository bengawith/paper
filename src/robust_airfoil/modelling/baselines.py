from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from robust_airfoil.modelling.dataset import FEATURE_COLUMNS, TARGET_COLUMNS
from robust_airfoil.modelling.evaluate import regression_metrics


@dataclass
class BaselineResult:
    metrics: dict[str, dict[str, dict[str, float]]]
    predictions: pd.DataFrame


def fit_baselines(train: pd.DataFrame, validation: pd.DataFrame, seed: int = 20260824) -> BaselineResult:
    output = validation[["airfoil_id", *TARGET_COLUMNS, *[f"mask_{name}" for name in TARGET_COLUMNS]]].copy()
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    constructors = {
        "dummy": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: make_pipeline(RobustScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=200, random_state=seed),
    }
    for model_name, constructor in constructors.items():
        columns: dict[str, str] = {}
        for target in TARGET_COLUMNS:
            mask = train[f"mask_{target}"].astype(bool)
            model = constructor()
            model.fit(train.loc[mask, FEATURE_COLUMNS], train.loc[mask, target])
            column = f"prediction_{model_name}_{target}"
            output[column] = model.predict(validation[FEATURE_COLUMNS])
            columns[target] = column
        metrics[model_name] = regression_metrics(output, columns)
    return BaselineResult(metrics, output)
