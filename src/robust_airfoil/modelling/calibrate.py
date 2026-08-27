from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_airfoil.modelling.dataset import FEATURE_COLUMNS, TARGET_COLUMNS, ScalingBundle


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(finite):
        raise ValueError("Cannot calibrate a conformal quantile without finite residuals")
    rank = min(len(finite), int(np.ceil((len(finite) + 1) * coverage)))
    return float(finite[rank - 1])


def calibrate_residual_quantiles(predictions: pd.DataFrame, output_path: Path, quantiles: tuple[float, ...] = (0.5, 0.9, 0.95)) -> dict[str, object]:
    targets: dict[str, dict[str, float]] = {}
    for target in ("cl", "log_cd", "cm"):
        mask = predictions[f"mask_{target}"].astype(bool)
        residuals = np.abs(predictions.loc[mask, target] - predictions.loc[mask, f"prediction_{target}"])
        targets[target] = {str(q): float(np.quantile(residuals, q)) for q in quantiles}
    result: dict[str, object] = {"method": "empirical_absolute_residual_quantiles", "targets": targets}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def support_distance(train_features: np.ndarray, query_features: np.ndarray, k: int = 5) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    neighbours = NearestNeighbors(n_neighbors=min(k, len(train_features))).fit(train_features)
    distances, _ = neighbours.kneighbors(query_features)
    return distances.mean(axis=1)


def calibrate_trust_model(
    development_points: pd.DataFrame,
    calibration_predictions: pd.DataFrame,
    scaling: ScalingBundle,
    output_path: Path,
) -> dict[str, object]:
    development_features = (development_points[FEATURE_COLUMNS].to_numpy(float) - scaling.feature_center) / scaling.feature_scale
    calibration_features = (calibration_predictions[FEATURE_COLUMNS].to_numpy(float) - scaling.feature_center) / scaling.feature_scale
    distances = support_distance(development_features, calibration_features)
    calibration_predictions["support_distance"] = distances
    residuals: dict[str, dict[str, float]] = {}
    disagreement: dict[str, dict[str, float]] = {}
    empirical_coverage: dict[str, float] = {}
    trusted = distances <= float(np.quantile(distances, 0.99))
    for target in TARGET_COLUMNS:
        mask = calibration_predictions[f"mask_{target}"].astype(bool).to_numpy()
        absolute = np.abs(
            calibration_predictions.loc[mask, target].to_numpy(float)
            - calibration_predictions.loc[mask, f"prediction_{target}"].to_numpy(float)
        )
        residuals[target] = {str(q): conformal_quantile(absolute, q) for q in (0.5, 0.9, 0.95, 0.99)}
        spread = calibration_predictions.loc[mask, f"ensemble_std_{target}"].to_numpy(float)
        disagreement[target] = {str(q): conformal_quantile(spread, q) for q in (0.5, 0.9, 0.95, 0.99)}
        residual_tolerance = residuals[target]["0.95"]
        ordered = np.argsort(spread)
        candidate_thresholds: list[float] = []
        for fraction in np.linspace(0.2, 1.0, 9):
            count = max(1, int(np.floor(fraction * len(ordered))))
            selected = ordered[:count]
            if conformal_quantile(absolute[selected], 0.95) <= residual_tolerance:
                candidate_thresholds.append(float(np.max(spread[selected])))
        threshold = max(candidate_thresholds) if candidate_thresholds else disagreement[target]["0.5"]
        disagreement[target]["trusted_threshold_from_residual_tolerance"] = threshold
        target_trusted = calibration_predictions[f"ensemble_std_{target}"].to_numpy(float) <= threshold
        trusted &= target_trusted
        empirical_coverage[target] = float(np.mean(absolute <= residuals[target]["0.95"]))
    alpha_min = float(development_points["alpha_deg"].min())
    alpha_max = float(development_points["alpha_deg"].max())
    trusted &= calibration_predictions["alpha_deg"].between(alpha_min, alpha_max).to_numpy()
    calibration_predictions["trusted_domain"] = trusted
    source_tiers = sorted(development_points["source_tier"].dropna().astype(str).unique().tolist()) if "source_tier" in development_points else []
    support_tolerance = max(residuals[target]["0.95"] for target in TARGET_COLUMNS)
    support_thresholds: list[float] = []
    aggregate_scaled_residual = np.zeros(len(calibration_predictions), dtype=float)
    for target in TARGET_COLUMNS:
        mask = calibration_predictions[f"mask_{target}"].astype(bool).to_numpy()
        residual = np.abs(
            calibration_predictions[target].to_numpy(float)
            - calibration_predictions[f"prediction_{target}"].to_numpy(float)
        )
        residual[~mask] = 0.0
        aggregate_scaled_residual = np.maximum(aggregate_scaled_residual, residual / max(residuals[target]["0.95"], 1e-12))
    ordered_support = np.argsort(distances)
    for fraction in np.linspace(0.2, 1.0, 9):
        count = max(1, int(np.floor(fraction * len(ordered_support))))
        selected = ordered_support[:count]
        if conformal_quantile(aggregate_scaled_residual[selected], 0.95) <= 1.0:
            support_thresholds.append(float(np.max(distances[selected])))
    trusted_support_threshold = max(support_thresholds) if support_thresholds else float(np.quantile(distances, 0.5))
    trusted &= distances <= trusted_support_threshold
    payload: dict[str, object] = {
        "method": "split_conformal_residual_band_with_empirical_trust_thresholds",
        "residual_absolute_quantiles": residuals,
        "ensemble_disagreement_quantiles": disagreement,
        "support_distance_quantiles": {
            **{str(q): float(np.quantile(distances, q)) for q in (0.5, 0.9, 0.95, 0.99)},
            "trusted_threshold_from_residual_tolerance": trusted_support_threshold,
        },
        "alpha_coverage_deg": [alpha_min, alpha_max],
        "target_coverage": {
            target: [float(development_points.loc[development_points[f"mask_{target}"].astype(bool), target].min()), float(development_points.loc[development_points[f"mask_{target}"].astype(bool), target].max())]
            for target in TARGET_COLUMNS
        },
        "source_tiers": source_tiers,
        "empirical_95_interval_coverage": empirical_coverage,
        "calibration_points": len(calibration_predictions),
        "trusted_calibration_fraction": float(np.mean(trusted)),
        "residual_tolerance_reference": support_tolerance,
        "uncertainty_terms": ["ensemble disagreement", "empirical predictive uncertainty", "calibrated residual band"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def apply_trust_model(
    development_points: pd.DataFrame,
    predictions: pd.DataFrame,
    scaling: ScalingBundle,
    calibration: dict[str, Any],
) -> pd.DataFrame:
    result = predictions.copy()
    development_features = (development_points[FEATURE_COLUMNS].to_numpy(float) - scaling.feature_center) / scaling.feature_scale
    query_features = (result[FEATURE_COLUMNS].to_numpy(float) - scaling.feature_center) / scaling.feature_scale
    result["support_distance"] = support_distance(development_features, query_features)
    support_quantiles = calibration["support_distance_quantiles"]
    disagreement_quantiles = calibration["ensemble_disagreement_quantiles"]
    alpha_min, alpha_max = calibration["alpha_coverage_deg"]
    trusted = result["support_distance"].to_numpy(float) <= float(support_quantiles["trusted_threshold_from_residual_tolerance"])
    trusted &= result["alpha_deg"].between(float(alpha_min), float(alpha_max)).to_numpy()
    for target in TARGET_COLUMNS:
        trusted &= result[f"ensemble_std_{target}"].to_numpy(float) <= float(
            disagreement_quantiles[target]["trusted_threshold_from_residual_tolerance"]
        )
    result["trusted_domain"] = trusted
    return result
