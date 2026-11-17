"""Calibration-only residual-risk ranking with a portable, frozen linear artifact.

This is a research ranking model, not an aerodynamic predictor, a coverage
certificate or a replacement for the operational empirical screening policy.
Confirmation outcomes never alter its target scales, cutoffs or fitted weights.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from robust_airfoil.constants import TARGET_COLUMNS

FEATURE_SETS = {
    'support_only': ['support_distance'],
    'ensemble_only': ['ensemble_std_' + target for target in TARGET_COLUMNS],
    'support_plus_ensemble': ['support_distance'] + ['ensemble_std_' + target for target in TARGET_COLUMNS],
}


def _feature_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not set(columns).issubset(frame.columns):
        raise ValueError('Missing frozen residual-risk input features')
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Residual-risk features must be finite and nonnegative')
    return values


def _absolute_errors(frame: pd.DataFrame) -> np.ndarray:
    required = [*TARGET_COLUMNS, *['prediction_' + target for target in TARGET_COLUMNS]]
    if not set(required).issubset(frame.columns):
        raise ValueError('Residual evaluation requires actual and predicted aerodynamic values')
    values = np.column_stack([abs(frame[target].to_numpy(float) - frame['prediction_' + target].to_numpy(float)) for target in TARGET_COLUMNS])
    if not np.isfinite(values).all():
        raise ValueError('Residual labels must be finite; missing solver observations are not imputed')
    return values


def fit_residual_risk(frame: pd.DataFrame, ridge_alpha: float = 10.0) -> dict[str, Any]:
    """Fit all ranking inputs and target definitions on calibration observations."""
    if 'split' not in frame or not frame['split'].eq('calibration').all():
        raise ValueError('Only calibration rows may fit the residual-risk model')
    if len(frame) < 2 or not np.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError('At least two calibration rows and nonnegative finite ridge alpha are required')
    masks = ['mask_' + target for target in TARGET_COLUMNS]
    if not set(masks).issubset(frame.columns) or not frame[masks].astype(bool).all().all():
        raise ValueError('Use only complete observed calibration target rows; do not fill missing labels')
    features = _feature_matrix(frame, FEATURE_SETS['support_plus_ensemble'])
    del features  # Validation only; each model below uses its declared feature subset.
    errors = _absolute_errors(frame)
    scales = np.maximum(np.quantile(errors, .9, axis=0), 1e-8)
    score = np.max(errors / scales, axis=1)
    high_error_threshold = float(np.quantile(score, .9))
    models: dict[str, Any] = {}
    for name, columns in FEATURE_SETS.items():
        raw = _feature_matrix(frame, columns)
        scaler = StandardScaler().fit(raw)
        model = Ridge(alpha=ridge_alpha).fit(scaler.transform(raw), np.log1p(score))
        predictions = model.predict(scaler.transform(raw))
        models[name] = {
            'features': list(columns),
            'feature_center': scaler.mean_.tolist(),
            'feature_scale': scaler.scale_.tolist(),
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'calibration_retention_thresholds': {str(percent): float(np.quantile(predictions, percent / 100)) for percent in range(10, 101, 10)},
        }
    return {
        'schema': 'frozen-calibration-linear-residual-risk-v1',
        'training_partition': 'calibration',
        'training_rows': len(frame),
        'training_airfoil_count': int(frame.airfoil_id.nunique()) if 'airfoil_id' in frame else None,
        'target_order': list(TARGET_COLUMNS),
        'training_residual_p90_scales': scales.tolist(),
        'training_high_error_threshold': high_error_threshold,
        'training_high_error_fraction': float(np.mean(score >= high_error_threshold)),
        'ridge_alpha': float(ridge_alpha),
        'models': models,
        'policy_status': 'Research ranking only; no operational acceptance rule is changed',
        'confirmation_labels_used_for_fit': False,
    }


def predict_residual_risk(frame: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    """Score only the frozen observable inputs, without reading target labels."""
    if artifact.get('schema') != 'frozen-calibration-linear-residual-risk-v1':
        raise ValueError('Unknown residual-risk artifact schema')
    predictions = {}
    for name, spec in artifact['models'].items():
        raw = _feature_matrix(frame, spec['features'])
        center = np.asarray(spec['feature_center'], float)
        scale = np.asarray(spec['feature_scale'], float)
        coefficient = np.asarray(spec['coefficients'], float)
        if center.shape != (raw.shape[1],) or center.shape != scale.shape or center.shape != coefficient.shape:
            raise ValueError('Risk coefficient/scaler shape mismatch')
        if not np.isfinite(np.r_[center, scale, coefficient, spec['intercept']]).all() or np.any(scale <= 0):
            raise ValueError('Frozen risk coefficients/scales must be finite with positive scales')
        predictions[name] = ((raw - center) / scale) @ coefficient + float(spec['intercept'])
    return predictions


def label_residual_risk(frame: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    """Evaluate the already-frozen high-error definition, never refit on query errors."""
    if artifact.get('target_order') != list(TARGET_COLUMNS):
        raise ValueError('Frozen residual target ordering differs')
    scales = np.asarray(artifact['training_residual_p90_scales'], float)
    if scales.shape != (len(TARGET_COLUMNS),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Frozen residual scales must be positive and finite')
    scores = np.max(_absolute_errors(frame) / scales, axis=1)
    return {'scaled_error': scores, 'high_error': scores >= float(artifact['training_high_error_threshold'])}
