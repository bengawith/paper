"""Auditable prediction metrics with explicit physical units and resampling units.

Geometry bootstrap summaries condition on observed numerical labels and hold
all fitted models, screens and predictions fixed. They are not simultaneous
coverage proofs, complete refitting uncertainty or manufacturing-population CIs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from robust_airfoil.constants import TARGET_COLUMNS


def _target_rows(frame: pd.DataFrame, target: str, rule: dict[str, Any]) -> pd.DataFrame:
    base = 'log_cd' if target == 'cd' else target
    required = ['airfoil_id',base,'prediction_'+base,'mask_'+base]
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing observed/predicted target accounting columns')
    selected = frame.loc[frame['mask_'+base].astype(bool)].copy()
    actual = selected[base].to_numpy(float)
    predicted = selected['prediction_'+base].to_numpy(float)
    width = float(rule['residual_absolute_quantiles'][base]['0.95'])
    if not np.isfinite(width) or width < 0:
        raise ValueError('Prediction-band width must be finite and nonnegative')
    lower, upper = predicted-width, predicted+width
    if target == 'cd':
        with np.errstate(over='ignore',invalid='ignore'):
            actual,predicted,lower,upper = np.exp(actual),np.exp(predicted),np.exp(lower),np.exp(upper)
    finite = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(lower) & np.isfinite(upper)
    selected = selected.loc[finite]
    actual,predicted,lower,upper = actual[finite],predicted[finite],lower[finite],upper[finite]
    return pd.DataFrame({'airfoil_id':selected.airfoil_id.astype(str).to_numpy(),
                         'actual':actual,'predicted':predicted,'abs_error':abs(actual-predicted),
                         'squared_error':(actual-predicted)**2,'actual_squared':actual**2,
                         'covered':(actual>=lower)&(actual<=upper)})


def prediction_metrics(frame: pd.DataFrame, rule: dict[str, Any]) -> list[dict[str, Any]]:
    """Report point, equal-geometry and all-observed-polar statistics separately."""
    if 'trusted_domain' not in frame or frame['trusted_domain'].isna().any():
        raise ValueError('Every evaluated state must have an explicit frozen-screen decision')
    records = []
    trusted = frame['trusted_domain'].astype(bool)
    for subset, values in [('all',frame),('accepted',frame.loc[trusted]),('rejected',frame.loc[~trusted])]:
        for target in [*TARGET_COLUMNS,'cd']:
            rows = _target_rows(values,target,rule)
            record: dict[str, Any] = {'subset':subset,'target':target,'states':len(rows),
                                      'airfoils':int(rows.airfoil_id.nunique()),'mae':None,'mse':None,'r2':None,
                                      'geometry_macro_mae':None,'geometry_macro_mse':None,
                                      'state_interval_coverage':None,'geometry_macro_interval_coverage':None,
                                      'whole_observed_curve_coverage':None}
            if len(rows):
                grouped = rows.groupby('airfoil_id',sort=True)
                denominator = float(np.sum((rows.actual.to_numpy()-rows.actual.mean())**2))
                record.update(
                    mae=float(rows.abs_error.mean()),mse=float(rows.squared_error.mean()),
                    r2=1.0-float(rows.squared_error.sum())/denominator if len(rows)>1 and denominator>0 else None,
                    geometry_macro_mae=float(grouped.abs_error.mean().mean()),
                    geometry_macro_mse=float(grouped.squared_error.mean().mean()),
                    state_interval_coverage=float(rows.covered.mean()),
                    geometry_macro_interval_coverage=float(grouped.covered.mean().mean()),
                    whole_observed_curve_coverage=float(grouped.covered.all().mean()),
                )
            records.append(record)
    return records


def geometry_bootstrap_metrics(frame: pd.DataFrame, rule: dict[str, Any], *,
                               replicates: int = 1000, seed: int = 20261020) -> dict[str, Any]:
    """Resample complete geometry groups, never independent incidence rows."""
    if replicates < 1 or frame.empty or 'airfoil_id' not in frame:
        raise ValueError('Geometry bootstrap requires observations and a positive replication count')
    units = sorted(frame.airfoil_id.astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(units),np.full(len(units),1./len(units)),size=replicates)
    targets: dict[str, Any] = {}
    for target in [*TARGET_COLUMNS,'cd']:
        rows = _target_rows(frame,target,rule)
        grouped = rows.groupby('airfoil_id',sort=True)
        means = grouped[['abs_error','squared_error','covered']].mean().reindex(units)
        sums = grouped[['abs_error','squared_error','actual','actual_squared']].sum().reindex(units).fillna(0.)
        counts = grouped.size().reindex(units).fillna(0.).to_numpy(float)
        available = np.isfinite(means.abs_error.to_numpy(float))
        target_count = int(available.sum())
        if not target_count:
            targets[target] = {'observed_states':0,'observed_geometries':0}
            continue
        count_weight = weights @ counts
        geometry_weight = weights @ available.astype(float)
        good = (count_weight>0)&(geometry_weight>0)
        macro_mae = np.divide(weights @ means.abs_error.fillna(0.).to_numpy(float),geometry_weight,
                              out=np.full(replicates,np.nan),where=geometry_weight>0)
        macro_coverage = np.divide(weights @ means.covered.fillna(0.).to_numpy(float),geometry_weight,
                                   out=np.full(replicates,np.nan),where=geometry_weight>0)
        micro_mae = np.divide(weights @ sums.abs_error.to_numpy(float),count_weight,
                              out=np.full(replicates,np.nan),where=count_weight>0)
        def interval(values: np.ndarray, valid_mask: np.ndarray = good) -> list[float] | None:
            retained = values[valid_mask & np.isfinite(values)]
            return np.quantile(retained,[.025,.975]).tolist() if len(retained) else None
        targets[target] = {
            'observed_states':len(rows),'observed_geometries':target_count,
            'geometry_macro_mae_point':float(means.abs_error.mean()),
            'geometry_macro_mae_interval':interval(macro_mae),
            'geometry_macro_coverage_point':float(means.covered.mean()),
            'geometry_macro_coverage_interval':interval(macro_coverage),
            'point_mae_point':float(rows.abs_error.mean()),'point_mae_interval':interval(micro_mae),
            'usable_bootstrap_replicates':int(good.sum()),
        }
    return {'schema':'fixed-model-whole-geometry-bootstrap-v1','resampling_units':len(units),
            'replicates':replicates,'seed':seed,'targets':targets,
            'scope':'Descriptive fixed-model resampling of complete geometry groups conditional on observed solver labels. Shared source parents and numerical selection may create further dependence; no manufacturing-population or complete-refitting confidence guarantee.'}
