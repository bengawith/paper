"""Geometry-only, reproducible preparation of unseen represented aerofoil cases.

No aerodynamic labels, trained model, residuals, trust thresholds or optimiser
outcomes are accepted as inputs. The chosen geometric sampling law is explicit;
it is not a random sample of manufactured aerofoils or unseen natural families.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.geometry.validity import validate_geometry


def build_confirmation_geometry(
    parameters: np.ndarray,
    development_indices: np.ndarray,
    cluster_ids: np.ndarray,
    count: int,
    seed: int,
    minimum_rms_separation: float,
    mixing_limits: tuple[float, float] = (.2, .8),
    maximum_attempts: int = 100000,
) -> dict[str, Any]:
    """Blend distinct development clusters, retaining only separated valid shapes.

    The same geometry-only admission rule applies to all attempts. A failed
    target count raises an error rather than relaxing separation or choosing a
    new seed. Parent rows refer to the supplied immutable source matrix.
    """
    values = np.asarray(parameters, dtype=float)
    development = np.asarray(development_indices)
    clusters = np.asarray(cluster_ids)
    if values.ndim != 2 or values.shape[1] != 12 or not np.isfinite(values).all():
        raise ValueError('Confirmation sources must be finite twelve-parameter CST vectors')
    if development.ndim != 1 or not np.issubdtype(development.dtype, np.integer):
        raise ValueError('Development indices must be a one-dimensional integer vector')
    if clusters.shape != (len(values),) or np.any(development < 0) or np.any(development >= len(values)):
        raise ValueError('Source cluster identities and development indices must align')
    if len(np.unique(development)) != len(development) or len(development) < 2:
        raise ValueError('At least two distinct development source rows are required')
    if len(np.unique(clusters[development])) < 2:
        raise ValueError('At least two distinct development geometry clusters are required')
    if count < 1 or maximum_attempts < count or minimum_rms_separation <= 0 or not np.isfinite(minimum_rms_separation):
        raise ValueError('Positive count, separation and sufficient attempt limit are required')
    low, high = mixing_limits
    if not np.isfinite([low, high]).all() or not 0 < low <= high < 1:
        raise ValueError('Mixing limits must lie strictly inside zero and one')
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    dense_x = (1 - np.cos(np.linspace(0, np.pi, 2001))) / 2
    source_vectors = np.asarray([np.concatenate(reconstruct_cst(row, x)) for row in values])
    prior = source_vectors.copy()
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    rejected = {'same_parent_cluster': 0, 'invalid_reconstructed_geometry': 0, 'insufficient_rms_separation': 0}
    for attempt in range(1, maximum_attempts + 1):
        first, second = rng.choice(development, size=2, replace=False)
        if clusters[first] == clusters[second]:
            rejected['same_parent_cluster'] += 1
            continue
        weight = float(rng.uniform(low, high))
        candidate = weight * values[first] + (1.0 - weight) * values[second]
        top, bottom = reconstruct_cst(candidate, dense_x)
        if not validate_geometry(np.column_stack([dense_x, top]), np.column_stack([dense_x, bottom])).valid:
            rejected['invalid_reconstructed_geometry'] += 1
            continue
        vector = np.concatenate(reconstruct_cst(candidate, x))
        nearest = float(np.sqrt(np.mean((prior - vector) ** 2, axis=1)).min())
        if nearest <= minimum_rms_separation:
            rejected['insufficient_rms_separation'] += 1
            continue
        candidates.append({
            'confirmation_id': f'confirmation_{len(candidates):04d}',
            'attempt': attempt,
            'parent_indices': [int(first), int(second)],
            'first_parent_weight': weight,
            'parameters': candidate.tolist(),
            'minimum_rms_to_prior_shapes': nearest,
        })
        prior = np.vstack([prior, vector])
        if len(candidates) == count:
            return {
                'schema': 'geometry-only-confirmation-mixtures-v1',
                'requested_count': count, 'seed': seed, 'attempts': attempt,
                'minimum_rms_separation': minimum_rms_separation,
                'mixing_limits': [low, high], 'source_shape_count': len(values),
                'development_source_count': len(development),
                'rejected_attempts': rejected, 'candidates': candidates,
                'selection_scope': 'Geometry-only mixtures of distinct development clusters; no aerodynamic or trained-model outcomes used',
            }
    raise RuntimeError(f'Confirmation geometry admission/separation yielded {len(candidates)}/{count} after {maximum_attempts} fixed-seed attempts')
