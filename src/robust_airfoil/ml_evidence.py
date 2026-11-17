"""Auditable ML-utility calculations, never aerodynamic labels or a trained model.

Selection does not inspect reference labels. Missing references remain in
coverage denominators, and bootstrap inputs must already be resolved paired
losses from independent geometry groups.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def _vector(values: Any, name: str, *, finite: bool = True) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains unresolved/nonfinite values")
    return result


def _groups(values: Sequence[str], count: int) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or len(raw) != count:
        raise ValueError("group IDs must be one-dimensional and aligned with requests")
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ValueError("group IDs must be nonempty strings, not missing values")
    return raw.astype(str)


def _mask(values: Any, count: int, name: str) -> np.ndarray:
    mask = np.asarray(values)
    if mask.ndim != 1 or len(mask) != count or mask.dtype.kind != "b":
        raise ValueError(f"{name} must be an aligned Boolean vector")
    return mask


def _same_length(count: int, **arrays: np.ndarray) -> None:
    for name, array in arrays.items():
        if len(array) != count:
            raise ValueError(f"{name} is not aligned with requests")


def group_balanced_weights(group_ids: Sequence[str]) -> np.ndarray:
    """Give each independent geometry group total weight 1 / number_of_groups."""
    groups = _groups(group_ids, len(group_ids))
    unique, inverse, group_counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * group_counts[inverse])


def selective_regression_report(
    reference: Sequence[float],
    prediction: Sequence[float],
    uncertainty: Sequence[float],
    group_ids: Sequence[str],
    accepted_by_policy: Sequence[bool],
    reference_complete: Sequence[bool],
    *,
    tolerance: float,
    requested_coverages: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0),
    interval_lower: Sequence[float] | None = None,
    interval_upper: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Calculate group-balanced risk/coverage diagnostics without label-based selection."""
    true = _vector(reference, "reference", finite=False)
    predicted = _vector(prediction, "prediction", finite=False)
    score = _vector(uncertainty, "uncertainty", finite=False)
    count = len(true)
    _same_length(count, prediction=predicted, uncertainty=score)
    groups = _groups(group_ids, count)
    accepted = _mask(accepted_by_policy, count, "accepted_by_policy")
    complete = _mask(reference_complete, count, "reference_complete")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    coverages = _vector(requested_coverages, "requested_coverages")
    if np.any((coverages < 0) | (coverages > 1)):
        raise ValueError("coverage must lie in [0, 1]")
    if np.any(np.isfinite(score) & (score < 0)):
        raise ValueError("uncertainty must be nonnegative where available")
    if np.any(complete & ~np.isfinite(true)):
        raise ValueError("A completed reference must contain a finite value")

    eligible = np.isfinite(predicted) & np.isfinite(score)
    if np.any(accepted & ~eligible):
        raise ValueError("Policy cannot accept a missing prediction or missing trust score")
    weights = group_balanced_weights(groups.tolist())
    resolved = complete & np.isfinite(true)
    lower = upper = None
    if (interval_lower is None) != (interval_upper is None):
        raise ValueError("Both interval bounds must be supplied together")
    if interval_lower is not None and interval_upper is not None:
        lower = _vector(interval_lower, "interval_lower", finite=False)
        upper = _vector(interval_upper, "interval_upper", finite=False)
        _same_length(count, interval_lower=lower, interval_upper=upper)
        both_finite = np.isfinite(lower) & np.isfinite(upper)
        if np.any(both_finite & (lower > upper)):
            raise ValueError("Lower interval bound exceeds upper bound")

    def row(selected: np.ndarray) -> dict[str, Any]:
        known = selected & resolved
        selected_mass = float(weights[selected].sum())
        known_mass = float(weights[known].sum())
        unknown_mass = float(weights[selected & ~resolved].sum())
        errors = predicted[known] - true[known]
        absolute = np.abs(errors)
        local_weights = weights[known]
        result: dict[str, Any] = {
            "accepted_requests": int(selected.sum()),
            "accepted_mass": selected_mass,
            "accepted_reference_resolved_requests": int(known.sum()),
            "accepted_reference_resolved_mass": known_mass,
            "accepted_reference_unresolved_requests": int((selected & ~resolved).sum()),
            "accepted_reference_unresolved_mass": unknown_mass,
            "unresolved_fraction_of_accepted_mass": (
                unknown_mass / selected_mass if selected_mass else None
            ),
            "conditional_mae_on_resolved": (
                float(np.dot(local_weights, absolute) / known_mass) if known_mass else None
            ),
            "conditional_rmse_on_resolved": (
                float(np.sqrt(np.dot(local_weights, errors**2) / known_mass)) if known_mass else None
            ),
            "accepted_resolved_exceedance_requests": int((absolute > tolerance).sum()),
            "accepted_resolved_exceedance_mass": float(local_weights[absolute > tolerance].sum()),
            "conditional_exceedance_on_resolved": (
                float(local_weights[absolute > tolerance].sum() / known_mass) if known_mass else None
            ),
            "risk_status": (
                "empty_acceptance"
                if selected_mass == 0
                else "unresolved_references_present"
                if unknown_mass > 0
                else "resolved"
            ),
        }
        if lower is not None and upper is not None:
            interval_known = known & np.isfinite(lower) & np.isfinite(upper)
            interval_mass = float(weights[interval_known].sum())
            inside = (true >= lower) & (true <= upper)
            fully_resolved_groups: list[str] = []
            simultaneous: list[bool] = []
            for group in np.unique(groups[selected]):
                selected_group = selected & (groups == group)
                if np.all(interval_known[selected_group]):
                    fully_resolved_groups.append(group)
                    simultaneous.append(bool(np.all(inside[selected_group])))
            result["intervals"] = {
                "accepted_with_resolved_reference_and_interval_mass": interval_mass,
                "conditional_empirical_coverage": (
                    float(weights[interval_known & inside].sum() / interval_mass)
                    if interval_mass
                    else None
                ),
                "conditional_mean_width": (
                    float(np.dot(weights[interval_known], (upper - lower)[interval_known]) / interval_mass)
                    if interval_mass
                    else None
                ),
                "fully_resolved_accepted_group_count": len(fully_resolved_groups),
                "accepted_group_count": int(len(np.unique(groups[selected]))),
                "simultaneous_coverage_in_fully_resolved_accepted_groups": (
                    float(np.mean(simultaneous)) if simultaneous else None
                ),
                "formal_coverage_guarantee": False,
            }
        return result

    thresholds = np.unique(score[eligible])
    curve: list[dict[str, Any]] = []
    for requested in coverages:
        chosen = np.zeros(count, dtype=bool)
        chosen_threshold: float | None = None
        for threshold in thresholds:
            proposed = eligible & (score <= threshold)
            if float(weights[proposed].sum()) <= float(requested) + 1e-12:
                chosen = proposed
                chosen_threshold = float(threshold)
            else:
                break
        curve.append(
            {
                "requested_coverage": float(requested),
                "score_threshold": chosen_threshold,
                **row(chosen),
            }
        )
    good = (~accepted) & resolved & np.isfinite(predicted) & (np.abs(predicted - true) <= tolerance)
    return {
        "schema_version": "selective-regression-diagnostics-v1",
        "attempted_requests": count,
        "independent_group_count": len(np.unique(groups)),
        "weighting": "equal_group_mass_on_full_attempted_pool",
        "reference_resolved_requests": int(resolved.sum()),
        "reference_resolved_mass": float(weights[resolved].sum()),
        "prediction_available_requests": int(np.isfinite(predicted).sum()),
        "trust_score_available_requests": int(np.isfinite(score).sum()),
        "tolerance": float(tolerance),
        "tie_policy": "whole_equal_score_bins_at_or_below_requested_coverage",
        "policy_rejected_reference_resolved_good_requests": int(good.sum()),
        "policy_rejected_reference_resolved_good_mass": float(weights[good].sum()),
        "frozen_policy": row(accepted),
        "risk_coverage": curve,
    }


def paired_group_bootstrap(
    left_losses: Sequence[float],
    right_losses: Sequence[float],
    group_ids: Sequence[str],
    *,
    seed: int = 20260908,
    repetitions: int = 2000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Calculate a percentile interval over independent geometry-group differences."""
    left = _vector(left_losses, "left_losses")
    right = _vector(right_losses, "right_losses")
    _same_length(len(left), right_losses=right)
    groups = _groups(group_ids, len(left))
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 100:
        raise ValueError("At least 100 integer bootstrap repetitions required")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    unique = np.unique(groups)
    differences = np.asarray([np.mean((left - right)[groups == group]) for group in unique])
    result: dict[str, Any] = {
        "estimand": "mean_group_average_left_minus_right_loss",
        "negative_means_left_better": True,
        "group_count": len(unique),
        "row_count": len(left),
        "mean_difference": float(differences.mean()),
        "confidence": confidence,
        "seed": seed,
        "repetitions": repetitions,
        "assumption": "independent_exchangeable_groups_established_by_caller",
    }
    if len(unique) < 2:
        return {**result, "status": "insufficient_independent_groups", "interval": None}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for start in range(0, repetitions, 256):
        count = min(256, repetitions - start)
        draws = rng.integers(0, len(unique), size=(count, len(unique)))
        estimates.extend(differences[draws].mean(axis=1).tolist())
    tail = (1 - confidence) / 2
    return {
        **result,
        "status": "calculated",
        "interval": np.quantile(estimates, [tail, 1 - tail]).tolist(),
    }


def nested_group_subsets(
    group_ids: Sequence[str],
    fractions: Sequence[float],
    *,
    repetitions: int = 3,
    seed: int = 20260908,
) -> list[dict[str, Any]]:
    """Produce reproducible nested geometry-group subsets for learning curves."""
    groups = _groups(group_ids, len(group_ids))
    fractions_array = _vector(fractions, "fractions")
    if np.any((fractions_array <= 0) | (fractions_array > 1)) or np.any(
        np.diff(fractions_array) <= 0
    ):
        raise ValueError("fractions must be unique, increasing and in (0, 1]")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    unique = np.unique(groups)
    result: list[dict[str, Any]] = []
    for repetition, child in enumerate(np.random.SeedSequence(seed).spawn(repetitions)):
        order = np.random.default_rng(child).permutation(unique).tolist()
        subsets = []
        for fraction in fractions_array:
            count = max(1, math.ceil(float(fraction) * len(unique)))
            subsets.append({"fraction": float(fraction), "group_count": count, "group_ids": order[:count]})
        result.append(
            {
                "repetition": repetition,
                "seed_entropy": seed,
                "seed_spawn_key": list(child.spawn_key),
                "cluster_order": order,
                "subsets": subsets,
            }
        )
    return result


def finite_pool_regret(
    case_ids: Sequence[str],
    prediction_scores: Sequence[float],
    reference_losses: Sequence[float],
    reference_complete: Sequence[bool],
) -> dict[str, Any]:
    """Report regret only for a fully resolved declared finite pool."""
    scores = _vector(prediction_scores, "prediction_scores", finite=False)
    true = _vector(reference_losses, "reference_losses", finite=False)
    _same_length(len(scores), reference_losses=true)
    case_id_array = _groups(case_ids, len(scores))
    if len(np.unique(case_id_array)) != len(case_id_array):
        raise ValueError("Finite-pool candidate IDs must be unique")
    complete = _mask(reference_complete, len(case_id_array), "reference_complete")
    if np.any(complete & ~np.isfinite(true)):
        raise ValueError("Complete reference losses must be finite")
    candidates = np.flatnonzero(np.isfinite(scores))
    selected = min(candidates, key=lambda index: (scores[index], case_id_array[index])) if len(candidates) else None
    missing = case_id_array[~complete | ~np.isfinite(true)].tolist()
    base: dict[str, Any] = {
        "pool_size": len(case_id_array),
        "predicted_candidate_count": len(candidates),
        "nomination": str(case_id_array[selected]) if selected is not None else None,
        "unresolved_reference_ids": missing,
        "global_optimality_claimed": False,
    }
    if selected is None:
        return {**base, "status": "no_prediction_nomination", "finite_pool_regret": None}
    if missing:
        return {**base, "status": "unresolved_reference_pool", "finite_pool_regret": None}
    best = min(range(len(case_id_array)), key=lambda index: (true[index], case_id_array[index]))
    return {
        **base,
        "status": "resolved_finite_pool",
        "best_verified_pool_id": str(case_id_array[best]),
        "nominated_reference_loss": float(true[selected]),
        "best_verified_pool_loss": float(true[best]),
        "finite_pool_regret": float(true[selected] - true[best]),
    }


def cost_comparison(
    *,
    offline_seconds: float,
    assisted_online_seconds: float,
    direct_online_seconds: float,
    assisted_solver_attempts: int,
    direct_solver_attempts: int,
    assisted_context_hash: str,
    direct_context_hash: str,
    verified_quality_comparable: bool,
    quality_evidence_id: str | None,
) -> dict[str, Any]:
    """Calculate matched-workload timing arithmetic without creating quality claims."""
    timings = [offline_seconds, assisted_online_seconds, direct_online_seconds]
    if not np.isfinite(timings).all() or min(timings) < 0:
        raise ValueError("Measured times must be finite and nonnegative")
    for value in (assisted_solver_attempts, direct_solver_attempts):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Solver attempts must be nonnegative integers")
    if not assisted_context_hash or assisted_context_hash != direct_context_hash:
        raise ValueError("Only matched-workload contexts may be compared")
    if not isinstance(verified_quality_comparable, bool):
        raise ValueError("Quality-comparable flag must be Boolean")
    if verified_quality_comparable and not quality_evidence_id:
        raise ValueError("Comparable verified quality requires evidence ID")
    savings = direct_online_seconds - assisted_online_seconds
    admissible = verified_quality_comparable and bool(quality_evidence_id)
    return {
        "context_hash": assisted_context_hash,
        "offline_seconds": offline_seconds,
        "assisted_online_seconds": assisted_online_seconds,
        "direct_online_seconds": direct_online_seconds,
        "online_seconds_saved": savings,
        "descriptive_online_time_ratio": (
            direct_online_seconds / assisted_online_seconds if assisted_online_seconds > 0 else None
        ),
        "assisted_solver_attempts": assisted_solver_attempts,
        "direct_solver_attempts": direct_solver_attempts,
        "solver_attempts_saved": direct_solver_attempts - assisted_solver_attempts,
        "verified_quality_comparable": verified_quality_comparable,
        "quality_evidence_id": quality_evidence_id,
        "quality_matched_cost_claim_admissible": admissible,
        "amortised_workload_break_even": (
            max(1, math.ceil(offline_seconds / savings)) if admissible and savings > 0 else None
        ),
        "break_even_status": (
            "unverified_quality"
            if not admissible
            else "no_positive_online_saving"
            if savings <= 0
            else "calculated"
        ),
        "historical_unmeasured_acquisition_cost_included": False,
    }


def benchmark_callable(
    call: Callable[[], Any],
    *,
    unit: str,
    cases_per_call: int = 1,
    warmups: int = 20,
    repetitions: int = 100,
    synchronise: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Time a caller-declared workload with optional explicit synchronization."""
    units = {
        "operating_point",
        "operational_polar",
        "uncertainty_assessment",
        "verified_design_workflow",
        "software_test",
    }
    if unit not in units:
        raise ValueError("Declare a supported workload unit")
    for name, value, minimum in (
        ("cases_per_call", cases_per_call, 1),
        ("warmups", warmups, 0),
        ("repetitions", repetitions, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Invalid {name}")
    sync = synchronise or (lambda: None)
    for _ in range(warmups):
        sync()
        call()
        sync()
    durations: list[float] = []
    for _ in range(repetitions):
        sync()
        start = time.perf_counter_ns()
        call()
        sync()
        durations.append((time.perf_counter_ns() - start) / 1e9)
    values = np.asarray(durations)
    median = float(np.median(values))
    return {
        "unit": unit,
        "cases_per_call": cases_per_call,
        "warmups": warmups,
        "repetitions": repetitions,
        "explicit_synchronisation_callback": synchronise is not None,
        "seconds_per_call": durations,
        "median_seconds_per_call": median,
        "p95_seconds_per_call": float(np.quantile(values, 0.95)),
        "median_amortised_seconds_per_case": median / cases_per_call,
        "median_throughput_cases_per_second": cases_per_call / median if median > 0 else None,
        "timing_scope": "caller_declared_workload_not_an_automatic_solver_comparison",
    }
