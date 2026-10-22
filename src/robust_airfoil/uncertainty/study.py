from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from robust_airfoil.config import UncertaintyConfig
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.normalise import resample_surfaces_to_common_x
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import hash_object, sha256_bytes
from robust_airfoil.modelling.calibrate import apply_trust_model
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.modelling.evaluate import cvar
from robust_airfoil.uncertainty.fields import PerturbedGeometry, smooth_normal_perturbation
from robust_airfoil.uncertainty.sampling import sobol_normal_samples


def _parameter_frame(parameters: np.ndarray, alpha: list[float], sample_ids: np.ndarray) -> pd.DataFrame:
    repeated = np.repeat(parameters, len(alpha), axis=0)
    frame = pd.DataFrame(
        repeated,
        columns=[f"lower_weight_{i}" for i in range(5)]
        + [f"upper_weight_{i}" for i in range(5)]
        + ["leading_edge_weight", "TE_thickness"],
    )
    frame["alpha_deg"] = np.tile(np.asarray(alpha, dtype=float), len(parameters))
    frame["airfoil_id"] = np.repeat(sample_ids.astype(str), len(alpha))
    return frame


def _matrix_hash(*matrices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for matrix in matrices:
        canonical = np.ascontiguousarray(matrix, dtype="<f8")
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {}
    return {
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "q05": float(np.quantile(finite, 0.05)),
        "q50": float(np.quantile(finite, 0.50)),
        "q95": float(np.quantile(finite, 0.95)),
        "cvar_95": float(cvar(finite, 0.95)),
    }


def _fit_valid(
    perturbations: list[PerturbedGeometry],
    maximum_cst_refit_error: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    parameters: list[np.ndarray] = []
    sample_ids: list[int] = []
    records: list[dict[str, Any]] = []
    invalid_flags = np.zeros(len(perturbations), dtype=bool)
    for index, perturbation in enumerate(perturbations):
        reasons = list(perturbation.invalid_reasons)
        record: dict[str, Any] = {
            "sample": index,
            "valid_geometry": perturbation.valid,
            "invalid_reasons": reasons,
            "maximum_absolute_displacement": float(
                max(
                    np.max(np.abs(perturbation.upper_displacement)),
                    np.max(np.abs(perturbation.lower_displacement)),
                )
            ),
        }
        if perturbation.valid:
            cosine_x = (1 - np.cos(np.linspace(0, np.pi, len(perturbation.upper)))) / 2
            fit_upper, fit_lower = resample_surfaces_to_common_x(
                perturbation.upper,
                perturbation.lower,
                cosine_x,
            )
            fit = fit_cst(fit_upper, fit_lower)
            record.update({"cst_rms_error": fit.rms_error, "cst_max_error": fit.max_error})
            if fit.max_error > maximum_cst_refit_error:
                reasons.append("excessive_cst_refit_error")
                record["valid_geometry"] = False
            else:
                parameters.append(fit.parameters)
                sample_ids.append(index)
        if reasons:
            invalid_flags[index] = True
        record["invalid_reasons"] = reasons
        records.append(record)
    matrix = np.asarray(parameters, dtype=float)
    if not len(parameters):
        matrix = np.empty((0, 12), dtype=float)
    return matrix, np.asarray(sample_ids, dtype=int), records, invalid_flags


def _evaluate_perturbations(
    perturbations: list[PerturbedGeometry],
    ensemble: LoadedEnsemble,
    alpha: list[float],
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
    maximum_cst_refit_error: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    parameters, sample_ids, records, invalid_flags = _fit_valid(
        perturbations, maximum_cst_refit_error
    )
    outputs = np.full(len(perturbations), np.nan, dtype=float)
    trust_flags = np.zeros(len(perturbations), dtype=bool)
    distributions: dict[str, Any] = {}
    if len(parameters):
        frame = _parameter_frame(parameters, alpha, sample_ids)
        predictions = ensemble.predict(frame)
        predictions = apply_trust_model(
            development_points,
            predictions,
            ensemble.first_scaling,
            calibration,
        )
        predictions["cd"] = np.exp(predictions["prediction_log_cd"].to_numpy(float))
        per_sample = predictions.groupby("airfoil_id", sort=False)["cd"].mean()
        per_sample_trusted = predictions.groupby("airfoil_id", sort=False)["trusted_domain"].all()
        for sample_id, value in per_sample.items():
            index = int(sample_id)
            outputs[index] = float(value)
            trust_flags[index] = not bool(per_sample_trusted.loc[sample_id])
        for target in ("prediction_cl", "prediction_log_cd", "prediction_cm", "cd"):
            distributions[target] = _distribution(predictions[target].to_numpy(float))
    violation_flags = invalid_flags | trust_flags
    finite = outputs[np.isfinite(outputs) & ~violation_flags]
    penalty = float(np.max(finite) * 1.5) if len(finite) else 1.0
    outputs[~np.isfinite(outputs) | violation_flags] = penalty
    invalid_reason_counts = Counter(
        reason for record in records for reason in record["invalid_reasons"]
    )
    summary = {
        "samples": len(perturbations),
        "valid_geometry": int((~invalid_flags).sum()),
        "invalid_geometry": int(invalid_flags.sum()),
        "trust_domain_violations": int(trust_flags.sum()),
        "total_violations": int(violation_flags.sum()),
        "violation_probability": float(np.mean(violation_flags)) if len(violation_flags) else 1.0,
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "maximum_observed_absolute_displacement": max(
            (float(record["maximum_absolute_displacement"]) for record in records),
            default=0.0,
        ),
        "prediction_distributions": distributions,
        "weighted_cd_distribution_with_violation_penalty": _distribution(outputs),
        "sample_record_hash": hash_object(records),
        "cst_max_refit_error_tolerance": maximum_cst_refit_error,
    }
    return summary, outputs, violation_flags


def _from_latent(
    upper: np.ndarray,
    lower: np.ndarray,
    latent: np.ndarray,
    amplitude: float,
    config: UncertaintyConfig,
) -> list[PerturbedGeometry]:
    controls = config.basis_control_points_per_surface
    correlation = config.upper_lower_correlation
    independent_scale = float(np.sqrt(max(0.0, 1 - correlation**2)))
    return [
        smooth_normal_perturbation(
            upper,
            lower,
            row[:controls],
            correlation * row[:controls] + independent_scale * row[controls:],
            amplitude,
            config.correlation_length_chord,
            config.leading_edge_taper,
            config.trailing_edge_zero_displacement,
        )
        for row in latent
    ]


def _validate_convergence_counts(config: UncertaintyConfig) -> None:
    maximum_count = config.final_evaluation_samples.count
    if not config.convergence_counts:
        raise ValueError("At least one Sobol convergence count is required")
    for count in config.convergence_counts:
        if count < 2 or count & (count - 1):
            raise ValueError("Sobol convergence counts must be powers of two >= 2")
        if count > maximum_count:
            raise ValueError("Sobol convergence count exceeds final evaluation sample count")


def _sobol_study(
    upper: np.ndarray,
    lower: np.ndarray,
    ensemble: LoadedEnsemble,
    alpha: list[float],
    config: UncertaintyConfig,
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
    amplitude: float,
) -> dict[str, Any]:
    _validate_convergence_counts(config)
    maximum_count = max(config.convergence_counts)
    dimension = config.basis_control_points_per_surface * 2
    a = sobol_normal_samples(maximum_count, dimension, config.seed + 91, antithetic=False)
    b = sobol_normal_samples(maximum_count, dimension, config.seed + 92, antithetic=False)
    _, f_a, violations_a = _evaluate_perturbations(
        _from_latent(upper, lower, a, amplitude, config),
        ensemble,
        alpha,
        development_points,
        calibration,
        config.maximum_cst_refit_error_fraction_chord,
    )
    _, f_b, _ = _evaluate_perturbations(
        _from_latent(upper, lower, b, amplitude, config),
        ensemble,
        alpha,
        development_points,
        calibration,
        config.maximum_cst_refit_error_fraction_chord,
    )
    hybrid_outputs: list[np.ndarray] = []
    matrix_digest = hashlib.sha256()
    matrix_digest.update(bytes.fromhex(_matrix_hash(a, b)))
    for dimension_index in range(dimension):
        hybrid = a.copy()
        hybrid[:, dimension_index] = b[:, dimension_index]
        matrix_digest.update(np.ascontiguousarray(hybrid, dtype="<f8").tobytes())
        _, values, _ = _evaluate_perturbations(
            _from_latent(upper, lower, hybrid, amplitude, config),
            ensemble,
            alpha,
            development_points,
            calibration,
            config.maximum_cst_refit_error_fraction_chord,
        )
        hybrid_outputs.append(values)
    convergence: dict[str, Any] = {}
    for count in config.convergence_counts:
        values = f_a[:count]
        variance = float(np.var(np.concatenate([values, f_b[:count]]), ddof=1))
        first_order: list[float] = []
        total_order: list[float] = []
        for hybrid_values in hybrid_outputs:
            first_order.append(
                float(np.mean(f_b[:count] * (hybrid_values[:count] - values)) / max(variance, 1e-15))
            )
            total_order.append(
                float(0.5 * np.mean((values - hybrid_values[:count]) ** 2) / max(variance, 1e-15))
            )
        statistics = _distribution(values)
        convergence[str(count)] = {
            **statistics,
            "standard_error": float(np.std(values, ddof=1) / np.sqrt(count)),
            "violation_probability": float(np.mean(violations_a[:count])),
            "first_order_indices": first_order,
            "total_order_indices": total_order,
        }
    labels = [f"upper_control_{index}" for index in range(config.basis_control_points_per_surface)] + [
        f"lower_control_{index}" for index in range(config.basis_control_points_per_surface)
    ]
    return {
        "method": "pick_freeze_scrambled_sobol",
        "amplitude": amplitude,
        "dimension_labels": labels,
        "base_sample_count": maximum_count,
        "total_model_evaluations": (dimension + 2) * maximum_count,
        "sample_matrix_sha256": matrix_digest.hexdigest(),
        "sample_matrix_encoding": "little_endian_float64_row_major; A,B,then A_Bi hybrids",
        "convergence": convergence,
    }


def _ablation_summary(
    nominal_parameters: np.ndarray,
    development_parameter_std: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    ensemble: LoadedEnsemble,
    alpha: list[float],
    config: UncertaintyConfig,
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    count = config.final_evaluation_samples.count
    amplitude = max(config.amplitudes_fraction_chord)
    payload: dict[str, Any] = {}
    if "independent_multiplicative_cst" in config.ablations:
        latent = sobol_normal_samples(count, len(nominal_parameters), config.seed + 500, antithetic=True)
        uniform = 2 * norm.cdf(latent) - 1
        scales = np.where(
            np.abs(nominal_parameters) > 1e-6,
            np.abs(nominal_parameters) * 0.01,
            np.maximum(development_parameter_std, 1e-12) * 0.01,
        )
        cst_parameters = nominal_parameters[None, :] + uniform * scales[None, :]
        x = upper[:, 0]
        cst_perturbations: list[PerturbedGeometry] = []
        for parameters in cst_parameters:
            upper_y, lower_y = reconstruct_cst(parameters, x)
            candidate_upper = np.column_stack([x, upper_y])
            candidate_lower = np.column_stack([x, lower_y])
            validity = validate_geometry(candidate_upper, candidate_lower)
            cst_perturbations.append(
                PerturbedGeometry(
                    candidate_upper,
                    candidate_lower,
                    np.zeros(len(x)),
                    np.zeros(len(x)),
                    validity.valid,
                    validity.reasons,
                )
            )
        summary, _, _ = _evaluate_perturbations(
            cst_perturbations,
            ensemble,
            alpha,
            development_points,
            calibration,
            config.maximum_cst_refit_error_fraction_chord,
        )
        summary.update(
            {
                "parameter_rule": "u_j~Uniform[-0.01,0.01] multiplicative; near-zero uses 1% development std",
                "sample_matrix_sha256": _matrix_hash(uniform),
            }
        )
        payload["independent_multiplicative_cst"] = summary
    if "independent_coordinate_noise" in config.ablations:
        coordinate_latent = sobol_normal_samples(count, len(upper) * 2, config.seed + 501, antithetic=True)
        coordinate_perturbations: list[PerturbedGeometry] = []
        x = upper[:, 0]
        taper = np.sqrt(np.clip(x / 0.10, 0.0, 1.0)) * (1 - x)
        for row in coordinate_latent:
            upper_noise = row[: len(upper)]
            lower_noise = row[len(upper) :]
            upper_noise /= max(float(np.max(np.abs(upper_noise))), 1e-12)
            lower_noise /= max(float(np.max(np.abs(lower_noise))), 1e-12)
            perturbed_upper = upper.copy()
            perturbed_lower = lower.copy()
            perturbed_upper[:, 1] += amplitude * upper_noise * taper
            perturbed_lower[:, 1] += amplitude * lower_noise * taper
            validity = validate_geometry(perturbed_upper, perturbed_lower)
            coordinate_perturbations.append(
                PerturbedGeometry(
                    perturbed_upper,
                    perturbed_lower,
                    amplitude * upper_noise * taper,
                    amplitude * lower_noise * taper,
                    validity.valid,
                    validity.reasons,
                )
            )
        summary, _, _ = _evaluate_perturbations(
            coordinate_perturbations,
            ensemble,
            alpha,
            development_points,
            calibration,
            config.maximum_cst_refit_error_fraction_chord,
        )
        summary["sample_matrix_sha256"] = _matrix_hash(coordinate_latent)
        payload["independent_coordinate_noise"] = summary
    if "smooth_correlated_surface_normal" in config.ablations:
        payload["smooth_correlated_surface_normal"] = {
            "primary_model": True,
            "referenced_level": str(amplitude),
        }
    return payload


def run_manufacturing_study(
    nominal_parameters: np.ndarray,
    development_parameter_std: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    member_root: Path,
    alpha: list[float],
    config: UncertaintyConfig,
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if len(upper) != config.cosine_points_per_surface or len(lower) != config.cosine_points_per_surface:
        raise ValueError("Manufacturing study geometry does not match configured cosine point count")
    if config.final_evaluation_samples.method != "scrambled_sobol":
        raise ValueError("Only scrambled Sobol final evaluation is implemented")
    ensemble = LoadedEnsemble(member_root)
    levels: dict[str, Any] = {}
    sobol: dict[str, Any] = {}
    dimension = config.basis_control_points_per_surface * 2
    for level_index, amplitude in enumerate(config.amplitudes_fraction_chord):
        latent = sobol_normal_samples(
            config.final_evaluation_samples.count,
            dimension,
            config.seed + level_index,
            antithetic=True,
        )
        perturbations = _from_latent(upper, lower, latent, amplitude, config)
        summary, _, _ = _evaluate_perturbations(
            perturbations,
            ensemble,
            alpha,
            development_points,
            calibration,
            config.maximum_cst_refit_error_fraction_chord,
        )
        summary["amplitude_bound"] = amplitude
        summary["bound_respected"] = (
            summary["maximum_observed_absolute_displacement"] <= amplitude + 1e-12
        )
        summary["sample_matrix_sha256"] = sha256_bytes(
            np.ascontiguousarray(latent, dtype="<f8").tobytes()
        )
        summary["sample_matrix_shape"] = list(latent.shape)
        levels[str(amplitude)] = summary
        sobol[str(amplitude)] = _sobol_study(
            upper,
            lower,
            ensemble,
            alpha,
            config,
            development_points,
            calibration,
            amplitude,
        )
    payload = {
        "schema_version": 2,
        "representation": config.representation,
        "interpretation": "prescribed bounded manufacturing perturbations, not a measured manufacturing distribution",
        "effective_configuration": config.model_dump(mode="json"),
        "levels": levels,
        "sobol": sobol,
        "ablations": _ablation_summary(
            nominal_parameters,
            development_parameter_std,
            upper,
            lower,
            ensemble,
            alpha,
            config,
            development_points,
            calibration,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
