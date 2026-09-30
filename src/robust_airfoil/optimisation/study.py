from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from scipy.stats import norm

from robust_airfoil.config import Nsga2Config, OptimisationConfig, UncertaintyConfig
from robust_airfoil.front_agreement import front_agreement
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.metrics import geometry_metrics
from robust_airfoil.geometry.normalise import resample_surfaces_to_common_x
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import hash_object, sha256_bytes, sha256_file
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.modelling.evaluate import cvar
from robust_airfoil.optimisation.objectives import (
    ava_lift_sensitivity,
    weighted_required_lift_drag,
)
from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.uncertainty.propagation import generate_perturbations
from robust_airfoil.uncertainty.sampling import sobol_normal_samples

PARAMETER_COLUMNS = FEATURE_COLUMNS[:-1]


def _frame(parameters: np.ndarray, alpha: np.ndarray) -> pd.DataFrame:
    repeated = np.repeat(parameters, len(alpha), axis=0)
    frame = pd.DataFrame(repeated, columns=pd.Index(PARAMETER_COLUMNS))
    frame["alpha_deg"] = np.tile(alpha, len(parameters))
    frame["airfoil_id"] = np.repeat(np.arange(len(parameters)).astype(str), len(alpha))
    return frame


def manufacturing_parameter_deltas(
    reference_parameters: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    uncertainty: UncertaintyConfig,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    perturbations = generate_perturbations(
        upper,
        lower,
        count,
        uncertainty.basis_control_points_per_surface,
        max(uncertainty.amplitudes_fraction_chord),
        seed,
        uncertainty.upper_lower_correlation,
        uncertainty.correlation_length_chord,
        uncertainty.leading_edge_taper,
        uncertainty.trailing_edge_zero_displacement,
    )
    deltas = np.zeros((len(perturbations), len(reference_parameters)), dtype=float)
    invalid_flags = np.zeros(len(perturbations), dtype=bool)
    reasons: dict[str, int] = {}
    for index, perturbation in enumerate(perturbations):
        invalid_reason: str | None = None
        if not perturbation.valid:
            invalid_reason = ";".join(perturbation.invalid_reasons) or "invalid_geometry"
        else:
            fit_upper, fit_lower = resample_surfaces_to_common_x(
                perturbation.upper,
                perturbation.lower,
                (1 - np.cos(np.linspace(0, np.pi, len(perturbation.upper)))) / 2,
            )
            fit = fit_cst(fit_upper, fit_lower)
            if fit.max_error > uncertainty.maximum_cst_refit_error_fraction_chord:
                invalid_reason = "excessive_cst_refit_error"
            else:
                deltas[index] = fit.parameters - reference_parameters
        if invalid_reason:
            invalid_flags[index] = True
            reasons[invalid_reason] = reasons.get(invalid_reason, 0) + 1
    if int((~invalid_flags).sum()) < 16:
        raise RuntimeError("Too few valid manufacturing perturbations for robust optimisation")
    return deltas, invalid_flags, {
        "requested": len(perturbations),
        "valid": int((~invalid_flags).sum()),
        "invalid": int(invalid_flags.sum()),
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "common_random_numbers": True,
        "amplitude": max(uncertainty.amplitudes_fraction_chord),
        "seed": seed,
        "parameter_delta_sha256": sha256_bytes(np.ascontiguousarray(deltas, dtype="<f8").tobytes()),
    }


class RobustCandidateEvaluator:
    def __init__(
        self,
        reference_parameters: np.ndarray,
        deltas: np.ndarray,
        manufacturing_invalid_flags: np.ndarray,
        development_points: pd.DataFrame,
        ensemble: LoadedEnsemble,
        calibration: dict[str, Any],
        optimisation: OptimisationConfig,
    ):
        if len(deltas) != len(manufacturing_invalid_flags):
            raise ValueError("Manufacturing delta and validity arrays differ")
        self.reference_parameters = reference_parameters
        self.deltas = deltas
        self.manufacturing_invalid_flags = manufacturing_invalid_flags
        self.ensemble = ensemble
        self.config = optimisation
        self.alpha = np.asarray(optimisation.ava_baseline.alpha_deg, dtype=float)
        self.required = np.asarray(optimisation.service_targets.cl_required, dtype=float)
        self.weights = np.asarray(optimisation.service_targets.weights, dtype=float)
        x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
        upper_y, lower_y = reconstruct_cst(reference_parameters, x)
        self.x = x
        self.reference_metrics = geometry_metrics(
            np.column_stack([x, upper_y]), np.column_stack([x, lower_y])
        )
        scaling = ensemble.first_scaling
        support_features = (
            development_points[FEATURE_COLUMNS].to_numpy(float) - scaling.feature_center
        ) / scaling.feature_scale
        self.support_tree = cKDTree(support_features)
        self.feature_center = scaling.feature_center
        self.feature_scale = scaling.feature_scale
        self.support_threshold = float(calibration["support_distance_quantiles"]["0.99"])
        self.disagreement_thresholds = {
            target: float(values["0.99"])
            for target, values in calibration["ensemble_disagreement_quantiles"].items()
        }
        reference_prediction = ensemble.predict(_frame(reference_parameters[None, :], self.alpha))
        self.reference_cm = float(reference_prediction["prediction_cm"].mean())
        self.counters = {
            "candidate_evaluations": 0,
            "invalid_geometry": 0,
            "trust_rejections": 0,
            "missing_lift_roots": 0,
            "manufacturing_invalid_samples": 0,
        }

    def _support_and_disagreement(self, predictions: pd.DataFrame, shape: tuple[int, ...]) -> np.ndarray:
        standardised = (
            predictions[FEATURE_COLUMNS].to_numpy(float) - self.feature_center
        ) / self.feature_scale
        distances, _ = self.support_tree.query(standardised, k=5)
        support = np.mean(np.atleast_2d(distances), axis=1).reshape(shape)
        trusted = support <= self.support_threshold
        for target, threshold in self.disagreement_thresholds.items():
            trusted &= predictions[f"ensemble_std_{target}"].to_numpy(float).reshape(shape) <= threshold
        return trusted

    def _nominal_constraints(
        self, candidates: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        constraint_rows: list[list[float]] = []
        diagnostics: list[dict[str, Any]] = []
        nominal_predictions = self.ensemble.predict(_frame(candidates, self.alpha))
        alpha_count = len(self.alpha)
        trusted_rows = self._support_and_disagreement(
            nominal_predictions, (len(candidates), alpha_count)
        )
        for index, candidate in enumerate(candidates):
            upper_y, lower_y = reconstruct_cst(candidate, self.x)
            upper = np.column_stack([self.x, upper_y])
            lower = np.column_stack([self.x, lower_y])
            validity = validate_geometry(upper, lower)
            metrics = geometry_metrics(upper, lower)
            ratio_thickness = metrics["maximum_thickness"] / self.reference_metrics["maximum_thickness"]
            ratio_area = metrics["section_area"] / self.reference_metrics["section_area"]
            ratio_radius = metrics["leading_edge_radius_proxy"] / self.reference_metrics["leading_edge_radius_proxy"]
            prediction = nominal_predictions.iloc[index * alpha_count : (index + 1) * alpha_count]
            trust_ok = bool(np.all(trusted_rows[index]))
            candidate_cm = float(prediction["prediction_cm"].mean())
            margins = [
                ratio_thickness - self.config.relative_constraints.thickness_ratio_lower,
                self.config.relative_constraints.thickness_ratio_upper - ratio_thickness,
                ratio_area - self.config.relative_constraints.section_area_lower,
                ratio_radius - self.config.relative_constraints.leading_edge_radius_lower,
                candidate_cm
                - (self.reference_cm - self.config.relative_constraints.cm_allowance_below_reference),
                1.0 if validity.valid and trust_ok else -1.0,
            ]
            constraint_rows.append([-value for value in margins])
            diagnostics.append(
                {
                    "valid_geometry": validity.valid,
                    "invalid_reasons": list(validity.reasons),
                    "trusted": trust_ok,
                    "maximum_support_distance": float(
                        np.max(
                            np.mean(
                                np.atleast_2d(
                                    self.support_tree.query(
                                        (
                                            prediction[FEATURE_COLUMNS].to_numpy(float)
                                            - self.feature_center
                                        )
                                        / self.feature_scale,
                                        k=5,
                                    )[0]
                                ),
                                axis=1,
                            )
                        )
                    ),
                    "geometry_metrics": metrics,
                    "nominal_predictions": {
                        target: prediction[f"prediction_{target}"].to_numpy(float).tolist()
                        for target in ("cl", "log_cd", "cm")
                    },
                    "cm": candidate_cm,
                    "margins": margins,
                }
            )
            self.counters["invalid_geometry"] += int(not validity.valid)
            self.counters["trust_rejections"] += int(not trust_ok)
        return np.asarray(constraint_rows), diagnostics

    def evaluate(
        self, candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        candidates = np.atleast_2d(candidates).astype(float)
        self.counters["candidate_evaluations"] += len(candidates)
        perturbed = candidates[:, None, :] + self.deltas[None, :, :]
        flat = perturbed.reshape(-1, perturbed.shape[-1])
        predictions = self.ensemble.predict(_frame(flat, self.alpha))
        shape = (len(candidates), len(self.deltas), len(self.alpha))
        cl = predictions["prediction_cl"].to_numpy(float).reshape(shape)
        cd = np.exp(predictions["prediction_log_cd"].to_numpy(float)).reshape(shape)
        cm = predictions["prediction_cm"].to_numpy(float).reshape(shape)
        trusted = self._support_and_disagreement(predictions, shape)
        objectives = np.empty((len(candidates), 2), dtype=float)
        sample_drags: list[list[float]] = []
        trust_violations: list[int] = []
        missing_roots: list[int] = []
        for candidate_index in range(len(candidates)):
            drags = np.asarray(
                [
                    weighted_required_lift_drag(
                        self.alpha,
                        cl[candidate_index, sample],
                        cd[candidate_index, sample],
                        self.required,
                        self.weights,
                    )
                    for sample in range(len(self.deltas))
                ],
                dtype=float,
            )
            missing = ~np.isfinite(drags)
            untrusted = ~np.all(trusted[candidate_index], axis=1)
            violations = missing | untrusted | self.manufacturing_invalid_flags
            finite = drags[np.isfinite(drags) & ~violations]
            penalty = float(np.max(finite) * 2) if len(finite) else 1.0
            drags[violations] = penalty
            objectives[candidate_index] = [float(np.mean(drags)), cvar(drags, 0.95)]
            sample_drags.append(drags.tolist())
            trust_violations.append(int(untrusted.sum()))
            missing_roots.append(int(missing.sum()))
        constraints, diagnostics = self._nominal_constraints(candidates)
        for index, diagnostic in enumerate(diagnostics):
            diagnostic["sample_weighted_drag"] = sample_drags[index]
            diagnostic["mean_lift_by_alpha"] = np.mean(cl[index], axis=0).tolist()
            diagnostic["lift_std_by_alpha"] = np.std(cl[index], axis=0, ddof=1).tolist()
            diagnostic["mean_cm_by_alpha"] = np.mean(cm[index], axis=0).tolist()
            diagnostic["uncertainty_trust_violations"] = trust_violations[index]
            diagnostic["missing_lift_roots"] = missing_roots[index]
            diagnostic["manufacturing_invalid_samples"] = int(
                self.manufacturing_invalid_flags.sum()
            )
            diagnostic["evaluation_sample_count"] = len(self.deltas)
        self.counters["trust_rejections"] += sum(trust_violations)
        self.counters["missing_lift_roots"] += sum(missing_roots)
        self.counters["manufacturing_invalid_samples"] += int(
            self.manufacturing_invalid_flags.sum() * len(candidates)
        )
        return objectives, constraints, diagnostics

    def deterministic_drag_objective(self, candidate: np.ndarray) -> float:
        candidate = np.asarray(candidate, dtype=float)
        prediction = self.ensemble.predict(_frame(candidate[None, :], self.alpha))
        drag = weighted_required_lift_drag(
            self.alpha,
            prediction["prediction_cl"].to_numpy(float),
            np.exp(prediction["prediction_log_cd"].to_numpy(float)),
            self.required,
            self.weights,
        )
        constraints, _ = self._nominal_constraints(candidate[None, :])
        penalty = float(np.sum(np.maximum(constraints[0], 0)) * 1_000)
        return (float(drag) if np.isfinite(drag) else 1.0) + penalty

    def ava_objective(self, candidate: np.ndarray, sensitivity_lambda: float) -> float:
        candidate = np.asarray(candidate, dtype=float)
        perturbed = candidate[None, :] + self.deltas
        predictions = self.ensemble.predict(_frame(perturbed, self.alpha))
        lift = predictions["prediction_cl"].to_numpy(float).reshape(
            len(self.deltas), len(self.alpha)
        )
        score = ava_lift_sensitivity(
            np.mean(lift, axis=0),
            np.std(lift, axis=0, ddof=1),
            np.full(len(self.alpha), 1 / len(self.alpha)),
            sensitivity_lambda,
        )
        constraints, _ = self._nominal_constraints(candidate[None, :])
        return score + float(np.sum(np.maximum(constraints[0], 0)) * 1_000)


class _RobustProblem(Problem):
    def __init__(self, evaluator: RobustCandidateEvaluator, bounds: DesignBounds):
        super().__init__(n_var=12, n_obj=2, n_ieq_constr=6, xl=bounds.lower, xu=bounds.upper)
        self.candidate_evaluator = evaluator

    def _evaluate(
        self, x: np.ndarray, out: dict[str, np.ndarray], *args: object, **kwargs: object
    ) -> None:
        objectives, constraints, _ = self.candidate_evaluator.evaluate(x)
        out["F"] = objectives
        out["G"] = constraints


class _StudyCallback(Callback):
    def __init__(self, evaluator: RobustCandidateEvaluator, reference_point: np.ndarray):
        super().__init__()
        self.evaluator = evaluator
        self.reference_point = reference_point
        self.history: list[dict[str, Any]] = []
        self.previous_front: np.ndarray | None = None
        self.previous_counters = evaluator.counters.copy()

    def notify(self, algorithm: Any) -> None:
        population = algorithm.pop
        objectives = np.asarray(population.get("F"), dtype=float)
        constraints = np.asarray(population.get("G"), dtype=float)
        designs = np.asarray(population.get("X"), dtype=float)
        feasible = np.all(constraints <= 0, axis=1)
        front = objectives[feasible]
        hypervolume = float(HV(ref_point=self.reference_point)(front)) if len(front) else 0.0
        movement: float | None = None
        if self.previous_front is not None and len(front) and len(self.previous_front):
            scale = np.maximum(np.ptp(np.vstack([front, self.previous_front]), axis=0), 1e-12)
            tree = cKDTree(self.previous_front / scale)
            movement = float(np.mean(tree.query(front / scale, k=1)[0]))
        current = self.evaluator.counters.copy()
        counter_delta = {key: current[key] - self.previous_counters[key] for key in current}
        self.previous_counters = current
        self.previous_front = front.copy()
        unique = len(np.unique(np.round(designs, decimals=10), axis=0))
        self.history.append(
            {
                "generation": int(algorithm.n_gen),
                "evaluations": int(algorithm.evaluator.n_eval),
                "population_count": len(population),
                "feasible_count": int(feasible.sum()),
                "feasible_fraction": float(np.mean(feasible)),
                "objective_min": np.min(objectives, axis=0).tolist(),
                "objective_median": np.median(objectives, axis=0).tolist(),
                "constraint_violation_mean": float(np.mean(np.maximum(constraints, 0))),
                "constraint_violation_max": float(np.max(np.maximum(constraints, 0))),
                "hypervolume": hypervolume,
                "front_movement": movement,
                "duplicate_count": len(designs) - unique,
                **counter_delta,
            }
        )


def _serialise_nsga_result(
    result: Any,
    evaluator: RobustCandidateEvaluator,
    callback: _StudyCallback,
    seed: int,
    sample_count: int,
    cache_context_hash: str,
) -> dict[str, Any]:
    if result.X is not None:
        x = np.atleast_2d(result.X)
        f = np.atleast_2d(result.F)
    elif result.pop is not None:
        x = np.atleast_2d(np.asarray(result.pop.get("X"), dtype=float))
        f = np.atleast_2d(np.asarray(result.pop.get("F"), dtype=float))
    else:
        x = np.empty((0, 12))
        f = np.empty((0, 2))
    diagnostics: list[dict[str, Any]] = []
    if len(x):
        _, constraints, diagnostics = evaluator.evaluate(x)
    else:
        constraints = np.empty((0, 6))
    return {
        "schema_version": 3,
        "cache_context_hash": cache_context_hash,
        "seed": seed,
        "sample_count": sample_count,
        "solutions": x.tolist(),
        "objectives": f.tolist(),
        "constraint_values": constraints.tolist(),
        "diagnostics": diagnostics,
        "solution_count": len(x),
        "returned_population_fallback": result.X is None and len(x) > 0,
        "history": callback.history,
        "evaluation_counters": evaluator.counters,
    }


def _agreement(runs: list[dict[str, Any]], bounds: DesignBounds) -> dict[str, Any]:
    feasible_runs: list[tuple[np.ndarray, np.ndarray]] = []
    counts: list[int] = []
    for run in runs:
        designs = np.asarray(run["solutions"], dtype=float)
        objectives = np.asarray(run["objectives"], dtype=float)
        constraints = np.asarray(run["constraint_values"], dtype=float)
        feasible = np.all(constraints <= 0, axis=1) if len(constraints) else np.zeros(0, dtype=bool)
        counts.append(int(feasible.sum()))
        if feasible.any():
            feasible_runs.append((designs[feasible], objectives[feasible]))
    all_objectives = (
        np.vstack([values for _, values in feasible_runs]) if feasible_runs else np.empty((0, 2))
    )
    design_distances: list[float] = []
    design_scale = np.maximum(bounds.upper - bounds.lower, 1e-12)
    objective_scale = (
        np.maximum(np.ptp(all_objectives, axis=0), 1e-12) if len(all_objectives) else np.ones(2)
    )
    for (design_a, _), (design_b, _) in combinations(feasible_runs, 2):
        left_to_right = cKDTree(design_b / design_scale).query(design_a / design_scale, k=1)[0]
        right_to_left = cKDTree(design_a / design_scale).query(design_b / design_scale, k=1)[0]
        design_distances.append(
            float(0.5 * (left_to_right.mean() + right_to_left.mean()))
        )
    objective_fronts = [
        np.asarray(run["objectives"], dtype=float)[
            np.all(np.asarray(run["constraint_values"], dtype=float) <= 0, axis=1)
        ]
        if len(run["constraint_values"])
        else np.empty((0, len(objective_scale)))
        for run in runs
    ]
    objective_metrics = front_agreement(
        objective_fronts,
        objective_scale,
        threshold=0.20,
        minimum_feasible_fraction=0.80,
    )
    objective_mean_distance = objective_metrics["mean_symmetric_distance"]
    return {
        "run_count": len(runs),
        "feasible_run_count": len(feasible_runs),
        "feasible_solution_counts": counts,
        "objective_mean": np.mean(all_objectives, axis=0).tolist() if len(all_objectives) else None,
        "objective_std": np.std(all_objectives, axis=0, ddof=1).tolist() if len(all_objectives) > 1 else None,
        "objective_min": np.min(all_objectives, axis=0).tolist() if len(all_objectives) else None,
        "objective_max": np.max(all_objectives, axis=0).tolist() if len(all_objectives) else None,
        "pairwise_design_distance_mean": float(np.mean(design_distances)) if design_distances else None,
        "pairwise_design_distance_max": float(np.max(design_distances)) if design_distances else None,
        "pairwise_objective_distance_mean": objective_mean_distance,
        "pairwise_objective_distance_max": max(
            objective_metrics["pairwise_symmetric_distances"], default=None
        ),
        "agreement_status": (
            "pass"
            if objective_metrics["passed"]
            else "fail"
            if feasible_runs
            else "insufficient_feasible_runs"
        ),
        "agreement_rule": "at least 80% feasible runs and mean normalized symmetric cross-front distance <= 0.20",
        "objective_agreement": objective_metrics,
        "objective_scale_context": "frozen_once_from_all_feasible_runs_for_this_profile",
    }


def _run_nsga_profile(
    name: str,
    profile: Nsga2Config,
    optimisation: OptimisationConfig,
    reference_parameters: np.ndarray,
    master_deltas: np.ndarray,
    master_invalid: np.ndarray,
    development_points: pd.DataFrame,
    ensemble: LoadedEnsemble,
    calibration: dict[str, Any],
    bounds: DesignBounds,
    output_root: Path,
    profile_index: int,
    ensemble_manifest_hash: str,
) -> dict[str, Any]:
    if profile.uncertainty_samples > len(master_deltas):
        raise ValueError(f"{name} requests more uncertainty samples than the common master set")
    sample_slice = slice(0, profile.uncertainty_samples)
    seeds = profile.independent_seeds or 1
    runs: list[dict[str, Any]] = []
    development_matrix = development_points[FEATURE_COLUMNS].to_numpy(float)
    cache_context_hash = hash_object(
        {
            "objective_schema_version": 2,
            "profile_name": name,
            "profile": profile.model_dump(mode="json"),
            "optimisation": optimisation.model_dump(mode="json"),
            "reference_parameters": reference_parameters.tolist(),
            "manufacturing_deltas_sha256": sha256_bytes(
                np.ascontiguousarray(
                    master_deltas[: profile.uncertainty_samples], dtype="<f8"
                ).tobytes()
            ),
            "manufacturing_invalid_sha256": sha256_bytes(
                np.ascontiguousarray(
                    master_invalid[: profile.uncertainty_samples], dtype=np.uint8
                ).tobytes()
            ),
            "development_features_sha256": sha256_bytes(
                np.ascontiguousarray(development_matrix, dtype="<f8").tobytes()
            ),
            "calibration": calibration,
            "bounds": {"lower": bounds.lower.tolist(), "upper": bounds.upper.tolist()},
            "ensemble_manifest_sha256": ensemble_manifest_hash,
        }
    )
    for seed_index in range(seeds):
        seed = optimisation.seed + 1_000 * (profile_index + 1) + seed_index
        seed_path = output_root / f"nsga2_{name}_seed_{seed}.json"
        evaluator = RobustCandidateEvaluator(
            reference_parameters,
            master_deltas[sample_slice],
            master_invalid[sample_slice],
            development_points,
            ensemble,
            calibration,
            optimisation,
        )
        if seed_path.is_file():
            cached = json.loads(seed_path.read_text(encoding="utf-8"))
            structurally_current = (
                cached.get("sample_count") == profile.uncertainty_samples
                and len(cached.get("history", [])) == profile.generations
            )
            if (
                structurally_current
                and cached.get("schema_version") == 3
                and cached.get("cache_context_hash") == cache_context_hash
            ):
                runs.append(cached)
                continue
            if structurally_current and cached.get("schema_version") == 2:
                cached_designs = np.asarray(cached.get("solutions", []), dtype=float)
                cached_objectives = np.asarray(cached.get("objectives", []), dtype=float)
                cached_constraints = np.asarray(cached.get("constraint_values", []), dtype=float)
                if len(cached_designs):
                    verified_objectives, verified_constraints, _ = evaluator.evaluate(
                        cached_designs
                    )
                    if np.allclose(
                        verified_objectives, cached_objectives, rtol=1e-7, atol=1e-9
                    ) and np.allclose(
                        verified_constraints, cached_constraints, rtol=1e-7, atol=1e-9
                    ):
                        cached["schema_version"] = 3
                        cached["cache_context_hash"] = cache_context_hash
                        seed_path.write_text(
                            json.dumps(cached, indent=2, sort_keys=True), encoding="utf-8"
                        )
                        runs.append(cached)
                        continue
        reference_objectives, _, _ = evaluator.evaluate(reference_parameters[None, :])
        reference_point = np.maximum(reference_objectives[0] * 2, reference_objectives[0] + 0.01)
        callback = _StudyCallback(evaluator, reference_point)
        problem = _RobustProblem(evaluator, bounds)
        rng = np.random.default_rng(seed)
        initial_population = rng.uniform(
            bounds.lower,
            bounds.upper,
            size=(profile.population, len(reference_parameters)),
        )
        initial_population[0] = reference_parameters
        observed_designs = development_points[PARAMETER_COLUMNS].drop_duplicates().to_numpy(float)
        within_bounds = np.all(
            (observed_designs >= bounds.lower) & (observed_designs <= bounds.upper), axis=1
        )
        observed_designs = observed_designs[within_bounds]
        if len(observed_designs):
            observed_constraints, _ = evaluator._nominal_constraints(observed_designs)
            observed_violation = np.sum(np.maximum(observed_constraints, 0), axis=1)
            observed_distance = np.linalg.norm(
                (observed_designs - reference_parameters)
                / np.maximum(bounds.upper - bounds.lower, 1e-12),
                axis=1,
            )
            order = np.lexsort((observed_distance, observed_violation))
            observed_seed_count = min(len(order), max(0, profile.population // 2 - 1))
            initial_population[1 : observed_seed_count + 1] = observed_designs[
                order[:observed_seed_count]
            ]
        result = minimize(
            problem,
            NSGA2(pop_size=profile.population, sampling=initial_population),
            ("n_gen", profile.generations),
            seed=seed,
            callback=callback,
            verbose=False,
        )
        serialised = _serialise_nsga_result(
            result,
            evaluator,
            callback,
            seed,
            profile.uncertainty_samples,
            cache_context_hash,
        )
        seed_path.write_text(json.dumps(serialised, indent=2, sort_keys=True), encoding="utf-8")
        runs.append(serialised)
    return {
        "population": profile.population,
        "generations": profile.generations,
        "uncertainty_samples": profile.uncertainty_samples,
        "independent_seeds": seeds,
        "runs": runs,
        "agreement": _agreement(runs, bounds),
    }


def _full_selection_pool(
    full_runs: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, str]:
    designs: list[np.ndarray] = []
    objectives: list[np.ndarray] = []
    constraint_values: list[np.ndarray] = []
    for run in full_runs:
        run_designs = np.asarray(run["solutions"], dtype=float)
        run_objectives = np.asarray(run["objectives"], dtype=float)
        constraints = np.asarray(run["constraint_values"], dtype=float)
        if len(run_designs):
            designs.append(run_designs)
            objectives.append(run_objectives)
            constraint_values.append(constraints)
    if not designs:
        raise RuntimeError("Full NSGA-II produced no candidate solutions")
    all_designs = np.vstack(designs)
    all_objectives = np.vstack(objectives)
    all_constraints = np.vstack(constraint_values)
    feasible = np.all(all_constraints <= 0, axis=1)
    if feasible.any():
        return all_designs[feasible], all_objectives[feasible], "feasible"
    violation = np.sum(np.maximum(all_constraints, 0), axis=1)
    count = min(32, len(all_designs))
    least_infeasible = np.argsort(violation)[:count]
    return (
        all_designs[least_infeasible],
        all_objectives[least_infeasible],
        "least_constraint_violation",
    )


def _select_knee(designs: np.ndarray, objectives: np.ndarray) -> np.ndarray:
    ideal = objectives.min(axis=0)
    scale = np.maximum(objectives.max(axis=0) - ideal, 1e-12)
    normalized = (objectives - ideal) / scale
    return designs[int(np.argmin(np.linalg.norm(normalized, axis=1)))]


def _five_design_comparison(
    reference_parameters: np.ndarray,
    deterministic_parameters: np.ndarray,
    ava: dict[str, Any],
    nsga: dict[str, Any],
    final_evaluator: RobustCandidateEvaluator,
) -> list[dict[str, Any]]:
    full_designs, full_objectives, selection_pool = _full_selection_pool(
        nsga["full"]["runs"]
    )
    robust_index = int(np.argmin(full_objectives[:, 1]))
    knee = _select_knee(full_designs, full_objectives)
    ava_key = "1.0" if "1.0" in ava else min(ava, key=lambda key: abs(float(key) - 1.0))
    definitions = [
        ("verified_reference", "verified_reference", reference_parameters),
        ("deterministic_optimum", "deterministic_expected_drag", deterministic_parameters),
        ("ava_robust_optimum", f"ava_lambda_{ava_key}", np.asarray(ava[ava_key]["parameters"], dtype=float)),
        ("smooth_correlated_robust_optimum", "full_nsga2_minimum_cvar", full_designs[robust_index]),
        ("trust_constrained_pareto_knee", "full_nsga2_knee", knee),
    ]
    result: list[dict[str, Any]] = []
    for design_id, source, parameters in definitions:
        objectives, constraints, diagnostics = final_evaluator.evaluate(parameters[None, :])
        upper_y, lower_y = reconstruct_cst(parameters, final_evaluator.x)
        result.append(
            {
                "design_id": design_id,
                "source": source,
                "selection_pool": selection_pool if design_id.startswith(("smooth_", "trust_")) else "not_applicable",
                "parameters": parameters.tolist(),
                "coordinates": {
                    "upper": np.column_stack([final_evaluator.x, upper_y]).tolist(),
                    "lower": np.column_stack([final_evaluator.x, lower_y]).tolist(),
                },
                "expected_weighted_cd": float(objectives[0, 0]),
                "cvar_95_weighted_cd": float(objectives[0, 1]),
                "constraint_values": constraints[0].tolist(),
                "feasible": bool(np.all(constraints[0] <= 0)),
                "diagnostics": diagnostics[0],
                "evaluation_sample_count": len(final_evaluator.deltas),
            }
        )
    return result


def run_optimisation_studies(
    reference_parameters: np.ndarray,
    bounds: DesignBounds,
    development_points: pd.DataFrame,
    ensemble_root: Path,
    calibration: dict[str, Any],
    optimisation: OptimisationConfig,
    uncertainty: UncertaintyConfig,
    upper: np.ndarray,
    lower: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    ensemble = LoadedEnsemble(ensemble_root)
    maximum_optimisation_samples = max(
        optimisation.debug_nsga2.uncertainty_samples,
        optimisation.pilot_nsga2.uncertainty_samples,
        optimisation.full_nsga2.uncertainty_samples,
    )
    master_deltas, master_invalid, delta_summary = manufacturing_parameter_deltas(
        reference_parameters,
        upper,
        lower,
        uncertainty,
        count=maximum_optimisation_samples,
        seed=uncertainty.seed + 2_000,
    )
    development_parameters = development_points[
        PARAMETER_COLUMNS
    ].drop_duplicates().to_numpy(float)
    development_parameter_std = development_parameters.std(axis=0, ddof=1)
    ava_latent = sobol_normal_samples(
        maximum_optimisation_samples,
        len(reference_parameters),
        uncertainty.seed + 1_000,
        antithetic=True,
    )
    ava_uniform = 2 * norm.cdf(ava_latent) - 1
    ava_scales = np.where(
        np.abs(reference_parameters) > 1e-6,
        np.abs(reference_parameters) * optimisation.ava_baseline.uncertainty_fraction,
        np.maximum(development_parameter_std, 1e-12)
        * optimisation.ava_baseline.uncertainty_fraction,
    )
    ava_deltas = ava_uniform * ava_scales[None, :]
    ava_invalid = np.zeros(len(ava_deltas), dtype=bool)
    ava_evaluator = RobustCandidateEvaluator(
        reference_parameters,
        ava_deltas,
        ava_invalid,
        development_points,
        ensemble,
        calibration,
        optimisation,
    )
    bound_pairs = list(zip(bounds.lower, bounds.upper, strict=True))
    deterministic_result = differential_evolution(
        ava_evaluator.deterministic_drag_objective,
        bound_pairs,
        seed=optimisation.seed + 900,
        maxiter=48,
        popsize=10,
        polish=True,
        workers=1,
        updating="immediate",
    )
    deterministic_objectives, deterministic_constraints, deterministic_diagnostics = ava_evaluator.evaluate(
        deterministic_result.x[None, :]
    )
    deterministic = {
        "parameters": deterministic_result.x.tolist(),
        "nominal_objective": float(deterministic_result.fun),
        "robust_drag_objectives": deterministic_objectives[0].tolist(),
        "constraints": deterministic_constraints[0].tolist(),
        "diagnostics": deterministic_diagnostics[0],
        "evaluations": int(deterministic_result.nfev),
    }
    ava: dict[str, Any] = {}
    for index, sensitivity_lambda in enumerate(optimisation.ava_baseline.lambda_sensitivity_values):
        result = differential_evolution(
            lambda candidate, value=sensitivity_lambda: ava_evaluator.ava_objective(
                candidate, value
            ),
            bound_pairs,
            seed=optimisation.seed + index,
            maxiter=24,
            popsize=8,
            polish=False,
            workers=1,
            updating="immediate",
        )
        objectives, constraints, diagnostics = ava_evaluator.evaluate(result.x[None, :])
        ava[str(sensitivity_lambda)] = {
            "parameters": result.x.tolist(),
            "ava_objective": float(result.fun),
            "robust_drag_objectives": objectives[0].tolist(),
            "constraints": constraints[0].tolist(),
            "diagnostics": diagnostics[0],
            "evaluations": int(result.nfev),
        }
    profiles = {
        "debug": optimisation.debug_nsga2,
        "pilot": optimisation.pilot_nsga2,
        "full": optimisation.full_nsga2,
    }
    nsga = {
        name: _run_nsga_profile(
            name,
            profile,
            optimisation,
            reference_parameters,
            master_deltas,
            master_invalid,
            development_points,
            ensemble,
            calibration,
            bounds,
            output_root,
            index,
            sha256_file(ensemble_root / "ensemble_manifest.json"),
        )
        for index, (name, profile) in enumerate(profiles.items())
    }
    final_deltas, final_invalid, final_delta_summary = manufacturing_parameter_deltas(
        reference_parameters,
        upper,
        lower,
        uncertainty,
        count=uncertainty.final_evaluation_samples.count,
        seed=uncertainty.seed + 9_000,
    )
    final_evaluator = RobustCandidateEvaluator(
        reference_parameters,
        final_deltas,
        final_invalid,
        development_points,
        ensemble,
        calibration,
        optimisation,
    )
    comparison = _five_design_comparison(
        reference_parameters,
        deterministic_result.x,
        ava,
        nsga,
        final_evaluator,
    )
    payload = {
        "schema_version": 2,
        "device": str(ensemble.device),
        "manufacturing_common_random_numbers": delta_summary,
        "ava_independent_cst_common_random_numbers": {
            "requested": len(ava_deltas),
            "valid": len(ava_deltas),
            "invalid": 0,
            "common_random_numbers": True,
            "uncertainty_fraction": optimisation.ava_baseline.uncertainty_fraction,
            "near_zero_rule": "fraction times development-set parameter standard deviation",
            "parameter_delta_sha256": sha256_bytes(
                np.ascontiguousarray(ava_deltas, dtype="<f8").tobytes()
            ),
        },
        "final_evaluation_common_random_numbers": final_delta_summary,
        "deterministic_drag_optimisation": deterministic,
        "ava_lambda_sweep": ava,
        "nsga2": nsga,
        "five_design_comparison": comparison,
        "reference_parameters": reference_parameters.tolist(),
        "bounds": {"lower": bounds.lower.tolist(), "upper": bounds.upper.tolist()},
    }
    (output_root / "selected_designs.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "optimisation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
