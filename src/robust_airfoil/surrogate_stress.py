"""Reviewer-driven adversarial surrogate-trust probing.

Reviewer 2 of the prior submission asked for an explicit optimisation that
searches the design space for the aerofoil maximising the disagreement between
the machine-learning surrogate and XFOIL under the same CST parameterisation.
This module implements that search and, in the same pass, relates the observed
disagreement to the calibrated surrogate trust domain, which addresses the
reviewer request to establish where the surrogate can and cannot be trusted.

The disagreement metrics are pure functions so they can be unit tested without
invoking XFOIL. The search itself keeps XFOIL strictly as an external oracle and
never lets a surrogate proxy stand in for a direct-solver evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.normalise import resample_surfaces_to_common_x
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import sha256_bytes
from robust_airfoil.modelling.calibrate import apply_trust_model
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.provenance import write_json_atomic
from robust_airfoil.validation.xfoil_runner import XFoilCase, run_xfoil

PARAMETER_COLUMNS = FEATURE_COLUMNS[:-1]


def paired_disagreement(
    alpha_deg: np.ndarray,
    cl_surrogate: np.ndarray,
    cd_surrogate: np.ndarray,
    xfoil_points: pd.DataFrame,
    tolerance_deg: float = 1e-6,
) -> dict[str, Any]:
    """Aggregate surrogate-vs-XFOIL disagreement over jointly converged angles.

    Lift disagreement is absolute in ``CL``; drag disagreement is relative to the
    XFOIL value because ``CD`` spans an order of magnitude across the polar. Only
    angles at which XFOIL produced a finite, positive-drag solution contribute,
    so an unconverged direct solve can never be silently scored as agreement.
    """
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    cl_surrogate = np.asarray(cl_surrogate, dtype=float)
    cd_surrogate = np.asarray(cd_surrogate, dtype=float)
    if not (len(alpha_deg) == len(cl_surrogate) == len(cd_surrogate)):
        raise ValueError("Surrogate arrays must share the requested alpha grid length")
    if not len(alpha_deg):
        raise ValueError("At least one requested angle is required")

    required = {"alpha_deg", "cl", "cd"}
    if not required.issubset(xfoil_points.columns):
        return {
            "matched_points": 0,
            "requested_points": int(len(alpha_deg)),
            "mean_absolute_cl_gap": None,
            "max_absolute_cl_gap": None,
            "mean_relative_cd_gap": None,
            "max_relative_cd_gap": None,
            "matched_alpha_deg": [],
        }

    xf: pd.DataFrame = xfoil_points.dropna(subset=["alpha_deg", "cl", "cd"]).copy()
    finite_mask = np.isfinite(xf[["alpha_deg", "cl", "cd"]].to_numpy(float)).all(axis=1)
    positive_drag = xf["cd"].to_numpy(float) > 0
    xf = xf.loc[finite_mask & positive_drag]
    xf_alpha = xf["alpha_deg"].to_numpy(float)
    cl_gaps: list[float] = []
    cd_gaps: list[float] = []
    matched_alpha: list[float] = []
    for index, target in enumerate(alpha_deg):
        hit = xf.loc[np.abs(xf_alpha - target) <= tolerance_deg]
        if hit.empty:
            continue
        row = hit.iloc[0]
        cl_gaps.append(abs(float(cl_surrogate[index]) - float(row["cl"])))
        cd_gaps.append(abs(float(cd_surrogate[index]) - float(row["cd"])) / float(row["cd"]))
        matched_alpha.append(float(target))
    if not matched_alpha:
        return {
            "matched_points": 0,
            "requested_points": int(len(alpha_deg)),
            "mean_absolute_cl_gap": None,
            "max_absolute_cl_gap": None,
            "mean_relative_cd_gap": None,
            "max_relative_cd_gap": None,
            "matched_alpha_deg": [],
        }
    return {
        "matched_points": len(matched_alpha),
        "requested_points": int(len(alpha_deg)),
        "mean_absolute_cl_gap": float(np.mean(cl_gaps)),
        "max_absolute_cl_gap": float(np.max(cl_gaps)),
        "mean_relative_cd_gap": float(np.mean(cd_gaps)),
        "max_relative_cd_gap": float(np.max(cd_gaps)),
        "matched_alpha_deg": matched_alpha,
    }


def sobol_design_pool(bounds: DesignBounds, count: int, seed: int) -> np.ndarray:
    """Scrambled Sobol design pool mapped into the declared CST design box."""
    lower = np.asarray(bounds.lower, dtype=float)
    upper = np.asarray(bounds.upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("Design bounds must be one-dimensional and consistent")
    if np.any(upper <= lower):
        raise ValueError("Every design coordinate must have a positive width")
    if count <= 0:
        raise ValueError("A positive candidate count is required")
    sampler = qmc.Sobol(d=len(lower), scramble=True, seed=seed)
    unit = sampler.random(count)
    return lower + unit * (upper - lower)


@dataclass
class _Candidate:
    origin: str
    parameters: np.ndarray


def _prediction_frame(parameters: np.ndarray, alpha: np.ndarray, airfoil_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        np.repeat(parameters[None, :], len(alpha), axis=0),
        columns=pd.Index(PARAMETER_COLUMNS),
    )
    frame["alpha_deg"] = alpha
    frame["airfoil_id"] = airfoil_id
    return frame


def _write_geometry(path: Path, name: str, upper: np.ndarray, lower: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.vstack([upper[::-1], lower[1:]])
    lines = [f"_{name}", *(f"{x:.12f} {y:.12f}" for x, y in coordinates)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_candidate(
    candidate: _Candidate,
    candidate_id: str,
    ensemble: LoadedEnsemble,
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
    alpha: np.ndarray,
    cosine_x: np.ndarray,
    max_cst_refit_error: float,
    executable: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    parameters = np.asarray(candidate.parameters, dtype=float)
    upper_y, lower_y = reconstruct_cst(parameters, cosine_x)
    upper = np.column_stack([cosine_x, upper_y])
    lower = np.column_stack([cosine_x, lower_y])
    validity = validate_geometry(upper, lower)
    fit_upper, fit_lower = resample_surfaces_to_common_x(upper, lower, cosine_x)
    refit = fit_cst(fit_upper, fit_lower)
    if not validity.valid or refit.max_error > max_cst_refit_error:
        return {
            "candidate_id": candidate_id,
            "origin": candidate.origin,
            "status": "rejected_geometry",
            "invalid_reasons": list(validity.reasons),
            "cst_max_refit_error": float(refit.max_error),
            "parameters": parameters.tolist(),
        }

    predictions = ensemble.predict(_prediction_frame(parameters, alpha, candidate_id))
    trusted = apply_trust_model(development_points, predictions, ensemble.first_scaling, calibration)
    cl_surrogate = trusted["prediction_cl"].to_numpy(float)
    cd_surrogate = np.exp(trusted["prediction_log_cd"].to_numpy(float))
    trusted_fraction = float(trusted["trusted_domain"].mean())
    mean_support_distance = float(trusted["support_distance"].mean())

    geometry_path = output_dir / candidate_id / "geometry.dat"
    _write_geometry(geometry_path, candidate_id, upper, lower)
    case = XFoilCase(
        geometry_path,
        1_000_000,
        0.0,
        9.0,
        float(alpha[0]),
        float(alpha[-1]),
        float(alpha[1] - alpha[0]),
        100,
        "positive",
    )
    result = run_xfoil(executable, case, output_dir / candidate_id, timeout_seconds)
    disagreement = paired_disagreement(alpha, cl_surrogate, cd_surrogate, result.parsed_points)
    return {
        "candidate_id": candidate_id,
        "origin": candidate.origin,
        "status": "evaluated",
        "parameters": parameters.tolist(),
        "cst_max_refit_error": float(refit.max_error),
        "trusted_fraction": trusted_fraction,
        "mean_support_distance": mean_support_distance,
        "xfoil_status": result.status,
        "xfoil_converged_points": int(len(result.parsed_points)),
        "surrogate_cl": cl_surrogate.tolist(),
        "surrogate_cd": cd_surrogate.tolist(),
        "disagreement": disagreement,
    }


def run_surrogate_stress(
    ensemble_root: Path,
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
    bounds: DesignBounds,
    reference_parameters: np.ndarray,
    executable: Path,
    output_dir: Path,
    *,
    alpha_deg: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
    pool: int = 48,
    refine_top: int = 6,
    refine_scale: float = 0.15,
    max_cst_refit_error: float = 0.005,
    seed: int = 20260908,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Search the CST design box for maximum surrogate-vs-XFOIL disagreement."""
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = np.asarray(alpha_deg, dtype=float)
    cosine_x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    ensemble = LoadedEnsemble(ensemble_root)
    lower = np.asarray(bounds.lower, dtype=float)
    upper = np.asarray(bounds.upper, dtype=float)
    span = upper - lower

    pool_parameters = sobol_design_pool(bounds, pool, seed)
    candidates: list[_Candidate] = [_Candidate("reference", np.asarray(reference_parameters, dtype=float))]
    candidates += [_Candidate(f"sobol_{i:03d}", pool_parameters[i]) for i in range(len(pool_parameters))]

    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        record = _evaluate_candidate(
            candidate,
            f"probe_{index:03d}",
            ensemble,
            development_points,
            calibration,
            alpha,
            cosine_x,
            max_cst_refit_error,
            executable,
            output_dir,
            timeout_seconds,
        )
        if record is not None:
            records.append(record)

    def cl_gap(record: dict[str, Any]) -> float:
        gap = record.get("disagreement", {}).get("mean_absolute_cl_gap")
        return float(gap) if gap is not None else -1.0

    evaluated = [record for record in records if record["status"] == "evaluated"]
    evaluated.sort(key=cl_gap, reverse=True)

    rng = np.random.default_rng(seed + 101)
    refine_seeds = [record for record in evaluated[:refine_top] if cl_gap(record) > 0]
    for parent_index, parent in enumerate(refine_seeds):
        base = np.asarray(parent["parameters"], dtype=float)
        for child in range(2):
            perturbed = np.clip(
                base + rng.normal(0.0, refine_scale, size=len(base)) * span,
                lower,
                upper,
            )
            record = _evaluate_candidate(
                _Candidate(f"refine_from_{parent['candidate_id']}", perturbed),
                f"refine_{parent_index:02d}_{child:02d}",
                ensemble,
                development_points,
                calibration,
                alpha,
                cosine_x,
                max_cst_refit_error,
                executable,
                output_dir,
                timeout_seconds,
            )
            if record is not None:
                records.append(record)

    evaluated = [record for record in records if record["status"] == "evaluated"]
    with_gap = [record for record in evaluated if cl_gap(record) >= 0]
    with_gap.sort(key=cl_gap, reverse=True)

    support = np.asarray([record["mean_support_distance"] for record in with_gap], dtype=float)
    cl_gaps = np.asarray([cl_gap(record) for record in with_gap], dtype=float)
    trusted_fraction = np.asarray([record["trusted_fraction"] for record in with_gap], dtype=float)
    fully_trusted = trusted_fraction >= 0.999
    correlation = (
        float(spearmanr(support, cl_gaps).statistic)
        if len(with_gap) >= 3 and np.ptp(support) > 0 and np.ptp(cl_gaps) > 0
        else None
    )
    worst = with_gap[0] if with_gap else None
    worst_trusted = next((record for record in with_gap if record["trusted_fraction"] >= 0.999), None)

    payload: dict[str, Any] = {
        "schema_version": "robust-v2-surrogate-stress-v1",
        "purpose": "Reviewer-directed adversarial surrogate-vs-XFOIL disagreement search",
        "executable": executable.as_posix(),
        "reynolds_number": 1_000_000,
        "mach": 0.0,
        "ncrit": 9.0,
        "alpha_deg": list(alpha_deg),
        "design_bounds_sha256": sha256_bytes(
            np.ascontiguousarray(np.vstack([lower, upper]), dtype="<f8").tobytes()
        ),
        "requested_candidates": len(candidates),
        "evaluated_candidates": len(evaluated),
        "rejected_geometry": sum(1 for r in records if r["status"] == "rejected_geometry"),
        "xfoil_convergence": {
            "fully_converged": int(sum(1 for r in evaluated if r["xfoil_status"] == "ok")),
            "partial_or_failed": int(sum(1 for r in evaluated if r["xfoil_status"] != "ok")),
        },
        "support_distance_vs_cl_gap_spearman": correlation,
        "mean_cl_gap_fully_trusted": (
            float(np.mean(cl_gaps[fully_trusted])) if fully_trusted.any() else None
        ),
        "mean_cl_gap_outside_trust": (
            float(np.mean(cl_gaps[~fully_trusted])) if (~fully_trusted).any() else None
        ),
        "worst_case_overall": worst,
        "worst_case_within_trust_domain": worst_trusted,
        "records": records,
    }
    write_json_atomic(output_dir / "surrogate_stress_summary.json", payload)
    return payload
