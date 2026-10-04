from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from robust_airfoil.config import OptimisationConfig, UncertaintyConfig
from robust_airfoil.data.polar_parser import parse_polar_file
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.normalise import normalise_geometry, resample_surfaces_to_common_x
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import sha256_bytes, sha256_file
from robust_airfoil.modelling.calibrate import apply_trust_model
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.optimisation.objectives import weighted_required_lift_drag
from robust_airfoil.provenance import write_json_atomic
from robust_airfoil.uncertainty.fields import PerturbedGeometry, smooth_normal_perturbation
from robust_airfoil.uncertainty.sampling import sobol_normal_samples


@dataclass(frozen=True)
class XFoilCase:
    geometry_path: Path
    reynolds_number: float
    mach: float
    ncrit: float
    alpha_start: float
    alpha_end: float
    alpha_step: float
    iterations: int
    mode: str


@dataclass
class XFoilResult:
    status: str
    return_code: int | None
    timed_out: bool
    command_text: str
    stdout_path: Path
    stderr_path: Path
    polar_path: Path | None
    parsed_points: pd.DataFrame
    failure_reason: str | None


def build_xfoil_commands(case: XFoilCase, geometry_name: str, polar_name: str) -> str:
    if case.alpha_step == 0:
        raise ValueError("alpha_step cannot be zero")
    if case.alpha_start < case.alpha_end and case.alpha_step < 0:
        raise ValueError("positive sweep requires positive step")
    if case.alpha_start > case.alpha_end and case.alpha_step > 0:
        raise ValueError("negative sweep requires negative step")
    return "\n".join([
        f"LOAD {geometry_name}", "PANE", "OPER",
        f"VISC {case.reynolds_number:.12g}", f"MACH {case.mach:.12g}", "VPAR", f"N {case.ncrit:.12g}", "",
        f"ITER {case.iterations}", "PACC", polar_name, "",
        f"ASEQ {case.alpha_start:.12g} {case.alpha_end:.12g} {case.alpha_step:.12g}",
        "PACC", "", "QUIT", "",
    ])


def _requested_alpha_grid(case: XFoilCase) -> np.ndarray:
    values = np.asarray([case.alpha_start, case.alpha_end, case.alpha_step], dtype=float)
    if not np.isfinite(values).all() or case.alpha_step == 0:
        raise ValueError("Finite sweep endpoints and a nonzero step are required")
    steps = (case.alpha_end - case.alpha_start) / case.alpha_step
    if steps < 0 or not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError("Sweep endpoints must lie on the requested directed alpha grid")
    return case.alpha_start + np.arange(int(round(steps)) + 1) * case.alpha_step


def _sweep_completion(points: pd.DataFrame, case: XFoilCase) -> tuple[int, float, bool]:
    targets = _requested_alpha_grid(case)
    if not len(points) or "alpha_deg" not in points.columns:
        return len(targets), 0.0, False
    alpha = points["alpha_deg"].to_numpy(float)
    alpha = alpha[np.isfinite(alpha)]
    # XFOIL text polars usually print alpha to 3 decimals. This permits half a
    # final printed unit plus roundoff, never one whole requested alpha step.
    tolerance = min(abs(case.alpha_step) * 0.01, 5.01e-4)
    matched = np.asarray([np.any(np.abs(alpha - target) <= tolerance) for target in targets])
    return len(targets), float(matched.mean()), bool(matched[0] and matched[-1])


def _valid_aero_points(points: pd.DataFrame) -> pd.DataFrame:
    """Keep only complete, finite, positive-drag aerodynamic rows.

    Coverage and success must never be credited to a printed but non-converged
    row (NaN or missing coefficient) or to a physically impossible nonpositive
    drag value.
    """
    if not len(points) or not {"alpha_deg", "cl", "cd", "cm"}.issubset(points.columns):
        return points.iloc[0:0]
    finite = np.isfinite(points[["alpha_deg", "cl", "cd", "cm"]].to_numpy(float)).all(axis=1)
    return points.loc[finite & (points["cd"].to_numpy(float) > 0)]


def _unresolved_alphas(points: pd.DataFrame, case: XFoilCase) -> np.ndarray:
    """Return requested targets not represented by a valid direct-solver row."""
    targets = _requested_alpha_grid(case)
    if not len(points) or "alpha_deg" not in points.columns:
        return targets
    alpha = points["alpha_deg"].to_numpy(float)
    alpha = alpha[np.isfinite(alpha)]
    tolerance = min(abs(case.alpha_step) * 0.01, 5.01e-4)
    return targets[
        ~np.asarray([np.any(np.abs(alpha - target) <= tolerance) for target in targets])
    ]


def _segment_status(
    return_code: int, points: pd.DataFrame, case: XFoilCase
) -> tuple[str, str | None]:
    required_columns = ("alpha_deg", "cl", "cd", "cm")
    missing_columns = sorted(set(required_columns) - set(points.columns))
    valid_points = points
    if missing_columns:
        valid_points = pd.DataFrame()
    else:
        valid = np.isfinite(points[list(required_columns)].to_numpy(float)).all(axis=1)
        valid &= points["cd"].to_numpy(float) > 0
        valid_points = points.loc[valid]
    expected, coverage, span_complete = _sweep_completion(valid_points, case)
    status = (
        "ok" if return_code == 0 and coverage == 1.0 and span_complete else "failed"
    )
    reason = (
        None
        if status == "ok"
        else (
            f"incomplete segment: {len(points)}/{expected} points, "
            f"coverage={coverage:.3f}, span_complete={span_complete}, "
            f"return_code={return_code}, missing_columns={missing_columns}"
        )
    )
    return status, reason


def _run_xfoil_segment(
    executable: Path,
    case: XFoilCase,
    output_dir: Path,
    timeout_seconds: int,
) -> XFoilResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    command_path = output_dir / "command.txt"
    with tempfile.TemporaryDirectory(prefix="robust-xfoil-") as temporary_text:
        temporary = Path(temporary_text)
        geometry_copy = temporary / "geometry.dat"
        polar = temporary / "polar.txt"
        shutil.copy2(case.geometry_path, geometry_copy)
        command = build_xfoil_commands(case, geometry_copy.name, polar.name)
        command_path.write_text(command, encoding="utf-8")
        try:
            completed = subprocess.run(
                [str(executable)], input=command, cwd=temporary, capture_output=True, text=True,
                timeout=timeout_seconds, check=False, encoding="utf-8", errors="replace",
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            if polar.exists():
                final_polar = output_dir / "polar.txt"
                shutil.copy2(polar, final_polar)
                try:
                    parsed = parse_polar_file(final_polar).points
                    status, reason = _segment_status(completed.returncode, parsed, case)
                except Exception as exc:
                    parsed, status, reason = pd.DataFrame(), "failed", repr(exc)
                return XFoilResult(status, completed.returncode, False, command, stdout_path, stderr_path, final_polar, parsed, reason)
            return XFoilResult("failed", completed.returncode, False, command, stdout_path, stderr_path, None, pd.DataFrame(), "polar output missing")
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(str(exc.stdout or ""), encoding="utf-8")
            stderr_path.write_text(str(exc.stderr or ""), encoding="utf-8")
            timeout_polar: Path | None = None
            parsed = pd.DataFrame()
            if polar.exists():
                timeout_polar = output_dir / "polar.txt"
                shutil.copy2(polar, timeout_polar)
                try:
                    parsed = parse_polar_file(timeout_polar).points
                except Exception:
                    parsed = pd.DataFrame()
            return XFoilResult(
                "timeout",
                None,
                True,
                command,
                stdout_path,
                stderr_path,
                timeout_polar,
                parsed,
                "timeout",
            )


def run_xfoil(
    executable: Path,
    case: XFoilCase,
    output_dir: Path,
    timeout_seconds: int = 240,
) -> XFoilResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = _requested_alpha_grid(case)
    chunks = [alpha[index : index + 12] for index in range(0, len(alpha), 12)]
    segment_timeout = max(15, timeout_seconds // max(1, len(chunks)))
    results: list[XFoilResult] = []
    for index, chunk in enumerate(chunks):
        segment_case = XFoilCase(
            case.geometry_path,
            case.reynolds_number,
            case.mach,
            case.ncrit,
            float(chunk[0]),
            float(chunk[-1]),
            case.alpha_step,
            case.iterations,
            case.mode,
        )
        results.append(
            _run_xfoil_segment(
                executable,
                segment_case,
                output_dir / "segments" / f"{index:03d}",
                segment_timeout,
            )
        )
    frames = [result.parsed_points for result in results if len(result.parsed_points)]
    parsed = (
        pd.concat(frames, ignore_index=True)
        .sort_values("alpha_deg")
        .drop_duplicates("alpha_deg", keep="first")
        if frames
        else pd.DataFrame()
    )
    # Chunking bounds individual process time, but each chunk cold-starts XFOIL.
    # Retry only missing targets as independent, auditable higher-iteration solves.
    unresolved = _unresolved_alphas(parsed, case)
    retry_timeout = max(15, timeout_seconds // max(1, len(unresolved)))
    retry_results: list[XFoilResult] = []
    for index, alpha in enumerate(unresolved):
        retry_case = XFoilCase(
            case.geometry_path,
            case.reynolds_number,
            case.mach,
            case.ncrit,
            float(alpha),
            float(alpha),
            case.alpha_step,
            max(100, case.iterations * 2),
            case.mode,
        )
        retry_results.append(
            _run_xfoil_segment(
                executable,
                retry_case,
                output_dir / "retries" / f"alpha_{alpha:+08.3f}_{index:03d}",
                retry_timeout,
            )
        )
    retry_frames = [result.parsed_points for result in retry_results if len(result.parsed_points)]
    if retry_frames:
        parsed = (
            pd.concat([parsed, *retry_frames], ignore_index=True)
            .sort_values("alpha_deg")
            .drop_duplicates("alpha_deg", keep="first")
        )
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    command_path = output_dir / "command.txt"
    stdout_path.write_text(
        "\n\n".join(
            f"===== SEGMENT {index:03d} =====\n{result.stdout_path.read_text(encoding='utf-8')}"
            for index, result in enumerate(results)
        )
        + "\n\n"
        + "\n\n".join(
            f"===== RETRY {index:03d} =====\n{result.stdout_path.read_text(encoding='utf-8')}"
            for index, result in enumerate(retry_results)
        ),
        encoding="utf-8",
    )
    stderr_path.write_text(
        "\n\n".join(
            f"===== SEGMENT {index:03d} =====\n{result.stderr_path.read_text(encoding='utf-8')}"
            for index, result in enumerate(results)
        )
        + "\n\n"
        + "\n\n".join(
            f"===== RETRY {index:03d} =====\n{result.stderr_path.read_text(encoding='utf-8')}"
            for index, result in enumerate(retry_results)
        ),
        encoding="utf-8",
    )
    command_text = "\n\n".join(
        f"===== SEGMENT {index:03d} =====\n{result.command_text}"
        for index, result in enumerate(results)
    )
    if retry_results:
        command_text += "\n\n" + "\n\n".join(
            f"===== RETRY {index:03d} =====\n{result.command_text}"
            for index, result in enumerate(retry_results)
        )
    command_path.write_text(command_text, encoding="utf-8")
    combined_path: Path | None = None
    if len(parsed):
        combined_path = output_dir / "combined_points.csv"
        parsed.to_csv(combined_path, index=False)
    failed_segments = [index for index, result in enumerate(results) if result.status != "ok"]
    failed_retries = [index for index, result in enumerate(retry_results) if result.status != "ok"]
    all_results = [*results, *retry_results]
    return_codes = [result.return_code for result in all_results if result.return_code is not None]
    clean_exit = (
        len(return_codes) == len(all_results)
        and not any(result.timed_out for result in all_results)
        and all(code == 0 for code in return_codes)
    )
    # Coverage must be assessed only over rows that are complete, finite aerodynamic
    # solutions with positive drag; a printed but non-converged row, or a clean grid
    # reached under a nonzero solver exit or timeout, can never count as success.
    valid_parsed = _valid_aero_points(parsed)
    expected_points, convergence_fraction, span_complete = _sweep_completion(valid_parsed, case)
    status = (
        "ok"
        if convergence_fraction == 1.0 and span_complete and clean_exit and not failed_retries
        else "failed"
    )
    reason = (
        None
        if status == "ok"
        else (
            f"incomplete sweep: {len(valid_parsed)}/{expected_points} valid points, "
            f"coverage={convergence_fraction:.3f}, span_complete={span_complete}, "
            f"clean_exit={clean_exit}, failed_segments={failed_segments}, "
            f"failed_retries={failed_retries}"
        )
    )
    return XFoilResult(
        status,
        0 if clean_exit else None,
        any(result.timed_out for result in all_results),
        command_text,
        stdout_path,
        stderr_path,
        combined_path,
        parsed,
        reason,
    )


def _comparison(local: pd.DataFrame, archive_path: Path | None) -> dict[str, object]:
    if not {"alpha_deg", "cl", "cd", "cm"}.issubset(local.columns):
        return {
            "local_converged_points": 0,
            "local_alpha_min": None,
            "local_alpha_max": None,
            "local_max_lift_to_drag": None,
            "archive_comparison": "unavailable",
        }
    local_valid = local.dropna(subset=["cl", "cd", "cm"]).copy()
    local_valid["lift_to_drag"] = local_valid["cl"] / local_valid["cd"]
    result: dict[str, object] = {
        "local_converged_points": len(local_valid),
        "local_alpha_min": float(local_valid["alpha_deg"].min()) if len(local_valid) else None,
        "local_alpha_max": float(local_valid["alpha_deg"].max()) if len(local_valid) else None,
        "local_max_lift_to_drag": float(local_valid["lift_to_drag"].max()) if len(local_valid) else None,
    }
    if archive_path is None or not archive_path.is_file() or not len(local_valid):
        result["archive_comparison"] = "unavailable"
        return result
    archive = parse_polar_file(archive_path).points.dropna(subset=["cl", "cd", "cm"]).copy()
    merged = local_valid.merge(archive, on="alpha_deg", suffixes=("_local", "_archive"))
    result["archive_path"] = archive_path.as_posix()
    result["archive_sha256"] = sha256_file(archive_path)
    result["shared_alpha_points"] = len(merged)
    for target in ("cl", "cd", "cm"):
        if len(merged):
            error = merged[f"{target}_local"].to_numpy(float) - merged[f"{target}_archive"].to_numpy(float)
            result[f"{target}_rmse"] = float(np.sqrt(np.mean(error**2)))
    archive["lift_to_drag"] = archive["cl"] / archive["cd"]
    result["archive_max_lift_to_drag"] = float(archive["lift_to_drag"].max()) if len(archive) else None
    return result


def run_canary_suite(executable: Path, root: Path, output_root: Path, timeout_seconds: int = 240) -> dict[str, object]:
    cases = {
        "naca0012": root / "data/aerofoil_data/n0012.dat",
        "naca2412": root / "data/aerofoil_data/naca2412.dat",
        "clark_y": root / "data/aerofoil_data/clarky.dat",
        "eppler_e387": root / "data/aerofoil_data/e387.dat",
        "selig_ag04": root / "data/aerofoil_data/ag04.dat",
        "wortmann_fx63137": root / "data/aerofoil_data/fx63137.dat",
    }
    archive_root = root / "data/robust_v2/raw/airfoiltools_snapshot/repo/dat/case-dat"
    archive_candidates = list(archive_root.glob("*-il-1000000.csv"))
    rows: dict[str, object] = {}
    ranking: list[tuple[str, float]] = []
    for name, geometry in cases.items():
        case_root = output_root / name
        normalised = normalise_geometry(parse_coordinate_file(geometry), cosine_points=81)
        normalised_geometry = case_root / "normalised_geometry.dat"
        _write_geometry(normalised_geometry, name, normalised.upper, normalised.lower)
        case = XFoilCase(
            normalised_geometry,
            1_000_000,
            0.0,
            9.0,
            0.0,
            15.0,
            0.25,
            70,
            "positive",
        )
        result = run_xfoil(executable, case, case_root, timeout_seconds)
        archive = next((path for path in archive_candidates if path.stem.split("-il-")[0].replace("-", "") in {geometry.stem.replace("-", ""), name.replace("_", "")}), None)
        comparison = _comparison(result.parsed_points, archive)
        expected_points, convergence_fraction, span_complete = _sweep_completion(
            result.parsed_points, case
        )
        local_lift_to_drag = comparison.get("local_max_lift_to_drag")
        if isinstance(local_lift_to_drag, (int, float, np.floating)):
            ranking.append((name, float(local_lift_to_drag)))
        rows[name] = {
            "status": result.status,
            "return_code": result.return_code,
            "timed_out": result.timed_out,
            "failure_reason": result.failure_reason,
            "expected_alpha_points": expected_points,
            "convergence_fraction": convergence_fraction,
            "alpha_span_complete": span_complete,
            "source_geometry": geometry.as_posix(),
            "source_geometry_sha256": sha256_file(geometry),
            "normalised_geometry": normalised_geometry.as_posix(),
            "normalised_geometry_sha256": sha256_file(normalised_geometry),
            "command": result.command_text,
            "stdout": result.stdout_path.as_posix(),
            "stderr": result.stderr_path.as_posix(),
            "polar": result.polar_path.as_posix() if result.polar_path else None,
            "comparison": comparison,
        }
    ranking.sort(key=lambda item: item[1], reverse=True)
    completed = sum(isinstance(value, dict) and value.get("status") == "ok" for value in rows.values())
    payload = {
        "source_tier": "local_xfoil_6_99_crosscheck",
        "executable": executable.as_posix(),
        "executable_sha256": sha256_file(executable),
        "requested_cases": len(cases),
        "completed_cases": completed,
        "all_cases_completed": completed == len(cases),
        "local_lift_to_drag_ranking": [name for name, _ in ranking],
        "cases": rows,
    }
    write_json_atomic(output_root / "canary_summary.json", payload)
    return payload


def _write_geometry(path: Path, name: str, upper: np.ndarray, lower: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.vstack([upper[::-1], lower[1:]])
    lines = [f"_{name}", *(f"{x:.12f} {y:.12f}" for x, y in coordinates)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prediction_frame(parameters: np.ndarray, alpha: np.ndarray, airfoil_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        np.repeat(parameters[None, :], len(alpha), axis=0),
        columns=pd.Index(FEATURE_COLUMNS[:-1]),
    )
    frame["alpha_deg"] = alpha
    frame["airfoil_id"] = airfoil_id
    return frame


def _weighted_drag(points: pd.DataFrame, optimisation: OptimisationConfig) -> float | None:
    if not {"alpha_deg", "cl", "cd"}.issubset(points.columns):
        return None
    valid = points.dropna(subset=["alpha_deg", "cl", "cd"]).sort_values("alpha_deg")
    if not len(valid):
        return None
    value = weighted_required_lift_drag(
        valid["alpha_deg"].to_numpy(float),
        valid["cl"].to_numpy(float),
        valid["cd"].to_numpy(float),
        np.asarray(optimisation.service_targets.cl_required, dtype=float),
        np.asarray(optimisation.service_targets.weights, dtype=float),
    )
    return float(value) if np.isfinite(value) else None


def _surrogate_comparison(
    points: pd.DataFrame,
    parameters: np.ndarray,
    airfoil_id: str,
    ensemble: LoadedEnsemble,
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
) -> dict[str, object]:
    if not {"alpha_deg", "cl", "cd", "cm"}.issubset(points.columns):
        return {"shared_points": 0, "trusted_points": 0}
    valid = points.dropna(subset=["alpha_deg", "cl", "cd", "cm"]).copy()
    if not len(valid):
        return {"shared_points": 0, "trusted_points": 0}
    predictions = ensemble.predict(
        _prediction_frame(parameters, valid["alpha_deg"].to_numpy(float), airfoil_id)
    )
    trusted = apply_trust_model(
        development_points,
        predictions,
        ensemble.first_scaling,
        calibration,
    )
    result: dict[str, object] = {
        "shared_points": len(valid),
        "trusted_points": int(trusted["trusted_domain"].sum()),
        "trusted_fraction": float(trusted["trusted_domain"].mean()),
    }
    pairs = {
        "cl": (valid["cl"].to_numpy(float), trusted["prediction_cl"].to_numpy(float)),
        "cd": (
            valid["cd"].to_numpy(float),
            np.exp(trusted["prediction_log_cd"].to_numpy(float)),
        ),
        "cm": (valid["cm"].to_numpy(float), trusted["prediction_cm"].to_numpy(float)),
    }
    for target, (observed, predicted) in pairs.items():
        error = predicted - observed
        result[f"{target}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[f"{target}_bias"] = float(np.mean(error))
    return result


def _bootstrap_paired_interval(values: np.ndarray, seed: int) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(10_000, len(finite)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(finite)),
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
        "paired_samples": len(finite),
        "bootstrap_resamples": 10_000,
    }


def _shared_perturbations(
    upper: np.ndarray,
    lower: np.ndarray,
    latent: np.ndarray,
    uncertainty: UncertaintyConfig,
) -> list[PerturbedGeometry]:
    controls = uncertainty.basis_control_points_per_surface
    correlation = uncertainty.upper_lower_correlation
    independent_scale = float(np.sqrt(max(0.0, 1 - correlation**2)))
    amplitude = max(uncertainty.amplitudes_fraction_chord)
    return [
        smooth_normal_perturbation(
            upper,
            lower,
            row[:controls],
            correlation * row[:controls] + independent_scale * row[controls:],
            amplitude,
            uncertainty.correlation_length_chord,
            uncertainty.leading_edge_taper,
            uncertainty.trailing_edge_zero_displacement,
        )
        for row in latent
    ]


def _adverse_indices(
    perturbations: list[PerturbedGeometry],
    ensemble: LoadedEnsemble,
    optimisation: OptimisationConfig,
    uncertainty: UncertaintyConfig,
    design_id: str,
    count: int = 2,
) -> list[int]:
    scored: list[tuple[int, float]] = []
    alpha = np.asarray(optimisation.ava_baseline.alpha_deg, dtype=float)
    for index, perturbation in enumerate(perturbations):
        if not perturbation.valid:
            continue
        fit_upper, fit_lower = resample_surfaces_to_common_x(
            perturbation.upper,
            perturbation.lower,
            (1 - np.cos(np.linspace(0, np.pi, len(perturbation.upper)))) / 2,
        )
        fit = fit_cst(fit_upper, fit_lower)
        if fit.max_error > uncertainty.maximum_cst_refit_error_fraction_chord:
            continue
        prediction = ensemble.predict(_prediction_frame(fit.parameters, alpha, design_id))
        points = pd.DataFrame(
            {
                "alpha_deg": alpha,
                "cl": prediction["prediction_cl"].to_numpy(float),
                "cd": np.exp(prediction["prediction_log_cd"].to_numpy(float)),
            }
        )
        drag = _weighted_drag(points, optimisation)
        if drag is not None:
            scored.append((index, drag))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [index for index, _ in scored[:count]]


def run_candidate_validation(
    executable: Path,
    selected_designs: list[dict[str, Any]],
    ensemble_root: Path,
    development_points: pd.DataFrame,
    calibration: dict[str, Any],
    uncertainty: UncertaintyConfig,
    optimisation: OptimisationConfig,
    output_root: Path,
    timeout_seconds: int = 240,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    ensemble = LoadedEnsemble(ensemble_root)
    x = (1 - np.cos(np.linspace(0, np.pi, uncertainty.cosine_points_per_surface))) / 2
    dimension = uncertainty.basis_control_points_per_surface * 2
    shared_latent = sobol_normal_samples(64, dimension, uncertainty.seed + 20_000, antithetic=True)
    adverse_latent = sobol_normal_samples(64, dimension, uncertainty.seed + 21_000, antithetic=True)
    case_records: list[dict[str, Any]] = []
    design_summary: dict[str, dict[str, Any]] = {}
    for design in selected_designs:
        design_id = str(design["design_id"])
        parameters = np.asarray(design["parameters"], dtype=float)
        upper_y, lower_y = reconstruct_cst(parameters, x)
        upper = np.column_stack([x, upper_y])
        lower = np.column_stack([x, lower_y])
        shared = _shared_perturbations(upper, lower, shared_latent, uncertainty)
        adverse_pool = _shared_perturbations(upper, lower, adverse_latent, uncertainty)
        adverse_indices = _adverse_indices(
            adverse_pool, ensemble, optimisation, uncertainty, design_id
        )
        cases: list[tuple[str, int | None, PerturbedGeometry | None]] = [("nominal", None, None)]
        cases.extend(("shared", index, perturbation) for index, perturbation in enumerate(shared))
        cases.extend(("adverse_tail", index, adverse_pool[index]) for index in adverse_indices)
        per_design: list[dict[str, Any]] = []
        for kind, sample_id, perturbation in cases:
            case_upper = perturbation.upper if perturbation is not None else upper
            case_lower = perturbation.lower if perturbation is not None else lower
            validity = validate_geometry(case_upper, case_lower)
            case_name = kind if sample_id is None else f"{kind}_{sample_id:03d}"
            case_dir = output_root / design_id / case_name
            geometry_path = case_dir / "geometry.dat"
            _write_geometry(geometry_path, f"{design_id}_{case_name}", case_upper, case_lower)
            fit_upper, fit_lower = resample_surfaces_to_common_x(
                case_upper,
                case_lower,
                (1 - np.cos(np.linspace(0, np.pi, len(case_upper)))) / 2,
            )
            fit = fit_cst(fit_upper, fit_lower)
            if (
                not validity.valid
                or fit.max_error > uncertainty.maximum_cst_refit_error_fraction_chord
            ):
                record: dict[str, Any] = {
                    "design_id": design_id,
                    "kind": kind,
                    "sample_id": sample_id,
                    "status": "rejected_before_xfoil",
                    "invalid_reasons": list(validity.reasons),
                    "cst_max_error": fit.max_error,
                    "geometry": geometry_path.as_posix(),
                    "geometry_sha256": sha256_file(geometry_path),
                }
            else:
                xfoil_case = XFoilCase(
                    geometry_path,
                    1_000_000,
                    0.0,
                    9.0,
                    0.0,
                    12.0,
                    0.25,
                    100,
                    "positive",
                )
                result = run_xfoil(executable, xfoil_case, case_dir, timeout_seconds)
                xfoil_drag = _weighted_drag(result.parsed_points, optimisation)
                expected_points, convergence_fraction, span_complete = _sweep_completion(
                    result.parsed_points, xfoil_case
                )
                surrogate = _surrogate_comparison(
                    result.parsed_points,
                    fit.parameters,
                    f"{design_id}_{case_name}",
                    ensemble,
                    development_points,
                    calibration,
                )
                surrogate_prediction = ensemble.predict(
                    _prediction_frame(
                        fit.parameters,
                        np.asarray(optimisation.ava_baseline.alpha_deg, dtype=float),
                        f"{design_id}_{case_name}",
                    )
                )
                surrogate_points = pd.DataFrame(
                    {
                        "alpha_deg": optimisation.ava_baseline.alpha_deg,
                        "cl": surrogate_prediction["prediction_cl"].to_numpy(float),
                        "cd": np.exp(
                            surrogate_prediction["prediction_log_cd"].to_numpy(float)
                        ),
                    }
                )
                evidence_status = (
                    result.status
                    if result.status != "ok" or xfoil_drag is not None
                    else "failed_objective_coverage"
                )
                record = {
                    "design_id": design_id,
                    "kind": kind,
                    "sample_id": sample_id,
                    "status": evidence_status,
                    "return_code": result.return_code,
                    "timed_out": result.timed_out,
                    "failure_reason": result.failure_reason,
                    "expected_alpha_points": expected_points,
                    "convergence_fraction": convergence_fraction,
                    "alpha_span_complete": span_complete,
                    "geometry": geometry_path.as_posix(),
                    "geometry_sha256": sha256_file(geometry_path),
                    "command": result.command_text,
                    "stdout": result.stdout_path.as_posix(),
                    "stderr": result.stderr_path.as_posix(),
                    "polar": result.polar_path.as_posix() if result.polar_path else None,
                    "polar_sha256": sha256_file(result.polar_path) if result.polar_path else None,
                    "converged_points": len(result.parsed_points),
                    "xfoil_weighted_cd": xfoil_drag,
                    "required_lift_targets_complete": xfoil_drag is not None,
                    "surrogate_weighted_cd": _weighted_drag(surrogate_points, optimisation),
                    "surrogate_comparison": surrogate,
                    "cst_max_error": fit.max_error,
                }
            per_design.append(record)
            case_records.append(record)
        successful = [record for record in per_design if record["status"] == "ok"]
        design_summary[design_id] = {
            "requested_cases": len(per_design),
            "completed_cases": len(successful),
            "nominal_completed": any(
                record["kind"] == "nominal" and record["status"] == "ok" for record in per_design
            ),
            "shared_completed": sum(
                record["kind"] == "shared" and record["status"] == "ok" for record in per_design
            ),
            "adverse_tail_completed": sum(
                record["kind"] == "adverse_tail" and record["status"] == "ok" for record in per_design
            ),
            "adverse_pool_size": len(adverse_pool),
            "adverse_selected_indices": adverse_indices,
        }
    nominal = [
        record
        for record in case_records
        if record["kind"] == "nominal"
        and record["status"] == "ok"
        and record.get("xfoil_weighted_cd") is not None
    ]
    nominal_xfoil_ranking = [
        str(record["design_id"])
        for record in sorted(nominal, key=lambda record: float(record["xfoil_weighted_cd"]))
    ]
    nominal_surrogate_ranking = [
        str(record["design_id"])
        for record in sorted(nominal, key=lambda record: float(record["surrogate_weighted_cd"]))
    ]
    rank_correlation: float | None = None
    if len(nominal_xfoil_ranking) >= 2:
        xfoil_rank = {name: index for index, name in enumerate(nominal_xfoil_ranking)}
        surrogate_rank = {name: index for index, name in enumerate(nominal_surrogate_ranking)}
        rank_correlation = float(
            spearmanr(
                [xfoil_rank[name] for name in nominal_xfoil_ranking],
                [surrogate_rank[name] for name in nominal_xfoil_ranking],
            ).statistic
        )
    reference_records = {
        (str(record["kind"]), record["sample_id"]): record
        for record in case_records
        if record["design_id"] == "verified_reference"
        and record["status"] == "ok"
        and record.get("xfoil_weighted_cd") is not None
    }
    effects: dict[str, Any] = {}
    for design_id in design_summary:
        if design_id == "verified_reference":
            continue
        design_records = {
            (str(record["kind"]), record["sample_id"]): record
            for record in case_records
            if record["design_id"] == design_id
            and record["status"] == "ok"
            and record.get("xfoil_weighted_cd") is not None
        }
        nominal_key = ("nominal", None)
        nominal_improvement = None
        if nominal_key in reference_records and nominal_key in design_records:
            nominal_improvement = float(reference_records[nominal_key]["xfoil_weighted_cd"]) - float(
                design_records[nominal_key]["xfoil_weighted_cd"]
            )
        paired_keys = sorted(
            key
            for key in reference_records.keys() & design_records.keys()
            if key[0] == "shared"
        )
        paired = np.asarray(
            [
                float(reference_records[key]["xfoil_weighted_cd"])
                - float(design_records[key]["xfoil_weighted_cd"])
                for key in paired_keys
            ],
            dtype=float,
        )
        effects[design_id] = {
            "nominal_weighted_cd_improvement_vs_reference": nominal_improvement,
            "shared_perturbation_paired_improvement": _bootstrap_paired_interval(
                paired, uncertainty.seed + 30_000
            ),
            "positive_means_lower_drag_than_reference": True,
        }
    all_completed = bool(design_summary) and all(
        int(summary["requested_cases"]) > 0
        and int(summary["completed_cases"]) == int(summary["requested_cases"])
        for summary in design_summary.values()
    )
    payload = {
        "schema_version": 2,
        "source_tier": "local_xfoil_6_99_candidate_validation",
        "executable": executable.as_posix(),
        "executable_sha256": sha256_file(executable),
        "reynolds_number": 1_000_000,
        "mach": 0.0,
        "ncrit": 9.0,
        "shared_perturbation_matrix_sha256": sha256_bytes(
            np.ascontiguousarray(shared_latent, dtype="<f8").tobytes()
        ),
        "adverse_pool_matrix_sha256": sha256_bytes(
            np.ascontiguousarray(adverse_latent, dtype="<f8").tobytes()
        ),
        "design_summary": design_summary,
        "nominal_xfoil_ranking": nominal_xfoil_ranking,
        "nominal_surrogate_ranking": nominal_surrogate_ranking,
        "nominal_rank_spearman": rank_correlation,
        "effect_sizes": effects,
        "case_records": case_records,
        "all_cases_completed": all_completed,
    }
    write_json_atomic(output_root / "candidate_validation_summary.json", payload)
    return payload
