"""User-facing trust-calibrated surrogate prediction.

This is the practical tool the manuscript describes: given a 12-parameter CST
aerofoil and one or more angles of attack, it returns the surrogate lift, drag
and pitching-moment coefficients together with the ensemble uncertainty and,
critically, whether each prediction lies inside the calibrated trust domain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS

PARAMETER_COLUMNS = FEATURE_COLUMNS[:-1]


def parse_cst(text: str) -> np.ndarray:
    """Parse 12 comma-separated CST parameters into an ordered vector.

    Order: lower_weight_0..4, upper_weight_0..4, leading_edge_weight, TE_thickness.
    """
    parts = [segment.strip() for segment in text.split(",") if segment.strip()]
    if len(parts) != len(PARAMETER_COLUMNS):
        raise ValueError(
            f"Expected {len(PARAMETER_COLUMNS)} CST parameters "
            f"(lower_0..4, upper_0..4, leading_edge_weight, TE_thickness); got {len(parts)}"
        )
    try:
        values = np.asarray([float(part) for part in parts], dtype=float)
    except ValueError as exc:
        raise ValueError("All CST parameters must be finite numbers") from exc
    if not np.isfinite(values).all():
        raise ValueError("All CST parameters must be finite numbers")
    if values[-1] < 0:
        raise ValueError("CST trailing-edge thickness must be nonnegative")
    return values


def cst_from_airfoil_file(
    path: Path, max_fit_error: float = 0.005
) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse an ordered aerofoil DAT and predict its validated CST *fit*.

    Report that the geometry was chord-normalised/rotated. The model only
    predicts the resulting reconstruction, not the unmodified source polar.
    """
    from robust_airfoil.geometry.cst import fit_cst
    from robust_airfoil.geometry.normalise import normalise_geometry
    from robust_airfoil.geometry.parser import parse_coordinate_file
    from robust_airfoil.geometry.validity import validate_geometry

    if not path.is_file():
        raise FileNotFoundError(f"Airfoil coordinate file not found: {path}")
    normalised = normalise_geometry(parse_coordinate_file(path))
    source_validity = validate_geometry(normalised.upper, normalised.lower)
    if not source_validity.valid:
        raise ValueError(f"Airfoil coordinate surfaces are invalid: {source_validity.reasons}")
    fitted = fit_cst(normalised.upper, normalised.lower)
    if fitted.max_error > max_fit_error:
        raise ValueError(
            f"CST reconstruction differs from the source by {fitted.max_error:.6f} chord, "
            f"exceeding {max_fit_error:.6f} chord"
        )
    fit_validity = validate_geometry(fitted.reconstructed_upper, fitted.reconstructed_lower)
    if not fit_validity.valid:
        raise ValueError(f"Reconstructed CST geometry is invalid: {fit_validity.reasons}")
    return fitted.parameters, {
        "input_type": "normalised_source_file_CST_fit",
        "source_sha256": sha256_file(path),
        "source_chord_units": normalised.chord,
        "source_chord_rotation_radians": normalised.rotation_radians,
        "cst_max_absolute_fit_error_chord": fitted.max_error,
        "cst_rms_fit_error_chord": fitted.rms_error,
        "prediction_geometry": "The normalized fitted CST reconstruction, not original file coordinates",
    }


def alpha_grid(start: float, end: float, step: float) -> np.ndarray:
    """Inclusive angle-of-attack grid with a strictly positive step."""
    if not np.isfinite([start, end, step]).all() or step <= 0:
        raise ValueError("alpha start, end and positive step must all be finite")
    if end < start:
        raise ValueError("alpha end must not precede alpha start")
    count = int(round((end - start) / step))
    if count > 4095:
        raise ValueError("At most 4096 angle-of-attack states may be requested")
    return start + np.arange(count + 1) * step


def assemble_polar(trusted: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a compact per-angle prediction record from a trust-annotated frame."""
    records: list[dict[str, Any]] = []
    for _, row in trusted.sort_values("alpha_deg").iterrows():
        records.append(
            {
                "alpha_deg": float(row["alpha_deg"]),
                "cl": float(row["prediction_cl"]),
                "cd": float(np.exp(row["prediction_log_cd"])),
                "cm": float(row["prediction_cm"]),
                "cl_ensemble_std": float(row["ensemble_std_cl"]),
                "log_cd_ensemble_std": float(row["ensemble_std_log_cd"]),
                "cm_ensemble_std": float(row["ensemble_std_cm"]),
                "support_distance": float(row["support_distance"]),
                "within_trust_domain": bool(row["trusted_domain"]),
            }
        )
    return records


def load_lineage_development(
    root: Path, lineage_id: str, results_root: Path
) -> tuple[pd.DataFrame, Path]:
    """Read the exact training-support source used to calibrate this lineage.

    The current/legacy model may use the historical full_exact archive. The
    rebuilt and controlled lineages MUST use their own frozen input partitions.
    This is deliberately independent of mutable run-state and never retrains.
    """
    partition = results_root / "partition_inputs/development.parquet"
    if partition.is_file():
        development = pd.read_parquet(partition)
        source = partition
    elif lineage_id.startswith(("robust-v2-controlled-", "robust-v2-rebuilt-")):
        raise FileNotFoundError("This reviewed lineage requires its own frozen development labels")
    else:
        import json

        split = json.loads((results_root / "splits/full.json").read_text(encoding="utf-8"))
        development_ids = {
            str(row["airfoil_id"]) for row in split["records"] if row["split"] == "development"
        }
        source = root / "data/robust_v2/processed/full_exact/model_points.parquet"
        data = pd.read_parquet(source)
        development = data.loc[data["airfoil_id"].astype(str).isin(development_ids)].copy()
    if development.empty:
        raise ValueError("No development states available for this model's trust calibration")
    if not set(FEATURE_COLUMNS).issubset(development.columns):
        raise ValueError("Development data lack required model input features")
    return development, source


def predict_polar(
    lineage_id: str,
    cst_parameters: np.ndarray,
    alpha_deg: np.ndarray,
) -> dict[str, Any]:
    """Predict a trust-annotated polar for one CST aerofoil from a frozen lineage."""
    import json

    from robust_airfoil.constants import ROOT
    from robust_airfoil.geometry.cst import reconstruct_cst
    from robust_airfoil.geometry.validity import validate_geometry
    from robust_airfoil.modelling.calibrate import apply_trust_model
    from robust_airfoil.modelling.ensemble import LoadedEnsemble
    from robust_airfoil.modelling.freeze import verify_frozen_artifacts
    from robust_airfoil.study import study_paths

    parameters = np.asarray(cst_parameters, dtype=float)
    if parameters.shape != (len(PARAMETER_COLUMNS),) or not np.isfinite(parameters).all():
        raise ValueError(f"CST vector must contain {len(PARAMETER_COLUMNS)} finite entries")
    if parameters[-1] < 0:
        raise ValueError("CST trailing-edge thickness must be nonnegative")
    geometry_x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    upper_y, lower_y = reconstruct_cst(parameters, geometry_x)
    geometry = validate_geometry(
        np.column_stack((geometry_x, upper_y)), np.column_stack((geometry_x, lower_y))
    )
    if not geometry.valid:
        raise ValueError(f"CST geometry is physically inadmissible: {geometry.reasons}")
    alpha = np.asarray(alpha_deg, dtype=float)
    if alpha.ndim != 1 or not 1 <= len(alpha) <= 4096 or not np.isfinite(alpha).all():
        raise ValueError("One to 4096 finite incidence values are required")

    paths = study_paths(lineage_id)
    ensemble_root = paths.results_root / "full_ensemble"
    if not ensemble_root.is_dir():
        raise FileNotFoundError(
            "Prediction requires the frozen ensemble; run the model phase first"
        )
    frozen_manifest = paths.results_root / "FROZEN_MODEL_MANIFEST.json"
    if frozen_manifest.is_file():
        artifacts = json.loads(frozen_manifest.read_text(encoding="utf-8"))
        verify_frozen_artifacts(
            frozen_manifest,
            {key: Path(path) for key, path in artifacts["artifact_paths"].items()},
        )
    elif lineage_id.startswith(("robust-v2-controlled-", "robust-v2-rebuilt-")):
        raise FileNotFoundError("Reviewed predictions require a verified frozen model manifest")
    calibration = json.loads(
        (paths.results_root / "calibration/trust_calibration.json").read_text(encoding="utf-8")
    )
    development_points, support_data_source = load_lineage_development(
        ROOT, lineage_id, paths.results_root
    )

    frame = pd.DataFrame(
        np.repeat(parameters[None, :], len(alpha), axis=0), columns=pd.Index(PARAMETER_COLUMNS)
    )
    frame["alpha_deg"] = alpha
    frame["airfoil_id"] = "query"
    ensemble = LoadedEnsemble(ensemble_root)
    predictions = ensemble.predict(frame)
    trusted = apply_trust_model(
        development_points, predictions, ensemble.first_scaling, calibration
    )
    polar = assemble_polar(trusted)

    return {
        "lineage_id": lineage_id,
        "support_data_sha256": sha256_file(support_data_source),
        "support_development_states": int(len(development_points)),
        "support_source": support_data_source.relative_to(ROOT).as_posix(),
        "reynolds_number": 1_000_000,
        "mach": 0.0,
        "ncrit": 9.0,
        "cst_parameters": parameters.tolist(),
        "within_trust_fraction": float(np.mean([r["within_trust_domain"] for r in polar])),
        "polar": polar,
    }
