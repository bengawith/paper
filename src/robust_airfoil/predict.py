"""User-facing trust-calibrated surrogate prediction.

This is the practical tool the manuscript describes: given a 12-parameter CST
aerofoil and one or more angles of attack, it returns the surrogate lift, drag
and pitching-moment coefficients together with the ensemble uncertainty and,
critically, whether each prediction lies inside the calibrated trust domain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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
    return values


def alpha_grid(start: float, end: float, step: float) -> np.ndarray:
    """Inclusive angle-of-attack grid with a strictly positive step."""
    if step <= 0:
        raise ValueError("alpha step must be positive")
    if end < start:
        raise ValueError("alpha end must not precede alpha start")
    count = int(round((end - start) / step))
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


def predict_polar(
    lineage_id: str,
    cst_parameters: np.ndarray,
    alpha_deg: np.ndarray,
) -> dict[str, Any]:
    """Predict a trust-annotated polar for one CST aerofoil from a frozen lineage."""
    import json

    from robust_airfoil.constants import ROOT
    from robust_airfoil.modelling.calibrate import apply_trust_model
    from robust_airfoil.modelling.ensemble import LoadedEnsemble
    from robust_airfoil.study import study_paths

    parameters = np.asarray(cst_parameters, dtype=float)
    if parameters.shape != (len(PARAMETER_COLUMNS),):
        raise ValueError(f"CST vector must have {len(PARAMETER_COLUMNS)} entries")
    alpha = np.asarray(alpha_deg, dtype=float)
    if alpha.ndim != 1 or not len(alpha):
        raise ValueError("At least one angle of attack is required")

    paths = study_paths(lineage_id)
    ensemble_root = paths.results_root / "full_ensemble"
    if not ensemble_root.is_dir():
        raise FileNotFoundError("Prediction requires the frozen ensemble; run the model phase first")
    calibration = json.loads(
        (paths.results_root / "calibration/trust_calibration.json").read_text(encoding="utf-8")
    )
    split = json.loads((paths.results_root / "splits/full.json").read_text(encoding="utf-8"))
    development_ids = {str(r["airfoil_id"]) for r in split["records"] if r["split"] == "development"}
    points = pd.read_parquet(ROOT / "data/robust_v2/processed/full_exact/model_points.parquet")
    development_points: pd.DataFrame = points.loc[
        points["airfoil_id"].astype(str).isin(sorted(development_ids))
    ].copy()

    frame = pd.DataFrame(
        np.repeat(parameters[None, :], len(alpha), axis=0), columns=pd.Index(PARAMETER_COLUMNS)
    )
    frame["alpha_deg"] = alpha
    frame["airfoil_id"] = "query"
    ensemble = LoadedEnsemble(ensemble_root)
    predictions = ensemble.predict(frame)
    trusted = apply_trust_model(development_points, predictions, ensemble.first_scaling, calibration)
    polar = assemble_polar(trusted)
    return {
        "lineage_id": lineage_id,
        "reynolds_number": 1_000_000,
        "mach": 0.0,
        "ncrit": 9.0,
        "cst_parameters": parameters.tolist(),
        "within_trust_fraction": float(np.mean([r["within_trust_domain"] for r in polar])),
        "polar": polar,
    }
