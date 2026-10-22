from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.data.mapping import build_name_index, canonical_name, map_by_strict_name
from robust_airfoil.data.polar_parser import parse_polar_file, validate_polar_conditions
from robust_airfoil.geometry.clustering import cluster_coordinate_vectors
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.metrics import geometry_metrics
from robust_airfoil.geometry.normalise import normalise_geometry
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import sha256_bytes, sha256_file


def build_long_form_dataset(
    polar_paths: list[Path],
    geometry_paths: list[Path],
    output_dir: Path,
    source_tier: str,
    cosine_points: int = 201,
    near_duplicate_threshold: float = 1e-4,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = build_name_index(geometry_paths)
    airfoil_rows: list[dict[str, object]] = []
    polar_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    vector_by_airfoil: dict[str, np.ndarray] = {}
    cache: dict[Path, tuple[str, np.ndarray]] = {}
    for polar_path in sorted(polar_paths):
        mapping = map_by_strict_name(polar_path.name, index)
        mapping_rows.append({**asdict(mapping), "geometry_path": str(mapping.geometry_path) if mapping.geometry_path else None})
        if mapping.geometry_path is None:
            rejected.append({"source_path": str(polar_path), "reason": mapping.reason})
            continue
        geometry_path = mapping.geometry_path
        try:
            parsed = parse_polar_file(polar_path)
            validate_polar_conditions(parsed.metadata)
            if geometry_path not in cache:
                normal = normalise_geometry(parse_coordinate_file(geometry_path), cosine_points)
                validity = validate_geometry(normal.upper, normal.lower)
                if not validity.valid:
                    raise ValueError(";".join(validity.reasons))
                fit = fit_cst(normal.upper, normal.lower)
                refit_upper, refit_lower = reconstruct_cst(fit.parameters, normal.upper[:, 0])
                refit_validity = validate_geometry(
                    np.column_stack([normal.upper[:, 0], refit_upper]),
                    np.column_stack([normal.lower[:, 0], refit_lower]),
                )
                if not refit_validity.valid:
                    raise ValueError("invalid_refitted_geometry:" + ";".join(refit_validity.reasons))
                metrics = geometry_metrics(normal.upper, normal.lower)
                vector = np.concatenate([normal.upper[:, 1], normal.lower[:, 1]])
                airfoil_id = canonical_name(geometry_path.name)
                geometry_hash = sha256_bytes(np.round(normal.coordinates, 8).tobytes())
                cache[geometry_path] = (airfoil_id, fit.parameters)
                vector_by_airfoil[airfoil_id] = vector
                airfoil_rows.append({
                    "airfoil_id": airfoil_id,
                    "canonical_name": airfoil_id,
                    "geometry_source": "continuity_geometry",
                    "geometry_path": geometry_path.as_posix(),
                    "geometry_sha256": sha256_file(geometry_path),
                    "normalised_geometry_hash": geometry_hash,
                    "cluster_id": -1,
                    "cst_rms_error": fit.rms_error,
                    "cst_max_error": fit.max_error,
                    **metrics,
                    "valid_geometry": True,
                    **{f"lower_weight_{i}": float(fit.parameters[i]) for i in range(5)},
                    **{f"upper_weight_{i}": float(fit.parameters[i + 5]) for i in range(5)},
                    "leading_edge_weight": float(fit.parameters[10]),
                    "TE_thickness": float(fit.parameters[11]),
                })
            airfoil_id, _ = cache[geometry_path]
            polar_id = polar_path.stem
            for row in parsed.points.to_dict(orient="records"):
                polar_rows.append({
                    "airfoil_id": airfoil_id,
                    "polar_id": polar_id,
                    "source_tier": source_tier,
                    "source_path": polar_path.as_posix(),
                    "source_sha256": sha256_file(polar_path),
                    "reynolds_number": parsed.metadata["reynolds_number"],
                    "mach": parsed.metadata["mach"],
                    "ncrit": parsed.metadata["ncrit"],
                    **row,
                })
        except Exception as exc:
            rejected.append({"source_path": str(polar_path), "reason": repr(exc)})
    airfoils = pd.DataFrame(airfoil_rows).drop_duplicates("airfoil_id", keep="first") if airfoil_rows else pd.DataFrame(columns=["airfoil_id"])
    if len(airfoils):
        retained_vectors = np.asarray([vector_by_airfoil[str(name)] for name in airfoils["airfoil_id"]])
        airfoils["cluster_id"] = cluster_coordinate_vectors(retained_vectors, near_duplicate_threshold)
    points = pd.DataFrame(polar_rows)
    if len(points):
        points["mask_cl"] = np.isfinite(points["cl"])
        points["mask_log_cd"] = points["cd"].gt(0) & np.isfinite(points["cd"])
        points["mask_cm"] = np.isfinite(points["cm"])
        points["log_cd"] = np.nan
        points.loc[points["mask_log_cd"], "log_cd"] = np.log(points.loc[points["mask_log_cd"], "cd"])
        points["model_eligible"] = ~points["conflicting_alpha"] & ~points["exact_duplicate"]
        eligible_points = points.loc[points["model_eligible"]].copy()
        feature_columns = [f"lower_weight_{i}" for i in range(5)] + [f"upper_weight_{i}" for i in range(5)] + ["leading_edge_weight", "TE_thickness"]
        model_points = eligible_points.merge(airfoils[["airfoil_id", *feature_columns]], on="airfoil_id", how="inner", validate="many_to_one")
        counts = model_points.groupby("airfoil_id").size()
        model_points["airfoil_weight"] = model_points["airfoil_id"].map(1.0 / counts)
    else:
        model_points = pd.DataFrame()
    for name, frame in {"airfoils": airfoils, "polar_points": points, "model_points": model_points}.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    (output_dir / "mappings.json").write_text(json.dumps(mapping_rows, indent=2, default=str), encoding="utf-8")
    (output_dir / "rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    return {
        "requested_polars": len(polar_paths),
        "mapped_airfoils": int(len(airfoils)),
        "accepted_polar_rows": int(len(points)),
        "joint_model_points": int(model_points[["mask_cl", "mask_log_cd", "mask_cm"]].all(axis=1).sum()) if len(model_points) else 0,
        "rejected_records": len(rejected),
        "mapping_fraction": len(airfoils) / max(len(polar_paths), 1),
    }


def admit_local_polar_rows(points: pd.DataFrame, requested_alphas: np.ndarray, *, return_code: int | None, timed_out: bool) -> pd.DataFrame:
    """Keep actual usable observations, without claiming a complete solver sweep.

    Partial polars may contain genuine converged rows after a clean exit.
    Abnormal exits/timeouts contribute no training labels. Missing observations
    are never filled, extrapolated or replaced by a model prediction.
    """
    required = ['alpha_deg', 'cl', 'cd', 'cm']
    empty = points.iloc[0:0].copy().reindex(columns=list(dict.fromkeys([*points.columns, *required, 'log_cd', 'mask_cl', 'mask_log_cd', 'mask_cm'])))
    requested = np.asarray(requested_alphas, float)
    if requested.ndim != 1 or not np.isfinite(requested).all():
        raise ValueError('Requested solver incidence must be a finite vector')
    if return_code != 0 or timed_out or not set(required).issubset(points.columns) or not len(points):
        return empty
    values = points[required].to_numpy(float)
    finite = np.isfinite(values).all(axis=1) & (values[:, 2] > 0)
    on_grid = np.any(abs(values[:, 0, None] - requested[None, :]) <= 5.01e-4, axis=1)
    result = points.loc[finite & on_grid].copy()
    result['log_cd'] = np.log(result['cd'].to_numpy(float))
    for target in ('cl', 'log_cd', 'cm'):
        result['mask_' + target] = True
    return result.reset_index(drop=True)


def merge_local_polar_rows(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Count consistent repeated angles once; exclude conflicting observations."""
    populated = [frame for frame in frames if len(frame)]
    if not populated:
        return pd.DataFrame(columns=['alpha_deg', 'cl', 'cd', 'cm', 'log_cd', 'mask_cl', 'mask_log_cd', 'mask_cm']), {'admitted_states': 0, 'conflicting_alpha_count': 0, 'conflicting_alpha_values': [], 'consistent_duplicate_rows_removed': 0}
    combined = pd.concat(populated, ignore_index=True)
    kept: list[pd.DataFrame] = []
    conflicts: list[float] = []
    consistent_removed = 0
    for _alpha, group in combined.groupby('alpha_deg', sort=True):
        values = group[['cl', 'cd', 'cm']].to_numpy(float)
        if not np.isfinite(values).all() or np.any(np.ptp(values, axis=0) > 1e-10):
            conflicts.append(float(group['alpha_deg'].to_numpy(dtype=float)[0]))
            continue
        kept.append(group.iloc[:1].copy())
        consistent_removed += len(group) - 1
    result = pd.concat(kept, ignore_index=True) if kept else combined.iloc[:0].copy()
    return result, {'admitted_states': len(result), 'conflicting_alpha_count': len(conflicts), 'conflicting_alpha_values': conflicts, 'consistent_duplicate_rows_removed': consistent_removed}
