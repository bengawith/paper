from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.data.mapping import build_name_index, canonical_name, map_by_strict_name
from robust_airfoil.data.polar_parser import parse_polar_file
from robust_airfoil.geometry.clustering import cluster_coordinate_vectors
from robust_airfoil.geometry.cst import fit_cst
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
            if geometry_path not in cache:
                normal = normalise_geometry(parse_coordinate_file(geometry_path), cosine_points)
                validity = validate_geometry(normal.upper, normal.lower)
                if not validity.valid:
                    raise ValueError(";".join(validity.reasons))
                fit = fit_cst(normal.upper, normal.lower)
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
            parsed = parse_polar_file(polar_path)
            polar_id = polar_path.stem
            for row in parsed.points.to_dict(orient="records"):
                polar_rows.append({
                    "airfoil_id": airfoil_id,
                    "polar_id": polar_id,
                    "source_tier": source_tier,
                    "source_path": polar_path.as_posix(),
                    "source_sha256": sha256_file(polar_path),
                    "reynolds_number": parsed.metadata.get("reynolds_number") or 1_000_000.0,
                    "mach": parsed.metadata.get("mach") or 0.0,
                    "ncrit": parsed.metadata.get("ncrit") or 9.0,
                    **row,
                })
        except Exception as exc:
            rejected.append({"source_path": str(polar_path), "reason": repr(exc)})
    airfoils = pd.DataFrame(airfoil_rows).drop_duplicates("airfoil_id", keep="first")
    if len(airfoils):
        retained_vectors = np.asarray([vector_by_airfoil[str(name)] for name in airfoils["airfoil_id"]])
        airfoils["cluster_id"] = cluster_coordinate_vectors(retained_vectors, near_duplicate_threshold)
    points = pd.DataFrame(polar_rows)
    if len(points):
        points["mask_cl"] = points["cl"].notna()
        points["mask_log_cd"] = points["cd"].gt(0) & points["cd"].notna()
        points["mask_cm"] = points["cm"].notna()
        points["log_cd"] = np.where(points["mask_log_cd"], np.log(points["cd"]), np.nan)
        feature_columns = [f"lower_weight_{i}" for i in range(5)] + [f"upper_weight_{i}" for i in range(5)] + ["leading_edge_weight", "TE_thickness"]
        model_points = points.merge(airfoils[["airfoil_id", *feature_columns]], on="airfoil_id", how="inner", validate="many_to_one")
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
        "joint_model_points": int(len(model_points)),
        "rejected_records": len(rejected),
        "mapping_fraction": len(airfoils) / max(len(polar_paths), 1),
    }
