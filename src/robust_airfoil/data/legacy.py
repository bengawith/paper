from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.hashing import sha256_file


def audit_legacy_dataset(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    columns = [str(column) for column in frame.columns]
    cst = [column for column in columns if column.startswith(("lower_weight_", "upper_weight_")) or column in {"leading_edge_weight", "TE_thickness"}]
    cl = [column for column in columns if column.startswith("CL_")]
    distance = [column for column in columns if column.startswith("dist_")]
    alpha_grid = np.linspace(-20.0, 20.0, len(cl)).tolist() if cl else []
    report: dict[str, object] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "cst_columns": cst,
        "cl_columns": cl,
        "distance_columns": distance,
        "alpha_grid": alpha_grid,
        "duplicate_names": int(frame["aerofoil_name"].duplicated().sum()) if "aerofoil_name" in frame else None,
        "missing_values": int(frame.isna().sum().sum()),
        "family_counts": frame["family"].value_counts().to_dict() if "family" in frame else {},
        "limitations": [
            "lift-only wide targets",
            "out-of-range targets are padded or extrapolated",
            "no raw polar-row provenance",
            "no drag, pressure drag, moment, or transition targets",
        ],
    }
    return report


def write_legacy_audit(dataset_path: Path, output_path: Path) -> dict[str, object]:
    report = audit_legacy_dataset(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
