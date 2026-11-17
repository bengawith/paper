from __future__ import annotations

import pandas as pd


def polar_quality_summary(points: pd.DataFrame) -> dict[str, float | int]:
    rows = len(points)
    duplicate = points["duplicate_alpha"] if "duplicate_alpha" in points else pd.Series(False, index=points.index)
    conflicting = points["conflicting_alpha"] if "conflicting_alpha" in points else pd.Series(False, index=points.index)
    return {
        "rows": rows,
        "finite_cl_fraction": float(points["cl"].notna().mean()) if rows else 0.0,
        "positive_cd_fraction": float((points["cd"] > 0).mean()) if rows else 0.0,
        "finite_cm_fraction": float(points["cm"].notna().mean()) if rows else 0.0,
        "duplicate_alpha_rows": int(duplicate.sum()),
        "conflicting_alpha_rows": int(conflicting.sum()),
    }
