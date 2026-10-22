from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS = [
    "Executive decision",
    "Branch",
    "Current SHA",
    "Working-tree status",
    "Environment",
    "PyTorch device",
    "XFOIL status",
    "V1 audit",
    "Source reachability",
    "Local recovery",
    "UIUC snapshot",
    "Pinned polar snapshot",
    "Live AirfoilTools",
    "UIUC continuity/current differences",
    "Mappings",
    "Accepted airfoils",
    "Accepted joint points",
    "Rejected records",
    "Geometry and CST results",
    "Leakage tests",
    "Pilot model",
    "200-airfoil model",
    "Full baseline",
    "Baseline comparisons",
    "Tuning status",
    "Calibration status",
    "NeuralFoil benchmark",
    "XFOIL comparison",
    "Uncertainty validity",
    "Optimisation study",
    "Five-design comparison",
    "Blockers and severity",
    "Scientific limitations",
    "Resolution plan",
    "Exact resume command",
    "Primary report path",
    "Handover path",
    "Files created or modified",
    "Local commits made",
    "Tests passed failed skipped",
    "Remaining failures",
    "Output hash manifest",
]


def build_viability_report(payload: dict[str, object], markdown_path: Path, json_path: Path) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Missing viability report fields: {missing}")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Robust Aerofoil V2 Viability Report",
        "",
        "This report records generated evidence. Full numerical arrays and command logs remain in the hashed artifacts listed below.",
        "",
    ]
    for field in REQUIRED_FIELDS:
        value = payload[field]
        lines.extend(
            [
                f"## {field}",
                "",
                "```json",
                json.dumps(value, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
