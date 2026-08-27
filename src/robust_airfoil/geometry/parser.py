from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_coordinate_text(text: str) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for line in text.replace("\r\n", "\n").splitlines():
        match = re.match(r"^\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)\s+([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)", line)
        if not match:
            continue
        pair = (float(match.group(1)), float(match.group(2)))
        if max(abs(pair[0]), abs(pair[1])) <= 5.0:
            rows.append(pair)
    if len(rows) < 5:
        raise ValueError("Coordinate file has fewer than five numeric pairs")
    points = np.asarray(rows, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    return points[keep]


def parse_coordinate_file(path: Path) -> np.ndarray:
    return parse_coordinate_text(path.read_text(encoding="utf-8", errors="replace"))
