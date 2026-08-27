from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

POLAR_COLUMNS = ["alpha_deg", "cl", "cd", "cdp", "cm", "xtr_top", "xtr_bottom"]


@dataclass(frozen=True)
class ParsedPolar:
    points: pd.DataFrame
    metadata: dict[str, str | float | None]
    rejected_lines: tuple[str, ...]


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("unknown", raw, 0, 1, "no supported encoding")


def parse_polar_bytes(raw: bytes, source_name: str = "") -> ParsedPolar:
    text = _decode(raw).replace("\r\n", "\n")
    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines) if re.search(r"\bAlpha\b", line, re.I) and re.search(r"\bCL\b", line, re.I) and re.search(r"\bCD\b", line, re.I)), None)
    if header_index is None:
        raise ValueError(f"No Alpha/CL/CD header in {source_name}")
    metadata: dict[str, str | float | None] = {"source_name": source_name, "reynolds_number": None, "mach": None, "ncrit": None}
    preamble = "\n".join(lines[:header_index])
    for key, pattern in {
        "reynolds_number": r"Re\s*=\s*([0-9.eE+\-]+)",
        "mach": r"Mach\s*=\s*([0-9.eE+\-]+)",
        "ncrit": r"Ncrit\s*=\s*([0-9.eE+\-]+)",
    }.items():
        match = re.search(pattern, preamble, re.I)
        if match:
            metadata[key] = float(match.group(1))
    rows: list[list[float]] = []
    rejected: list[str] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-", " ", "_"}:
            continue
        tokens = re.split(r"[\s,]+", stripped)
        if len(tokens) < 7:
            rejected.append(line)
            continue
        try:
            values = [float(token) if token not in {"*", "NaN", "nan"} else np.nan for token in tokens[:7]]
        except ValueError:
            rejected.append(line)
            continue
        if not np.isfinite(values[0]):
            rejected.append(line)
            continue
        rows.append(values)
    if not rows:
        raise ValueError(f"No numeric polar rows in {source_name}")
    frame = pd.DataFrame(rows, columns=POLAR_COLUMNS)
    frame["source_row"] = np.arange(len(frame), dtype=int)
    frame["duplicate_alpha"] = frame.duplicated("alpha_deg", keep=False)
    conflict = frame.groupby("alpha_deg")[POLAR_COLUMNS[1:]].nunique(dropna=False).max(axis=1) > 1
    frame["conflicting_alpha"] = frame["alpha_deg"].map(conflict).fillna(False)
    return ParsedPolar(frame, metadata, tuple(rejected))


def parse_polar_file(path: Path) -> ParsedPolar:
    return parse_polar_bytes(path.read_bytes(), path.name)


def parse_polar_text(text: str, source_name: str = "memory") -> ParsedPolar:
    return parse_polar_bytes(io.BytesIO(text.encode()).read(), source_name)
