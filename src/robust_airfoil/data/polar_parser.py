from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

POLAR_COLUMNS = ["alpha_deg", "cl", "cd", "cdp", "cm", "xtr_top", "xtr_bottom"]
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[eEdD]\s*[+-]?\s*\d+)?"
ALIASES = {"alpha": "alpha_deg", "cl": "cl", "cd": "cd", "cdp": "cdp", "cm": "cm", "topxtr": "xtr_top", "botxtr": "xtr_bottom"}
META = {"reynoldsnumber": "reynolds_number", "re": "reynolds_number", "mach": "mach", "ncrit": "ncrit", "polarkey": "polar_key", "airfoil": "source_airfoil_name", "url": "original_url", "xfoilversion": "xfoil_version"}


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
    raise ValueError("Unsupported polar encoding")


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _float(value: str) -> float:
    value = re.sub(r"\s+", "", value).replace("D", "e").replace("d", "e")
    try:
        result = float(value)
    except ValueError:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def validate_polar_conditions(metadata: dict[str, str | float | None], reynolds_number: float = 1_000_000.0, mach: float = 0.0, ncrit: float = 9.0) -> None:
    for key, expected in (("reynolds_number", reynolds_number), ("mach", mach), ("ncrit", ncrit)):
        actual = metadata.get(key)
        if not isinstance(actual, (int, float)) or not np.isfinite(actual):
            raise ValueError(f"Missing explicit source condition: {key}")
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Source condition mismatch: {key}={actual}, expected {expected}")


def parse_polar_bytes(raw: bytes, source_name: str = "") -> ParsedPolar:
    text = _decode(raw).replace("\r\n", "\n")
    if re.search(r"<(?:!doctype|html|body)\b", text[:4096], re.I):
        raise ValueError(f"HTML is not aerodynamic data: {source_name}")
    lines = text.splitlines()
    header_index = None
    header: list[str] = []
    comma = False
    for i, line in enumerate(lines):
        comma_line = "," in line
        tokens = next(csv.reader([line])) if comma_line else line.split()
        names = [ALIASES.get(_key(v), "") for v in tokens]
        if {"alpha_deg", "cl", "cd"} <= set(names):
            header_index, header, comma = i, names, comma_line
            break
    if header_index is None:
        raise ValueError(f"No Alpha/CL/CD header in {source_name}")
    metadata: dict[str, str | float | None] = {"source_name": source_name, "reynolds_number": None, "mach": None, "ncrit": None}
    for line in lines[:header_index]:
        fields = next(csv.reader([line]))
        if len(fields) >= 2 and _key(fields[0]) in META:
            name = META[_key(fields[0])]
            value = fields[1].strip()
            parsed_value: str | float = _float(value) if name in {"reynolds_number", "mach", "ncrit"} else value
            existing = metadata.get(name)
            if existing is not None and existing != parsed_value:
                raise ValueError(f"Conflicting source metadata for {name}")
            metadata[name] = parsed_value
        for key, label in (("reynolds_number", r"\bRe"), ("mach", r"\bMach"), ("ncrit", r"\bNcrit")):
            found = re.search(label + r"\s*=\s*(" + NUMBER + r")", line, re.I)
            if found:
                value_float = _float(found.group(1))
                existing = metadata.get(key)
                if existing is not None and existing != value_float:
                    raise ValueError(f"Conflicting source metadata for {key}")
                metadata[key] = value_float
    rows: list[dict[str, float | int]] = []
    rejected: list[str] = []
    source_row = 0
    for line in lines[header_index + 1:]:
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-", " ", "_"}:
            continue
        tokens = next(csv.reader([line])) if comma else stripped.split()
        if len(tokens) != len(header):
            rejected.append(line)
            continue
        values = {name: _float(token) for name, token in zip(header, tokens, strict=True) if name}
        if not np.isfinite(values.get("alpha_deg", np.nan)):
            rejected.append(line)
            continue
        rows.append({**dict.fromkeys(POLAR_COLUMNS, float("nan")), **values, "source_row": source_row})
        source_row += 1
    if not rows:
        raise ValueError(f"No numeric polar rows in {source_name}")
    frame = pd.DataFrame(rows)
    frame["duplicate_alpha"] = frame.duplicated("alpha_deg", keep=False)
    conflict = frame.groupby("alpha_deg")[POLAR_COLUMNS[1:]].nunique(dropna=False).max(axis=1) > 1
    frame["conflicting_alpha"] = frame["alpha_deg"].map(conflict).fillna(False)
    frame["exact_duplicate"] = frame.duplicated(POLAR_COLUMNS, keep="first")
    return ParsedPolar(frame, metadata, tuple(rejected))


def parse_polar_file(path: Path) -> ParsedPolar:
    return parse_polar_bytes(path.read_bytes(), path.name)


def parse_polar_text(text: str, source_name: str = "memory") -> ParsedPolar:
    return parse_polar_bytes(text.encode(), source_name)
