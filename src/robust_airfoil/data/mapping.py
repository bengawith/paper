from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from robust_airfoil.hashing import sha256_file


def canonical_name(value: str) -> str:
    name = Path(value).stem.lower()
    name = re.sub(r"-il-1000000$", "", name)
    name = re.sub(r"-il$", "", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


@dataclass(frozen=True)
class MappingResult:
    source_name: str
    geometry_path: Path | None
    method: str
    confidence: str
    reason: str


def build_name_index(paths: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in paths:
        index.setdefault(canonical_name(path.name), []).append(path)
    return index


def map_by_strict_name(source_name: str, index: dict[str, list[Path]], aliases: dict[str, str] | None = None) -> MappingResult:
    key = canonical_name(source_name)
    if aliases and key in aliases:
        key = canonical_name(aliases[key])
    candidates = index.get(key, [])
    if len(candidates) == 1:
        return MappingResult(source_name, candidates[0], "exact_canonical_name", "high", "unique canonical name")
    if len(candidates) > 1:
        hashes = {sha256_file(path) for path in candidates}
        if len(hashes) == 1:
            return MappingResult(source_name, sorted(candidates)[0], "exact_duplicate_content", "high", "multiple names, identical bytes")
        return MappingResult(source_name, None, "quarantine", "none", "ambiguous canonical name")
    return MappingResult(source_name, None, "unmapped", "none", "no exact canonical match")
