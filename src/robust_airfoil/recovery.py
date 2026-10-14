from __future__ import annotations

import csv
import os
import pickletools
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from robust_airfoil.hashing import sha256_file

RELEVANT_SUFFIXES = {".csv", ".dat", ".txt", ".zip", ".pkl", ".pickle", ".pt", ".pth", ".parquet"}
SIGNATURES = ("airfoil", "aerofoil", "polar", "xfoil", "cst", "dataset_12")


@dataclass(frozen=True)
class RecoveryCandidate:
    path_redacted: str
    suffix: str
    size_bytes: int
    sha256: str
    pickle_globals: tuple[str, ...] = ()


def safe_pickle_globals(path: Path) -> tuple[str, ...]:
    globals_seen: list[str] = []
    with path.open("rb") as handle:
        for opcode, argument, _ in pickletools.genops(handle):
            if opcode.name in {"GLOBAL", "STACK_GLOBAL"}:
                globals_seen.append(str(argument))
    return tuple(globals_seen)


def discover_candidates(roots: Iterable[Path], maximum_files: int = 100_000) -> list[RecoveryCandidate]:
    found: list[RecoveryCandidate] = []
    visited = 0
    home = Path.home().resolve()
    for root in roots:
        if not root.exists():
            continue
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [name for name in subdirs if name not in {".git", ".venv", "node_modules", "AppData"}]
            for name in files:
                visited += 1
                if visited > maximum_files:
                    return found
                path = Path(directory) / name
                lower = name.lower()
                if path.suffix.lower() not in RELEVANT_SUFFIXES or not any(token in lower for token in SIGNATURES):
                    continue
                try:
                    resolved = path.resolve()
                    redacted = "~" + os.sep + str(resolved.relative_to(home)) if resolved.is_relative_to(home) else f"<{resolved.drive}>/{resolved.name}"
                    pickle_globals = safe_pickle_globals(path) if path.suffix.lower() in {".pkl", ".pickle"} else ()
                    found.append(RecoveryCandidate(redacted, path.suffix.lower(), path.stat().st_size, sha256_file(path), pickle_globals))
                except (OSError, ValueError):
                    continue
    return found


def write_recovery_csv(path: Path, candidates: list[RecoveryCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path_redacted", "suffix", "size_bytes", "sha256", "pickle_globals"])
        writer.writeheader()
        for candidate in candidates:
            row = asdict(candidate)
            row["pickle_globals"] = ";".join(candidate.pickle_globals)
            writer.writerow(row)
