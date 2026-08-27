from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_object(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def hash_paths(paths: Sequence[str | Path], root: str | Path | None = None) -> dict[str, str]:
    base = Path(root).resolve() if root else None
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(raw).resolve()
        key = str(path.relative_to(base)) if base and path.is_relative_to(base) else str(path)
        result[key.replace("\\", "/")] = sha256_file(path)
    return dict(sorted(result.items()))
