from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robust_airfoil.hashing import sha256_file
from robust_airfoil.provenance import utc_now, write_json_atomic


def freeze_artifacts(
    manifest_path: Path,
    artifacts: dict[str, Path],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing artifacts: {missing}")
    artifact_hashes = {name: sha256_file(path) for name, path in sorted(artifacts.items())}
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "frozen",
        "created_utc": utc_now(),
        "artifact_paths": {name: path.as_posix() for name, path in sorted(artifacts.items())},
        "artifact_hashes": artifact_hashes,
        **metadata,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "created_utc"}
        comparable_new = {key: value for key, value in payload.items() if key != "created_utc"}
        if comparable_existing != comparable_new:
            raise RuntimeError("Refusing to overwrite a non-matching frozen model manifest")
        return existing
    write_json_atomic(manifest_path, payload)
    return payload


def verify_frozen_artifacts(manifest_path: Path, artifacts: dict[str, Path]) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("status") != "frozen":
        raise ValueError("Frozen manifest schema/status is invalid")
    expected = payload.get("artifact_hashes", {})
    for name, path in artifacts.items():
        if not path.is_file() or expected.get(name) != sha256_file(path):
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    return payload
