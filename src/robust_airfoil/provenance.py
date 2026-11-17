from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from robust_airfoil.hashing import sha256_file


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_output(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


@dataclass(frozen=True)
class FileEvidence:
    relative_path: str
    size_bytes: int
    sha256: str
    git_tracked: bool
    git_blob_sha: str
    modified_utc: str


def file_evidence(path: Path, root: Path) -> FileEvidence:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    blob = git_output("hash-object", str(path), cwd=root)
    tracked = bool(git_output("ls-files", "--error-unmatch", relative, cwd=root))
    return FileEvidence(
        relative_path=relative,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        git_tracked=tracked,
        git_blob_sha=blob if tracked else "",
        modified_utc=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    )


def environment_record(root: Path) -> dict[str, Any]:
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    except Exception as exc:  # pragma: no cover - evidence path
        torch_info = {"error": repr(exc), "device": "unavailable"}
    return {
        "captured_utc": utc_now(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "hostname": socket.gethostname(),
        "git_sha": git_output("rev-parse", "HEAD", cwd=root),
        "git_branch": git_output("branch", "--show-current", cwd=root),
        "torch": torch_info,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def json_default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return value.as_posix()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    temporary.replace(path)


def evidence_as_dict(evidence: FileEvidence) -> dict[str, Any]:
    return asdict(evidence)
