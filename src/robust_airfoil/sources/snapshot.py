from __future__ import annotations

import json
import subprocess
from pathlib import Path

from robust_airfoil.hashing import sha256_file


def acquire_pinned_snapshot(repo_url: str, commit: str, destination: Path, manifest: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        subprocess.run(["git", "clone", "--filter=blob:none", repo_url, str(destination)], check=True)
    subprocess.run(["git", "fetch", "origin", commit], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--detach", commit], cwd=destination, check=True)
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=destination, check=True, capture_output=True, text=True).stdout.strip()
    if actual != commit:
        raise ValueError(f"Snapshot checkout mismatch: {actual}")
    cases = sorted(destination.glob("dat/case-dat/*-il-1000000.csv"))
    geometries = sorted((destination / "dat" / "aerofoil-dat").glob("*"))
    license_candidates = [p for p in destination.glob("LICENSE*") if p.is_file()]
    payload = {
        "repo_url": repo_url,
        "requested_commit": commit,
        "actual_commit": actual,
        "case_count": len(cases),
        "geometry_count": len([p for p in geometries if p.is_file()]),
        "license_files": [{"path": p.name, "sha256": sha256_file(p)} for p in license_candidates],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
