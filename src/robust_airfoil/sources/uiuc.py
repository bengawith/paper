from __future__ import annotations

import json
import zipfile
from pathlib import Path

import httpx

from robust_airfoil.hashing import sha256_bytes


def acquire_uiuc_zip(url: str, destination: Path, manifest_path: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, timeout=90, follow_redirects=True)
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise ValueError("UIUC response is not a ZIP archive")
    destination.write_bytes(response.content)
    with zipfile.ZipFile(destination) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    record = {
        "url": url,
        "http_status": response.status_code,
        "etag": response.headers.get("etag"),
        "last_modified": response.headers.get("last-modified"),
        "content_length": len(response.content),
        "sha256": sha256_bytes(response.content),
        "member_count": len(members),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def extract_zip_safely(zip_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            extracted.append(target)
    return extracted
