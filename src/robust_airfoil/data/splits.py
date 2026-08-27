from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.hashing import hash_object


def make_cluster_splits(airfoils: pd.DataFrame, seed: int = 20260824, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> pd.DataFrame:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    clusters = np.asarray(sorted(airfoils["cluster_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n = len(clusters)
    n_dev = max(1, int(round(fractions[0] * n)))
    n_cal = max(1, int(round(fractions[1] * n))) if n >= 3 else 0
    if n_dev + n_cal >= n:
        n_dev, n_cal = max(1, n - 2), 1 if n >= 2 else 0
    assignment = {int(cluster): "development" for cluster in clusters[:n_dev]}
    assignment.update({int(cluster): "calibration" for cluster in clusters[n_dev : n_dev + n_cal]})
    assignment.update({int(cluster): "locked_test" for cluster in clusters[n_dev + n_cal :]})
    result = airfoils[["airfoil_id", "cluster_id"]].copy()
    result["split"] = result["cluster_id"].map(assignment)
    if result.groupby("cluster_id")["split"].nunique().max() != 1:
        raise AssertionError("Geometry cluster leakage")
    return result


def write_split_manifest(splits: pd.DataFrame, path: Path, seed: int) -> dict[str, object]:
    records = splits.sort_values("airfoil_id").to_dict(orient="records")
    payload = {"seed": seed, "records": records, "records_hash": hash_object(records)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def make_grouped_folds(airfoils: pd.DataFrame, folds: int = 3, seed: int = 20260824) -> pd.DataFrame:
    clusters = np.asarray(sorted(airfoils["cluster_id"].unique()))
    if folds < 2 or len(clusters) < folds:
        raise ValueError("Grouped folds require at least one geometry cluster per fold")
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    assignment = {int(cluster): int(index % folds) for index, cluster in enumerate(clusters)}
    result = airfoils[["airfoil_id", "cluster_id"]].copy()
    result["fold"] = result["cluster_id"].map(assignment).astype(int)
    if result.groupby("cluster_id")["fold"].nunique().max() != 1:
        raise AssertionError("Geometry cluster leaked between tuning folds")
    if set(result["fold"].unique()) != set(range(folds)):
        raise AssertionError("Every grouped fold must contain at least one cluster")
    return result


def write_fold_manifest(fold_assignments: pd.DataFrame, path: Path, seed: int, folds: int) -> dict[str, object]:
    records = fold_assignments.sort_values("airfoil_id").to_dict(orient="records")
    payload = {"seed": seed, "folds": folds, "records": records, "records_hash": hash_object(records)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def assert_locked_test_access(frozen_manifest: Path, requested_split: str, artifacts: dict[str, Path] | None = None) -> None:
    if requested_split != "locked_test":
        return
    if not frozen_manifest.is_file():
        raise PermissionError("Locked test access denied until FROZEN_MODEL_MANIFEST.json exists")
    payload = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("status") != "frozen":
        raise PermissionError("Locked test access denied: frozen manifest is incomplete")
    if artifacts:
        from robust_airfoil.hashing import sha256_file

        recorded = payload.get("artifact_hashes", {})
        for name, path in artifacts.items():
            if not path.is_file() or recorded.get(name) != sha256_file(path):
                raise PermissionError(f"Locked test access denied: frozen artifact mismatch for {name}")
