from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.dataset import TARGET_COLUMNS
from robust_airfoil.modelling.models import ConditionedPolarMLP
from robust_airfoil.modelling.train import predict_frame, train_model


def load_member(output_dir: Path, device: torch.device | None = None) -> tuple[ConditionedPolarMLP, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model = ConditionedPolarMLP(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, joblib.load(output_dir / "scaling.joblib")


def train_ensemble(
    development_points: pd.DataFrame,
    output_root: Path,
    hyperparameters: dict[str, Any],
    epochs: int,
    base_seed: int,
    members: int = 5,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    for index in range(members):
        member_dir = output_root / f"member_{index:03d}"
        rng = np.random.default_rng(base_seed + index)
        if "cluster_id" not in development_points:
            raise ValueError("Cluster-resampled ensembles require cluster_id on every development point")
        clusters = np.asarray(sorted(development_points["cluster_id"].unique()))
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        bootstrap_pieces: list[pd.DataFrame] = []
        for draw, cluster in enumerate(sampled_clusters):
            piece = development_points.loc[development_points["cluster_id"] == cluster].copy()
            piece["airfoil_id"] = piece["airfoil_id"].astype(str) + f"__cluster_bootstrap_{draw:05d}"
            bootstrap_pieces.append(piece)
        bootstrap = pd.concat(bootstrap_pieces, ignore_index=True)
        summary = train_model(
            bootstrap,
            bootstrap,
            member_dir,
            seed=base_seed + index,
            maximum_epochs=max(1, epochs),
            patience=max(2, epochs + 1),
            hidden_width=int(hyperparameters["hidden_width"]),
            hidden_layers=int(hyperparameters["hidden_layers"]),
            activation=str(hyperparameters.get("activation", "silu")),
            dropout=float(hyperparameters["dropout"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            weight_decay=float(hyperparameters["weight_decay"]),
            batch_size=int(hyperparameters["batch_size"]),
            loss_name=str(hyperparameters["loss"]),
            residual=bool(hyperparameters.get("residual", True)),
            layer_norm=True,
        )
        summaries.append(summary)
        summaries[-1]["cluster_bootstrap"] = {
            "draws": len(sampled_clusters),
            "unique_clusters": int(len(np.unique(sampled_clusters))),
            "sampled_cluster_ids": sampled_clusters.astype(int).tolist(),
        }
        artifact_hashes[f"member_{index:03d}_checkpoint"] = sha256_file(member_dir / "checkpoint.pt")
        artifact_hashes[f"member_{index:03d}_scaling"] = sha256_file(member_dir / "scaling.joblib")
    manifest = {
        "member_count": members,
        "epochs": epochs,
        "base_seed": base_seed,
        "hyperparameters": hyperparameters,
        "artifact_hashes": artifact_hashes,
        "summaries": summaries,
    }
    (output_root / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return manifest


def predict_ensemble(member_root: Path, frame: pd.DataFrame, members: int = 5) -> pd.DataFrame:
    return LoadedEnsemble(member_root, members).predict(frame)


class LoadedEnsemble:
    def __init__(self, member_root: Path, members: int = 5, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.member_root = member_root
        manifest_path = member_root / "ensemble_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not 1 <= members <= int(manifest["member_count"]):
            raise ValueError("Requested member count is inconsistent with the ensemble manifest")
        self.artifact_hashes: dict[str, str] = {}
        for index in range(members):
            for label, filename in (("checkpoint", "checkpoint.pt"), ("scaling", "scaling.joblib")):
                key = f"member_{index:03d}_{label}"
                path = member_root / f"member_{index:03d}" / filename
                actual = sha256_file(path)
                if actual != manifest["artifact_hashes"].get(key):
                    raise ValueError(f"Ensemble artifact hash mismatch: {key}")
                self.artifact_hashes[key] = actual
        self.artifact_hashes["ensemble_manifest"] = sha256_file(manifest_path)
        self.members = [load_member(member_root / f"member_{index:03d}", self.device) for index in range(members)]

    @property
    def first_scaling(self) -> Any:
        return self.members[0][1]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for index, (model, scaling) in enumerate(self.members):
            prediction = predict_frame(model, frame, scaling)
            for target in TARGET_COLUMNS:
                result[f"prediction_{target}_member_{index:03d}"] = prediction[f"prediction_{target}"].to_numpy()
        for target in TARGET_COLUMNS:
            columns = [f"prediction_{target}_member_{index:03d}" for index in range(len(self.members))]
            values = result[columns].to_numpy(float)
            result[f"prediction_{target}"] = values.mean(axis=1)
            result[f"ensemble_std_{target}"] = values.std(axis=1, ddof=1)
        return result


def cluster_bootstrap_with_oob(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Bootstrap DEVELOPMENT clusters, holding out unsampled clusters for stopping."""
    if 'cluster_id' not in frame or frame.cluster_id.isna().any():
        raise ValueError('Out-of-bag stopping requires complete geometry cluster identity')
    if 'split' in frame and not frame['split'].eq('development').all():
        raise ValueError('Calibration/test data cannot choose a member stopping epoch')
    clusters = np.asarray(sorted(frame.cluster_id.unique()))
    if len(clusters) < 2:
        raise ValueError('At least two development clusters are needed')
    sampled = np.random.default_rng(seed).choice(clusters,size=len(clusters),replace=True)
    training_clusters = np.unique(sampled)
    validation = frame.loc[~frame.cluster_id.isin(training_clusters)].copy()
    if validation.empty:
        raise ValueError('This declared bootstrap has no out-of-bag clusters; do not use training loss as validation')
    pieces = []
    for draw, cluster in enumerate(sampled):
        piece = frame.loc[frame.cluster_id == cluster].copy()
        piece['airfoil_id'] = piece.airfoil_id.astype(str) + f'__cluster_bootstrap_{draw:05d}'
        pieces.append(piece)
    bootstrap = pd.concat(pieces,ignore_index=True)
    if not set(bootstrap.cluster_id).isdisjoint(set(validation.cluster_id)):
        raise AssertionError('Training and stopping-validation clusters overlap')
    return bootstrap, validation, {
        'seed':seed, 'draws':len(sampled), 'unique_training_clusters':len(training_clusters),
        'sampled_cluster_ids':[int(x) for x in sampled],
        'out_of_bag_cluster_ids':[int(x) for x in sorted(validation.cluster_id.unique())],
        'out_of_bag_airfoil_ids':sorted(validation.airfoil_id.astype(str).unique().tolist()),
        'validation_policy':'out-of-bag development clusters only; calibration/test untouched',
    }
