from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from robust_airfoil.config import TuningConfig
from robust_airfoil.data.splits import make_grouped_folds, write_fold_manifest
from robust_airfoil.modelling.train import train_model


def _cap_points(frame: pd.DataFrame, maximum_per_airfoil: int = 32) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["airfoil_id", "alpha_deg"]).groupby("airfoil_id", sort=True):
        if len(group) <= maximum_per_airfoil:
            pieces.append(group)
        else:
            positions = np.linspace(0, len(group) - 1, maximum_per_airfoil, dtype=int)
            pieces.append(group.iloc[positions])
    return pd.concat(pieces, ignore_index=True)


def tune_model(
    development_points: pd.DataFrame,
    development_airfoils: pd.DataFrame,
    output_root: Path,
    config: TuningConfig,
    trials: int | None = None,
) -> optuna.Study:
    output_root.mkdir(parents=True, exist_ok=True)
    folds = make_grouped_folds(development_airfoils, config.grouped_folds, config.seed)
    fold_manifest = write_fold_manifest(folds, output_root / "fold_manifest.json", config.seed, config.grouped_folds)
    storage = f"sqlite:///{(output_root / 'conditioned_polar_v2.sqlite3').resolve().as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.pruner.startup_trials,
        n_warmup_steps=0,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=config.study_name,
        storage=storage,
        load_if_exists=True,
        direction=config.direction,
        sampler=sampler,
        pruner=pruner,
    )
    search = config.search

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "hidden_width": trial.suggest_categorical("hidden_width", search.hidden_width_choices),
            "hidden_layers": trial.suggest_int("hidden_layers", *search.hidden_layers),
            "activation": trial.suggest_categorical("activation", search.activation_choices),
            "dropout": trial.suggest_float("dropout", *search.dropout),
            "learning_rate": trial.suggest_float("learning_rate", *search.learning_rate_log, log=True),
            "weight_decay": trial.suggest_float("weight_decay", *search.weight_decay_log, log=True),
            "batch_size": trial.suggest_categorical("batch_size", search.batch_size_choices),
            "loss_name": trial.suggest_categorical("loss", search.loss_choices),
            "residual": trial.suggest_categorical("residual", search.residual_choices),
        }
        fold_values: list[float] = []
        best_epochs: list[int] = []
        for fold in range(config.grouped_folds):
            validation_ids = folds.loc[folds["fold"] == fold, "airfoil_id"].astype(str).tolist()
            train_ids = folds.loc[folds["fold"] != fold, "airfoil_id"].astype(str).tolist()
            train = _cap_points(development_points[development_points["airfoil_id"].isin(train_ids)].copy())
            validation = _cap_points(development_points[development_points["airfoil_id"].isin(validation_ids)].copy())
            result = train_model(
                train,
                validation,
                output_root / f"trial_{trial.number:04d}" / f"fold_{fold}",
                seed=config.seed + trial.number * config.grouped_folds + fold,
                maximum_epochs=24,
                patience=6,
                layer_norm=True,
                **params,
            )
            value = float(result["metrics"]["cl"]["macro_mse"] + result["metrics"]["log_cd"]["macro_mse"])
            fold_values.append(value)
            best_epochs.append(int(result["best_epoch"]) + 1)
            trial.report(float(np.mean(fold_values)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_values", fold_values)
        trial.set_user_attr("best_epochs", best_epochs)
        trial.set_user_attr("fold_manifest_hash", fold_manifest["records_hash"])
        return float(np.mean(fold_values))

    target_trials = trials or config.automatic_initial_full_trials
    remaining = max(0, target_trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=config.n_jobs)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    summary = {
        "study_name": config.study_name,
        "requested_trials": target_trials,
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_epochs": study.best_trial.user_attrs.get("best_epochs", []),
        "fold_manifest_hash": fold_manifest["records_hash"],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return study
