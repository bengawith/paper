from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from robust_airfoil.modelling.dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    PolarPointDataset,
    ScalingBundle,
    fit_scaling,
)
from robust_airfoil.modelling.evaluate import regression_metrics
from robust_airfoil.modelling.losses import masked_macro_multitask_loss
from robust_airfoil.modelling.models import ConditionedPolarMLP
from robust_airfoil.modelling.samplers import UniformAirfoilPointSampler


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batch_loss(model: ConditionedPolarMLP, batch: dict[str, torch.Tensor], device: torch.device, loss_name: str, *, uniform_airfoil_sampling: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features = batch["features"].to(device)
    predictions = model(features)
    targets = {name: batch[name].to(device) for name in TARGET_COLUMNS}
    masks = {name: batch[f"mask_{name}"].to(device) for name in TARGET_COLUMNS}
    weights = batch["airfoil_weight"].to(device)
    if uniform_airfoil_sampling:
        weights = torch.ones_like(weights)
    return masked_macro_multitask_loss(predictions, targets, masks, weights, loss_name, 1.0)


def overfit_one_batch(frame: pd.DataFrame, seed: int = 20260824, steps: int = 800) -> dict[str, float | bool]:
    set_deterministic_seed(seed)
    scaling = fit_scaling(frame)
    dataset = PolarPointDataset(frame.iloc[: min(len(frame), 256)].copy(), scaling)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedPolarMLP(hidden_width=128, hidden_layers=2, dropout=0).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = float(_batch_loss(model, batch, device, "huber")[0].detach().cpu())
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(model, batch, device, "huber")
        loss.backward()
        optimiser.step()
    final = float(_batch_loss(model, batch, device, "huber")[0].detach().cpu())
    reduction = 1 - final / max(initial, 1e-12)
    return {"initial_loss": initial, "final_loss": final, "relative_reduction": reduction, "passed": reduction >= 0.99}


def train_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    output_dir: Path,
    seed: int = 20260824,
    hidden_width: int = 256,
    hidden_layers: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 1024,
    maximum_epochs: int = 300,
    patience: int = 25,
    loss_name: str = "huber",
    activation: str = "silu",
    residual: bool = True,
    layer_norm: bool = True,
) -> dict[str, object]:
    set_deterministic_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaling = fit_scaling(train_frame)
    train_dataset = PolarPointDataset(train_frame, scaling)
    validation_dataset = PolarPointDataset(validation_frame, scaling)
    sampler = UniformAirfoilPointSampler(train_dataset.airfoil_ids, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), sampler=sampler)
    validation_loader = DataLoader(validation_dataset, batch_size=min(batch_size, len(validation_dataset)), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedPolarMLP(13, hidden_width, hidden_layers, dropout, activation, residual, layer_norm).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(maximum_epochs):
        sampler.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, batch, device, loss_name, uniform_airfoil_sampling=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        # Accumulate weighted sums/counts, not an unweighted mean of batch
        # means (which depends on the final batch size and target masks).
        numerators = dict.fromkeys(TARGET_COLUMNS, 0.0)
        denominators = dict.fromkeys(TARGET_COLUMNS, 0.0)
        with torch.no_grad():
            for batch in validation_loader:
                _, components = _batch_loss(model, batch, device, loss_name)
                for target in TARGET_COLUMNS:
                    denominator = float((batch["airfoil_weight"] * batch[f"mask_{target}"].float()).sum())
                    numerators[target] += float(components[target].cpu()) * denominator
                    denominators[target] += denominator
        active_losses = [numerators[target] / denominators[target] for target in TARGET_COLUMNS if denominators[target] > 0]
        if not active_losses:
            raise ValueError("Validation has no active targets")
        validation_loss = float(np.mean(active_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-7:
            best_loss, best_epoch = validation_loss, epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    model.load_state_dict(torch.load(output_dir / "model.pt", map_location=device, weights_only=True))
    predictions = predict_frame(model, validation_frame, scaling, device)
    metrics = regression_metrics(predictions, {name: f"prediction_{name}" for name in TARGET_COLUMNS})
    architecture = {
        "input_dim": 13,
        "hidden_width": hidden_width,
        "hidden_layers": hidden_layers,
        "dropout": dropout,
        "activation": activation,
        "residual": residual,
        "layer_norm": layer_norm,
    }
    torch.save({"state_dict": model.state_dict(), "architecture": architecture}, output_dir / "checkpoint.pt")
    import joblib

    joblib.dump(scaling, output_dir / "scaling.joblib")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"device": str(device), "best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs": len(history), "metrics": metrics, "scaling": asdict(scaling), "architecture": architecture, "seed": seed}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value), encoding="utf-8")
    return summary


def predict_frame(model: ConditionedPolarMLP, frame: pd.DataFrame, scaling: ScalingBundle, device: torch.device | None = None) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    features = ((frame[FEATURE_COLUMNS].to_numpy(float) - scaling.feature_center) / scaling.feature_scale).astype(np.float32)
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in TARGET_COLUMNS}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            batch = torch.from_numpy(features[start : start + 4096])
            prediction = model(batch.to(device))
            for name in TARGET_COLUMNS:
                outputs[name].append(prediction[name].cpu().numpy())
    result = frame.copy()
    for name in TARGET_COLUMNS:
        standardised = np.concatenate(outputs[name])
        result[f"prediction_{name}"] = standardised * scaling.target_scale[name] + scaling.target_center[name]
    return result
