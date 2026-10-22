from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from robust_airfoil.hashing import hash_object, sha256_file
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
from robust_airfoil.optimisation.checkpointing import assert_campaign_resources


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
    resident_batches: bool = False,
    checkpoint_each_epoch: bool = False,
    resume: bool = False,
    enforce_resource_floors: bool = False,
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
    train_tensors = train_dataset.tensor_dict(device) if resident_batches else None
    validation_tensors = validation_dataset.tensor_dict(device) if resident_batches else None
    context = hash_object({
        "train_frame": hashlib.sha256(pd.util.hash_pandas_object(train_frame,index=True).to_numpy().tobytes()).hexdigest(),
        "validation_frame": hashlib.sha256(pd.util.hash_pandas_object(validation_frame,index=True).to_numpy().tobytes()).hexdigest(),
        "config": [seed, hidden_width, hidden_layers, dropout, learning_rate, weight_decay,
                   batch_size, maximum_epochs, patience, loss_name, activation, residual,
                   layer_norm, resident_batches, enforce_resource_floors],
        "scientific_source": {name: sha256_file(Path(__file__).with_name(name))
                              for name in ["train.py","dataset.py","models.py","losses.py","samplers.py"]},
    })
    state_path = output_dir / "training_state.pt"
    receipt_path = output_dir / "training_state.json"
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    first_epoch = 0
    if resume:
        if not state_path.is_file() or not receipt_path.is_file():
            raise ValueError("Resume requires a complete training context checkpoint")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["context"] != context or receipt["sha256"] != sha256_file(state_path):
            raise ValueError("Training context or checkpoint hash differs; refusing resume")
        state = torch.load(state_path,map_location="cpu",weights_only=False)
        if state["context"] != context:
            raise ValueError("Training context differs inside checkpoint")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        history = state["history"]
        first_epoch = int(state["epoch"]) + 1
        best_loss, best_epoch = state["best_loss"], state["best_epoch"]
        epochs_without_improvement = state["epochs_without_improvement"]
    elif state_path.exists() or (output_dir/"model.pt").exists():
        raise FileExistsError("Training output already exists; explicitly resume matching context")
    start_time = time.monotonic()
    for epoch in range(first_epoch, maximum_epochs):
        if enforce_resource_floors:
            assert_campaign_resources(output_dir)
        if epochs_without_improvement >= patience:
            break
        sampler.set_epoch(epoch)
        model.train()
        train_losses: list[float] = []
        training_batches = (
            _resident_batches(train_tensors, np.fromiter(iter(sampler),dtype=np.int64), batch_size)
            if train_tensors is not None else iter(train_loader)
        )
        for batch in training_batches:
            optimiser.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, batch, device, loss_name, uniform_airfoil_sampling=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        numerators = dict.fromkeys(TARGET_COLUMNS, 0.0)
        denominators = dict.fromkeys(TARGET_COLUMNS, 0.0)
        validation_batches = (
            _resident_batches(validation_tensors, np.arange(len(validation_dataset)), batch_size)
            if validation_tensors is not None else iter(validation_loader)
        )
        with torch.no_grad():
            for batch in validation_batches:
                _, components = _batch_loss(model, batch, device, loss_name)
                for target in TARGET_COLUMNS:
                    denominator = float((batch["airfoil_weight"] * batch[f"mask_{target}"].float()).sum().cpu())
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
        if checkpoint_each_epoch:
            state = {"context": context, "epoch": epoch, "model": model.state_dict(),
                     "optimiser": optimiser.state_dict(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                     "history": history, "best_loss": best_loss, "best_epoch": best_epoch,
                     "epochs_without_improvement": epochs_without_improvement}
            temporary = output_dir / "training_state.next.pt"
            torch.save(state,temporary)
            temporary.replace(state_path)
            receipt = {"context":context,"epoch":epoch,"sha256":sha256_file(state_path)}
            receipt_path.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
            (output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            print(f"{output_dir.name} epoch={epoch+1} train={history[-1]['train_loss']:.6f} validation={validation_loss:.6f} best_epoch={best_epoch+1}",flush=True)
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
    summary.update(training_context_hash=context, resident_batches=resident_batches, resumed_from_epoch=first_epoch, invocation_seconds=time.monotonic()-start_time)
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


def _resident_batches(tensors: dict[str, torch.Tensor], indices: np.ndarray, batch_size: int):
    """Yield the same collated fields without per-observation Tensor objects."""
    device = tensors["features"].device
    for start in range(0,len(indices),batch_size):
        selected = torch.as_tensor(indices[start:start+batch_size],dtype=torch.long,device=device)
        yield {name: values.index_select(0,selected) for name,values in tensors.items()}
