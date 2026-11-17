"""Install the tested resident-batch/checkpoint path into the actual trainer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
p = ROOT/'src/robust_airfoil/modelling/train.py'
t = p.read_text(encoding='utf-8')
assert 'resident_batches' not in t, 'Inspect an existing patch; do not apply twice'
t = t.replace('import json\n', 'import hashlib\nimport json\nimport time\n')
t = t.replace('from robust_airfoil.modelling.dataset import (', 'from robust_airfoil.hashing import hash_object, sha256_file\nfrom robust_airfoil.modelling.dataset import (')
t = t.replace('    layer_norm: bool = True,\n) -> dict[str, object]:', '    layer_norm: bool = True,\n    resident_batches: bool = False,\n    checkpoint_each_epoch: bool = False,\n    resume: bool = False,\n) -> dict[str, object]:')
start = t.index('    best_loss = float("inf")',t.index('def train_model('))
end = t.index('    model.load_state_dict(', start)
replacement = '''    train_tensors = train_dataset.tensor_dict(device) if resident_batches else None
    validation_tensors = validation_dataset.tensor_dict(device) if resident_batches else None
    context = hash_object({
        "train_frame": hashlib.sha256(pd.util.hash_pandas_object(train_frame,index=True).values.tobytes()).hexdigest(),
        "validation_frame": hashlib.sha256(pd.util.hash_pandas_object(validation_frame,index=True).values.tobytes()).hexdigest(),
        "config": [seed, hidden_width, hidden_layers, dropout, learning_rate, weight_decay,
                   batch_size, maximum_epochs, patience, loss_name, activation, residual,
                   layer_norm, resident_batches],
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
'''
t = t[:start]+replacement+t[end:]
t = t.replace('    (output_dir / "metrics.json").write_text', '    summary.update(training_context_hash=context, resident_batches=resident_batches, resumed_from_epoch=first_epoch, invocation_seconds=time.monotonic()-start_time)\n    (output_dir / "metrics.json").write_text')
helper = '''\n\ndef _resident_batches(tensors: dict[str, torch.Tensor], indices: np.ndarray, batch_size: int):
    """Yield the same collated fields without per-observation Tensor objects."""
    device = tensors["features"].device
    for start in range(0,len(indices),batch_size):
        selected = torch.as_tensor(indices[start:start+batch_size],dtype=torch.long,device=device)
        yield {name: values.index_select(0,selected) for name,values in tensors.items()}
'''
t += helper
p.write_text(t,encoding='utf-8')
print('ACTUAL_TRAINER_PATCHED')
