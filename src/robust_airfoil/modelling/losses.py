from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as functional


def _element_loss(prediction: Tensor, target: Tensor, loss_name: str, huber_delta: float) -> Tensor:
    if loss_name == "huber":
        return functional.huber_loss(prediction, target, reduction="none", delta=huber_delta)
    if loss_name == "logcosh":
        error = prediction - target
        return error + functional.softplus(-2 * error) - torch.log(torch.tensor(2.0, device=error.device))
    if loss_name == "mse":
        return (prediction - target) ** 2
    raise ValueError(f"Unsupported loss: {loss_name}")


def masked_macro_multitask_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    masks: dict[str, Tensor],
    airfoil_weights: Tensor,
    loss_name: str,
    huber_delta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    components: dict[str, Tensor] = {}
    for name, prediction in predictions.items():
        mask = masks[name].bool()
        if not torch.any(mask):
            components[name] = prediction.sum() * 0
            continue
        element = _element_loss(prediction[mask], targets[name][mask], loss_name, huber_delta)
        weights = airfoil_weights[mask]
        components[name] = torch.sum(element * weights) / torch.clamp(torch.sum(weights), min=1e-12)
    active = [value for name, value in components.items() if torch.any(masks[name].bool())]
    if not active:
        raise ValueError("Batch has no valid targets")
    return torch.stack(active).mean(), components
