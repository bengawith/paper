from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float, activation: type[nn.Module], residual: bool, layer_norm: bool):
        super().__init__()
        normalisation: nn.Module = nn.LayerNorm(width) if layer_norm else nn.Identity()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            normalisation,
            activation(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width) if layer_norm else nn.Identity(),
        )
        self.activation = activation()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.block(x)
        return self.activation(x + transformed if self.residual else transformed)


class ConditionedPolarMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_width: int = 256,
        hidden_layers: int = 3,
        dropout: float = 0.10,
        activation: str = "silu",
        residual: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        activation_type = _activation(activation)
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.LayerNorm(hidden_width) if layer_norm else nn.Identity(),
            activation_type(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(hidden_width, dropout, activation_type, residual, layer_norm)
                for _ in range(hidden_layers)
            ]
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_width, 1) for name in ("cl", "log_cd", "cm")})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.input(x))
        return {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}
