"""
Common components and utilities for models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResidualBlock(nn.Module):
    """A residual block for 1D data."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class AttentionLayer(nn.Module):
    """Self-attention layer for sequence data."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        out = x + self.dropout(attn_out)
        out = self.norm(out)
        return out


def initialize_weights(module: nn.Module, init_type: str = "kaiming_uniform") -> None:
    """Initialize network weights using specified initialization method."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if init_type == "kaiming_uniform":
            init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                fan_in = init._calculate_fan_in_and_fan_out(module.weight)[0]
                bound = 1 / (fan_in**0.5)
                init.uniform_(module.bias, -bound, bound)
        elif init_type == "kaiming_normal":
            init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif init_type == "xavier_uniform":
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


class AdaptiveConcatPool1d(nn.Module):
    """Adaptive pooling that concatenates max and average pooling results."""

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(output_size)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.max_pool(x), self.avg_pool(x)], 1)


class SqueezeExcitation1d(nn.Module):
    """Squeeze-and-Excitation block for 1D data."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
