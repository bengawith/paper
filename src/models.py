"""
Improved neural network models with modern architecture components and better efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from .model_components import (
    ResidualBlock,
    AttentionLayer,
    initialize_weights,
    AdaptiveConcatPool1d,
    SqueezeExcitation1d,
)


class GRUModel(nn.Module):
    """
    GRU model with attention and residual connections.

    Features:
    - Bidirectional GRU layers
    - Self-attention mechanism
    - Residual connections
    - Dropout for regularization
    - Layer normalization
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = AttentionLayer(
            hidden_size * self.num_directions, num_heads=4, dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.apply(lambda m: initialize_weights(m, "kaiming_normal"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)
        device = x.device

        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

        # GRU forward pass
        out, _ = self.gru(x, h0)

        # Apply attention
        out = self.attention(out)

        # Use the last time step from each direction
        out = out[:, -1, :]

        # Final fully connected layers with residual connection
        identity = self.fc1(out)  # First transform identity to match dimensions
        out = F.relu(identity)
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out + identity

        return self.fc2(out)


class LSTMModel(nn.Module):
    """
    LSTM model with attention, residual connections, and adaptive features.

    Features:
    - Bidirectional LSTM layers
    - Self-attention mechanism
    - Residual connections
    - Layer normalization
    - Dropout for regularization
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = AttentionLayer(
            hidden_size * self.num_directions, num_heads=4, dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.apply(lambda m: initialize_weights(m, "kaiming_normal"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)
        device = x.device

        # Initialize hidden and cell states
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Apply attention
        out = self.attention(out)

        # Use the last time step from each direction
        out = out[:, -1, :]

        # Final fully connected layers with residual connection
        identity = self.fc1(out)
        out = F.relu(identity)
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out + identity

        return self.fc2(out)


class MLPModel(nn.Module):
    """
    MLP model with modern architecture features.

    Features:
    - Residual connections
    - Layer normalization
    - Adaptive dropout
    - Wide and deep architecture
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        width_factor: int = 2,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # Wide branch (skip connection)
        self.wide = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * width_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * width_factor, hidden_size),
        )

        # Deep branch (main path)
        deep_layers = []
        for _ in range(num_layers):
            deep_layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
        self.deep = nn.Sequential(*deep_layers)

        # Combine branches
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout / 2),  # Lower dropout at the end
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.apply(lambda m: initialize_weights(m, "kaiming_normal"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # If input is (batch, seq_len, features), flatten it
            x = x.reshape(x.size(0), -1)

        # Input layer
        x = F.relu(self.input_layer(x))
        x = self.input_norm(x)

        # Process wide and deep branches
        wide = self.wide(x)
        deep = self.deep(x)

        # Combine branches
        combined = torch.cat([wide, deep], dim=1)
        out = self.combine(combined)

        return self.output_layer(out)


class CNNModel(nn.Module):
    """
    1D CNN model with modern architecture components.

    Features:
    - 1D convolutions for sequential/feature data
    - Residual connections
    - Squeeze-and-Excitation blocks
    - Batch normalization
    - Adaptive pooling
    - Multi-scale feature processing

    Input:
        - 2D tensor (batch, features): Automatically converts to (batch, 1, features)
        - 3D tensor (batch, channels, length): Uses as-is
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_filters: List[int],
        kernel_sizes: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input layer with proper channel handling (starts with 1 channel)
        self.input_conv = nn.Sequential(
            nn.Conv1d(
                1,  # Single channel input (features treated as sequence)
                num_filters[0],
                kernel_sizes[0],
                padding=kernel_sizes[0] // 2,
            ),
            nn.BatchNorm1d(num_filters[0]),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # Create feature pyramid
        self.pyramid = nn.ModuleList()
        for i in range(len(num_filters) - 1):
            self.pyramid.append(
                nn.Sequential(
                    ResidualBlock(num_filters[i], kernel_sizes[i], dropout),
                    SqueezeExcitation1d(num_filters[i]),
                    nn.Conv1d(
                        num_filters[i],
                        num_filters[i + 1],
                        1,  # 1x1 conv for channel adjustment
                        bias=False,
                    ),
                    nn.BatchNorm1d(num_filters[i + 1]),
                    nn.ReLU(),
                )
            )

        # Global context block
        self.global_context = nn.Sequential(
            AdaptiveConcatPool1d(1),  # Combines max and avg pooling
            nn.Flatten(),
            nn.Linear(num_filters[-1] * 2, num_filters[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(num_filters[-1], num_filters[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(num_filters[-1] // 2, output_size),
        )

        # Initialize weights for better gradient flow
        self.apply(lambda m: initialize_weights(m, "kaiming_uniform"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is in the right shape (batch, channels, length)
        if x.dim() == 2:
            # If input is (batch, features), reshape to (batch, 1, features)
            x = x.unsqueeze(1)  # Add channel dimension at position 1
        elif x.dim() == 3 and x.size(1) != x.size(-1):
            # If input is (batch, length, features), transpose to (batch, features, length)
            x = x.transpose(1, 2)

        # Initial convolution
        x = self.input_conv(x)

        # Process through feature pyramid
        for pyramid_layer in self.pyramid:
            x = pyramid_layer(x)

        # Global context and final prediction
        x = self.global_context(x)
        x = self.classifier(x)

        return x


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """Create an optimizer with proper parameter grouping and weight decay."""
    # Separate parameters into weight decay and no weight decay groups
    decay = set()
    no_decay = set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name

            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight"):
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    decay.add(fpn)
                elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                    no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(decay)],
            "weight_decay": weight_decay,
        },
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]

    # Choose optimizer
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(optim_groups, lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def instantiate_model(
    net_name: str,
    input_size: int,
    output_size: int,
    params: Dict[str, Any],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Create and initialize an improved model based on the specified architecture."""
    if net_name == "GRUModel":
        model = GRUModel(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            output_size=output_size,
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.1),
            bidirectional=params.get("bidirectional", True),
        )
    elif net_name == "LSTMModel":
        model = LSTMModel(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            output_size=output_size,
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.1),
            bidirectional=params.get("bidirectional", True),
        )
    elif net_name == "MLPModel":
        model = MLPModel(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            output_size=output_size,
            num_layers=params.get("num_layers", 4),
            dropout=params.get("dropout", 0.1),
            width_factor=params.get("width_factor", 2),
        )
    elif net_name == "CNNModel":
        model = CNNModel(
            input_size=input_size,
            output_size=output_size,
            num_filters=params["num_filters"],
            kernel_sizes=params["kernel_sizes"],
            dropout=params.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown network type: {net_name}")

    return model.to(device)
