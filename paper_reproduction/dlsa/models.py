from __future__ import annotations

import torch
from torch import nn


class CausalConvolutionBlock(nn.Module):
    """Two causal convolutions with the paper's normalization and skip path."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        kernel_size: int = 2,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if out_channels % in_channels != 0:
            raise ValueError("out_channels must be divisible by in_channels for the skip path.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.input_norm = nn.InstanceNorm1d(in_channels)
        self.hidden_norm = nn.InstanceNorm1d(out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size)
        self.activation = nn.ReLU()
        self.left_pad = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        hidden = self.input_norm(values) if self.normalize else values
        hidden = self.activation(self.conv1(self.left_pad(hidden)))
        hidden = self.hidden_norm(hidden) if self.normalize else hidden
        hidden = self.activation(self.conv2(self.left_pad(hidden)))
        repeats = self.out_channels // self.in_channels
        return hidden + residual.repeat(1, repeats, 1)


class CNNTransformerAllocation(nn.Module):
    """Paper-aligned CNN+Transformer allocation model.

    Input has shape ``(residual_windows, lookback)`` and output is one raw
    allocation score per residual window. Cross-sectional portfolio
    normalization is deliberately handled outside the network.
    """

    def __init__(
        self,
        *,
        filters: int = 8,
        kernel_size: int = 2,
        attention_heads: int = 4,
        hidden_units: int = 16,
        dropout: float = 0.25,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.convolution = CausalConvolutionBlock(
            in_channels=1,
            out_channels=filters,
            kernel_size=kernel_size,
            normalize=True,
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=filters,
            nhead=attention_heads,
            dim_feedforward=hidden_units,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=False,
        )
        self.allocation = nn.Linear(filters, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 2:
            raise ValueError("windows must have shape (observations, lookback).")
        hidden = self.convolution(windows.unsqueeze(1))
        hidden = hidden.permute(2, 0, 1)
        hidden = self.transformer(hidden)
        return self.allocation(hidden[-1]).squeeze(-1)


class FeedForwardAllocation(nn.Module):
    """Feedforward allocation used by the Fourier and raw-signal benchmarks."""

    def __init__(
        self,
        *,
        input_size: int = 30,
        hidden_units: tuple[int, ...] = (16, 8, 4),
        dropout: float = 0.25,
        activation: str = "relu",
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        dimensions = (input_size, *hidden_units)
        layers: list[nn.Module] = []
        activation_type: type[nn.Module]
        if activation == "relu":
            activation_type = nn.ReLU
        elif activation == "sigmoid":
            activation_type = nn.Sigmoid
        else:
            raise ValueError("activation must be 'relu' or 'sigmoid'.")
        for input_dim, output_dim in zip(dimensions, dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    activation_type(),
                    nn.Dropout(dropout),
                ]
            )
        self.hidden = nn.Sequential(*layers)
        self.allocation = nn.Linear(dimensions[-1], 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape (observations, features).")
        return self.allocation(self.hidden(features)).squeeze(-1)
