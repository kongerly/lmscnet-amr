"""One-dimensional residual baseline for the frozen RadioML interface."""

from __future__ import annotations

import torch
from torch import nn


class _BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(dropout) if dropout else nn.Identity()
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class ResNet1D(nn.Module):
    """A compact four-stage, two-block-per-stage ResNet-18 analogue."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a float in [0, 1)")
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.stage1 = self._stage(32, 32, stride=1, dropout=dropout)
        self.stage2 = self._stage(32, 64, stride=2, dropout=dropout)
        self.stage3 = self._stage(64, 128, stride=2, dropout=dropout)
        self.stage4 = self._stage(128, 256, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _stage(
        in_channels: int, out_channels: int, *, stride: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            _BasicBlock1D(in_channels, out_channels, stride, dropout),
            _BasicBlock1D(out_channels, out_channels, 1, dropout),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1:] != (2, 128):
            raise ValueError(f"ResNet1D expects [batch, 2, 128], got {tuple(iq.shape)}")
        if not iq.dtype.is_floating_point:
            raise TypeError("ResNet1D input must use a floating-point dtype")
        features = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(iq)))))
        return self.classifier(self.pool(features).flatten(1))
