"""CNN2 baseline for RadioML 2016.10A."""

from __future__ import annotations

import torch
from torch import nn


class CNN2(nn.Module):
    """Two-convolution baseline operating on the 2 x 128 I/Q plane."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a float in [0, 1)")
        self.features = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 124, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1:] != (2, 128):
            raise ValueError(f"CNN2 expects [batch, 2, 128], got {tuple(iq.shape)}")
        if not iq.dtype.is_floating_point:
            raise TypeError("CNN2 input must use a floating-point dtype")
        return self.classifier(self.features(iq.unsqueeze(1)))
