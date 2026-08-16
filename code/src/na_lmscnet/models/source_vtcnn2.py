"""PyTorch port of the original RadioML VT-CNN2 notebook architecture."""

from __future__ import annotations

import torch
from torch import nn


class SourceVTCNN2(nn.Module):
    """Source-aligned VT-CNN2 with notebook padding and initializers."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.5) -> None:
        super().__init__()
        if num_classes < 2 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid VT-CNN2 constructor arguments")
        self.pad1 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3))
        self.drop1 = nn.Dropout(dropout)
        self.pad2 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3))
        self.drop2 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(80 * 132, 256)
        self.drop3 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(256, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity="relu")
        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            nn.init.zeros_(layer.bias)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1:] != (2, 128):
            raise ValueError(f"SourceVTCNN2 expects [batch,2,128], got {tuple(iq.shape)}")
        value = iq.unsqueeze(1)
        value = self.drop1(torch.relu(self.conv1(self.pad1(value))))
        value = self.drop2(torch.relu(self.conv2(self.pad2(value))))
        value = value.flatten(1)
        value = self.drop3(torch.relu(self.dense1(value)))
        return self.dense2(value)
