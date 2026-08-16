"""CLDNN baseline adapted to the frozen [B, 2, 128] I/Q interface."""

from __future__ import annotations

import torch
from torch import nn


class CLDNN(nn.Module):
    """Three temporal convolution blocks followed by an LSTM classifier."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a float in [0, 1)")
        self.conv1 = self._conv_block(1, 50, dropout)
        self.conv2 = self._conv_block(50, 50, dropout)
        self.conv3 = self._conv_block(50, 50, dropout)
        self.recurrent = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 8), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1:] != (2, 128):
            raise ValueError(f"CLDNN expects [batch, 2, 128], got {tuple(iq.shape)}")
        if not iq.dtype.is_floating_point:
            raise TypeError("CLDNN input must use a floating-point dtype")
        shallow = self.conv1(iq.unsqueeze(1))
        deep = self.conv3(self.conv2(shallow))
        features = torch.cat((shallow, deep), dim=3)
        features = features.flatten(start_dim=2).transpose(1, 2)
        sequence, _ = self.recurrent(features)
        return self.classifier(sequence[:, -1])
