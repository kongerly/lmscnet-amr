"""Source-traceable extended baselines for the final AMR comparison."""

from __future__ import annotations

import torch
from torch import nn


def _validate_inputs(iq: torch.Tensor, model: str, *, flexible_length: bool = False) -> None:
    valid_shape = (
        iq.ndim == 3
        and iq.shape[1] == 2
        and (iq.shape[2] >= 16 if flexible_length else iq.shape[2] == 128)
    )
    if not valid_shape:
        expected = "[batch, 2, length] with length >= 16" if flexible_length else "[batch, 2, 128]"
        raise ValueError(f"{model} expects {expected}, got {tuple(iq.shape)}")
    if not iq.dtype.is_floating_point:
        raise TypeError(f"{model} input must use a floating-point dtype")


def _validate_hyperparameters(num_classes: int, dropout: float) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a float in [0, 1)")


class _ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout) if dropout else nn.Identity(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.main(inputs) + self.shortcut(inputs))


class ResNet1DMACMatched(nn.Module):
    """A frozen-width ResNet1D control chosen before formal training to match S2 MACs."""

    BASE_CHANNELS = 29

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        _validate_hyperparameters(num_classes, dropout)
        channels = [self.BASE_CHANNELS * 2**index for index in range(4)]
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        stages = []
        in_channels = channels[0]
        for index, out_channels in enumerate(channels):
            stride = 1 if index == 0 else 2
            stages.append(
                nn.Sequential(
                    _ResidualBlock1D(in_channels, out_channels, stride, dropout),
                    _ResidualBlock1D(out_channels, out_channels, 1, dropout),
                )
            )
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[-1], num_classes))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        _validate_inputs(iq, "ResNet1DMACMatched")
        return self.classifier(self.pool(self.stages(self.stem(iq))).flatten(1))


class _InvertedResidual1D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, expansion: int
    ) -> None:
        super().__init__()
        hidden = in_channels * expansion
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden, 1, bias=False),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv1d(
                    hidden,
                    hidden,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden,
                    bias=False,
                ),
                nn.BatchNorm1d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv1d(hidden, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layers(inputs)
        return inputs + outputs if self.use_residual else outputs


class MobileNetV2_1D(nn.Module):
    """A one-dimensional MobileNetV2-style raw-I/Q baseline."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        _validate_hyperparameters(num_classes, dropout)
        self.stem = nn.Sequential(
            nn.Conv1d(2, 24, 3, padding=1, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU6(inplace=True),
        )
        settings = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 2, 2))
        blocks: list[nn.Module] = []
        in_channels = 24
        for expansion, out_channels, repeats, first_stride in settings:
            for repeat in range(repeats):
                stride = first_stride if repeat == 0 else 1
                blocks.append(_InvertedResidual1D(in_channels, out_channels, stride, expansion))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, 120, 1, bias=False),
            nn.BatchNorm1d(120),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(120, num_classes))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        _validate_inputs(iq, "MobileNetV2_1D")
        return self.classifier(self.pool(self.head(self.blocks(self.stem(iq)))).flatten(1))


class MCLDNN(nn.Module):
    """MCLDNN adapted from the fixed MIT SigDA source to one canonical input tensor."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        _validate_hyperparameters(num_classes, dropout)
        self.full_branch = nn.Sequential(
            nn.ZeroPad2d((3, 4, 1, 0)),
            nn.Conv2d(1, 50, kernel_size=(2, 8)),
            nn.ReLU(inplace=True),
        )
        self.i_branch = self._component_branch()
        self.q_branch = self._component_branch()
        self.component_fusion = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(50, 50, kernel_size=(1, 8)),
            nn.ReLU(inplace=True),
        )
        self.joint = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
        )
        self.recurrent = nn.LSTM(100, 128, num_layers=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(dropout),
            nn.SELU(inplace=True),
            nn.Linear(128, 128),
            nn.Dropout(dropout),
            nn.SELU(inplace=True),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _component_branch() -> nn.Sequential:
        return nn.Sequential(
            nn.ConstantPad1d((7, 0), 0.0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(inplace=True),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        _validate_inputs(iq, "MCLDNN")
        full = self.full_branch(iq.unsqueeze(1))
        i_features = self.i_branch(iq[:, :1]).unsqueeze(2)
        q_features = self.q_branch(iq[:, 1:]).unsqueeze(2)
        components = self.component_fusion(torch.cat((i_features, q_features), dim=2))
        sequence = self.joint(torch.cat((full, components), dim=1)).squeeze(2).transpose(1, 2)
        recurrent, _ = self.recurrent(sequence)
        return self.classifier(recurrent[:, -1])


class _SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.gate(self.pool(inputs).squeeze(-1)).unsqueeze(-1)
        return inputs * weights


class _SEMBottleneck(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channels, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 9, stride=stride, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            _SqueezeExcitation1D(channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(channels),
            )
            if stride != 1
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.main(inputs) + self.shortcut(inputs))


class SEMSFN1D(nn.Module):
    """Source-informed 1D adaptation of SE-MSFN for 128-sample RadioML windows."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        _validate_hyperparameters(num_classes, dropout)
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.high = _SEMBottleneck(32, stride=1)
        self.medium = _SEMBottleneck(32, stride=2)
        self.low = _SEMBottleneck(32, stride=2)
        self.high_to_low = nn.Sequential(
            nn.Conv1d(32, 32, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
        )
        self.medium_to_low = nn.Sequential(
            nn.Conv1d(32, 32, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
        )
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            _SEMBottleneck(32, stride=2),
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        _validate_inputs(iq, "SEMSFN1D", flexible_length=True)
        high = self.high(self.stem(iq))
        medium = self.medium(high)
        low = self.low(medium)
        fused = low + self.medium_to_low(medium) + self.high_to_low(high)
        return self.classifier(self.output(fused).flatten(1))


__all__ = ["MCLDNN", "MobileNetV2_1D", "ResNet1DMACMatched", "SEMSFN1D"]
