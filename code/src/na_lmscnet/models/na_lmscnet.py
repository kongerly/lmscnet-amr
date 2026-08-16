"""Noise-aware lightweight multi-scale convolutional network."""

from __future__ import annotations

import math

import torch
from torch import nn

NA_LMSCNET_KERNELS = (3, 7, 15)
NA_LMSCNET_SNR_MIN_DB = -20.0
NA_LMSCNET_SNR_MAX_DB = 18.0


def _check_input(iq: torch.Tensor) -> None:
    if iq.ndim != 3 or iq.shape[1:] != (2, 128):
        raise ValueError(f"NA-LMSCNet expects [batch, 2, 128], got {tuple(iq.shape)}")
    if not iq.dtype.is_floating_point:
        raise TypeError("NA-LMSCNet input must use a floating-point dtype")


class _DynamicMultiScaleBlock(nn.Module):
    """Residual block with depthwise multi-scale branches and noise-conditioned fusion."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        width = math.ceil(1.25 * out_channels)
        hidden_gate = max(8, width // 4)
        self.projection = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                    groups=width,
                    bias=False,
                ),
                nn.BatchNorm1d(width),
                nn.SiLU(inplace=True),
            )
            for kernel in NA_LMSCNET_KERNELS
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * width + 8, hidden_gate),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_gate, 3),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(width, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, noise_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(x)
        branches = [branch(projected) for branch in self.branches]
        pooled = torch.cat([branch.mean(dim=-1) for branch in branches], dim=1)
        weights = torch.softmax(self.gate(torch.cat((pooled, noise_embedding), dim=1)), dim=1)
        fused = sum(
            branch * weights[:, index].view(-1, 1, 1) for index, branch in enumerate(branches)
        )
        output = torch.nn.functional.silu(self.fusion(fused) + self.shortcut(x))
        return output, weights


class _StaticScaleBlock(nn.Module):
    """Residual block with one or more equally weighted depthwise branches."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, kernels: tuple[int, ...]
    ) -> None:
        super().__init__()
        width = math.ceil(1.25 * out_channels)
        self.branch_count = len(kernels)
        self.projection = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                    groups=width,
                    bias=False,
                ),
                nn.BatchNorm1d(width),
                nn.SiLU(inplace=True),
            )
            for kernel in kernels
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(width, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(x)
        branches = [branch(projected) for branch in self.branches]
        fused = torch.stack(branches, dim=0).mean(dim=0)
        weights = projected.new_full(
            (projected.shape[0], self.branch_count), 1.0 / self.branch_count
        )
        output = torch.nn.functional.silu(self.fusion(fused) + self.shortcut(x))
        return output, weights


class NALMSCNet(nn.Module):
    """NA-LMSCNet with a fixed [3, 7, 15] depthwise multi-scale backbone."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a float in [0, 1)")
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.ModuleList(
            [_DynamicMultiScaleBlock(32, 32, 1), _DynamicMultiScaleBlock(32, 32, 1)]
        )
        self.stage2 = nn.ModuleList(
            [_DynamicMultiScaleBlock(32, 64, 2), _DynamicMultiScaleBlock(64, 64, 1)]
        )
        self.stage3 = nn.ModuleList(
            [_DynamicMultiScaleBlock(64, 96, 2), _DynamicMultiScaleBlock(96, 96, 1)]
        )
        self.snr_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
        )
        self.noise_embedding = nn.Sequential(nn.Linear(1, 8), nn.SiLU(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_input(iq)
        stem = self.stem(iq)
        snr_raw = self.snr_head(stem.mean(dim=-1)).squeeze(-1)
        snr_hat = 19.0 * torch.tanh(snr_raw) - 1.0
        snr_hat_norm = (snr_hat + 1.0) / 19.0
        noise_embedding = self.noise_embedding(snr_hat_norm.unsqueeze(-1))

        features = stem
        weights: list[torch.Tensor] = []
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                features, block_weights = block(features, noise_embedding)
                weights.append(block_weights)
        scale_weights = torch.stack(weights, dim=1)
        logits = self.classifier(features.mean(dim=-1))
        return {"logits": logits, "snr_hat": snr_hat, "scale_weights": scale_weights}


class NALMSCNetWithoutSNRAuxiliary(nn.Module):
    """NA-LMSCNet ablation with a learned constant fusion embedding."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a float in [0, 1)")
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.ModuleList(
            [_DynamicMultiScaleBlock(32, 32, 1), _DynamicMultiScaleBlock(32, 32, 1)]
        )
        self.stage2 = nn.ModuleList(
            [_DynamicMultiScaleBlock(32, 64, 2), _DynamicMultiScaleBlock(64, 64, 1)]
        )
        self.stage3 = nn.ModuleList(
            [_DynamicMultiScaleBlock(64, 96, 2), _DynamicMultiScaleBlock(96, 96, 1)]
        )
        self.constant_noise_embedding = nn.Parameter(torch.zeros(1, 8))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_input(iq)
        features = self.stem(iq)
        noise_embedding = self.constant_noise_embedding.expand(iq.shape[0], -1)
        weights: list[torch.Tensor] = []
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                features, block_weights = block(features, noise_embedding)
                weights.append(block_weights)
        scale_weights = torch.stack(weights, dim=1)
        logits = self.classifier(features.mean(dim=-1))
        return {"logits": logits, "scale_weights": scale_weights}


class _NALMSCNetStaticScaleAblation(nn.Module):
    """Shared implementation for static scale-fusion ablations with the SNR loss intact."""

    def __init__(
        self, *, kernels: tuple[int, ...], num_classes: int = 11, dropout: float = 0.2
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a float in [0, 1)")
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.ModuleList(
            [
                _StaticScaleBlock(32, 32, 1, kernels),
                _StaticScaleBlock(32, 32, 1, kernels),
            ]
        )
        self.stage2 = nn.ModuleList(
            [
                _StaticScaleBlock(32, 64, 2, kernels),
                _StaticScaleBlock(64, 64, 1, kernels),
            ]
        )
        self.stage3 = nn.ModuleList(
            [
                _StaticScaleBlock(64, 96, 2, kernels),
                _StaticScaleBlock(96, 96, 1, kernels),
            ]
        )
        self.snr_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_input(iq)
        stem = self.stem(iq)
        snr_raw = self.snr_head(stem.mean(dim=-1)).squeeze(-1)
        snr_hat = 19.0 * torch.tanh(snr_raw) - 1.0
        features = stem
        weights: list[torch.Tensor] = []
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                features, block_weights = block(features)
                weights.append(block_weights)
        logits = self.classifier(features.mean(dim=-1))
        return {
            "logits": logits,
            "snr_hat": snr_hat,
            "scale_weights": torch.stack(weights, dim=1),
        }


class NALMSCNetWithoutMultiScale(_NALMSCNetStaticScaleAblation):
    """Ablation retaining only the kernel-7 branch in every residual block."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__(kernels=(7,), num_classes=num_classes, dropout=dropout)


class NALMSCNetFixedAverage(_NALMSCNetStaticScaleAblation):
    """Ablation using fixed equal weights for the [3, 7, 15] branches."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__(kernels=NA_LMSCNET_KERNELS, num_classes=num_classes, dropout=dropout)


NA_LMSCNet = NALMSCNet


__all__ = [
    "NA_LMSCNET_KERNELS",
    "NA_LMSCNET_SNR_MAX_DB",
    "NA_LMSCNET_SNR_MIN_DB",
    "NA_LMSCNet",
    "NALMSCNet",
    "NALMSCNetFixedAverage",
    "NALMSCNetWithoutMultiScale",
    "NALMSCNetWithoutSNRAuxiliary",
]
