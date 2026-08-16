"""Final S0/S1/S2 lightweight multi-scale models without SNR features."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Literal

import torch
from torch import nn

FINAL_LMSCNET_KERNELS = (3, 7, 15)
FusionMode = Literal["equal", "learned_static", "content_adaptive"]
NUM_FINAL_BLOCKS = 6


def _check_input(iq: torch.Tensor) -> None:
    if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[2] < 16:
        raise ValueError(
            f"Final LMSCNet expects [batch, 2, length] with length >= 16, got {tuple(iq.shape)}"
        )
    if not iq.dtype.is_floating_point:
        raise TypeError("Final LMSCNet input must use a floating-point dtype")


def _validate_common(num_classes: int, dropout: float, expansion: float) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    if not isinstance(dropout, float) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a float in [0, 1)")
    if not isinstance(expansion, float) or not math.isfinite(expansion) or expansion < 1.0:
        raise ValueError("expansion must be a finite float of at least one")


class _FinalScaleBlock(nn.Module):
    """Residual depthwise block with fixed or content-adaptive scale fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        kernels: tuple[int, ...],
        expansion: float,
        fusion_mode: FusionMode,
    ) -> None:
        super().__init__()
        width = math.ceil(expansion * out_channels)
        self.kernels = kernels
        self.fusion_mode = fusion_mode
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
        if fusion_mode == "content_adaptive":
            hidden_gate = max(8, width // 4)
            self.gate: nn.Module | None = nn.Sequential(
                nn.Linear(len(kernels) * width, hidden_gate),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_gate, len(kernels)),
            )
        else:
            self.gate = None
        self.static_logits = (
            nn.Parameter(torch.zeros(len(kernels))) if fusion_mode == "learned_static" else None
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
        self, x: torch.Tensor, gate_override: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(x)
        branches = [branch(projected) for branch in self.branches]
        if gate_override is not None:
            if gate_override.ndim != 2 or gate_override.shape != (projected.shape[0], len(branches)):
                raise ValueError("gate_override must have shape [batch, number_of_branches]")
            weights = gate_override.to(dtype=projected.dtype, device=projected.device)
            if not torch.isfinite(weights).all() or not torch.allclose(
                weights.sum(dim=1), torch.ones(projected.shape[0], device=projected.device)
            ):
                raise ValueError("gate_override must be finite and row-normalized")
        elif self.fusion_mode == "equal":
            weights = projected.new_full((projected.shape[0], len(branches)), 1.0 / len(branches))
        elif self.fusion_mode == "learned_static":
            assert self.static_logits is not None
            weights = torch.softmax(self.static_logits, dim=0).expand(projected.shape[0], -1)
        else:
            assert self.gate is not None
            pooled_content = torch.cat([branch.mean(dim=-1) for branch in branches], dim=1)
            weights = torch.softmax(self.gate(pooled_content), dim=1)
        fused = sum(
            branch * weights[:, index].view(-1, 1, 1) for index, branch in enumerate(branches)
        )
        output = torch.nn.functional.silu(self.fusion(fused) + self.shortcut(x))
        return output, weights


class FinalLMSCNet(nn.Module):
    """Shared backbone for the final nested S0/S1/S2 experiment family."""

    def __init__(
        self,
        *,
        kernels: tuple[int, ...],
        content_adaptive: bool | None = None,
        fusion_mode: FusionMode | None = None,
        num_classes: int = 11,
        dropout: float = 0.2,
        expansion: float = 1.25,
    ) -> None:
        super().__init__()
        _validate_common(num_classes, dropout, expansion)
        if fusion_mode is None:
            fusion_mode = "content_adaptive" if content_adaptive else "equal"
        if fusion_mode not in {"equal", "learned_static", "content_adaptive"}:
            raise ValueError(f"unsupported fusion mode: {fusion_mode}")
        if not kernels or any(kernel not in FINAL_LMSCNET_KERNELS for kernel in kernels):
            raise ValueError("kernels must be a non-empty ordered subset of (3, 7, 15)")
        if len(set(kernels)) != len(kernels):
            raise ValueError("kernels must not contain duplicates")
        if content_adaptive and kernels != FINAL_LMSCNET_KERNELS:
            raise ValueError("content-adaptive fusion is frozen to kernels (3, 7, 15)")
        self.kernels = kernels
        self.content_adaptive = fusion_mode == "content_adaptive"
        self.expansion = expansion
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
        )

        def block(in_channels: int, out_channels: int, stride: int) -> _FinalScaleBlock:
            return _FinalScaleBlock(
                in_channels,
                out_channels,
                stride,
                kernels=kernels,
                expansion=expansion,
                fusion_mode=fusion_mode,
            )

        self.stage1 = nn.ModuleList([block(32, 32, 1), block(32, 32, 1)])
        self.stage2 = nn.ModuleList([block(32, 64, 2), block(64, 64, 1)])
        self.stage3 = nn.ModuleList([block(64, 96, 2), block(96, 96, 1)])
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(
        self, iq: torch.Tensor, gate_override: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        _check_input(iq)
        if gate_override is not None:
            expected = (iq.shape[0], NUM_FINAL_BLOCKS, len(self.kernels))
            if tuple(gate_override.shape) != expected:
                raise ValueError(f"gate_override must have shape {expected}")
        features = self.stem(iq)
        weights: list[torch.Tensor] = []
        block_index = 0
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                override = None if gate_override is None else gate_override[:, block_index, :]
                features, block_weights = block(features, override)
                weights.append(block_weights)
                block_index += 1
        return {
            "logits": self.classifier(features.mean(dim=-1)),
            "scale_weights": torch.stack(weights, dim=1),
        }


class LMSCNetS0(FinalLMSCNet):
    """S0 single-scale model for one pre-registered kernel."""

    def __init__(
        self,
        *,
        kernel: int,
        num_classes: int = 11,
        dropout: float = 0.2,
        expansion: float = 1.25,
    ) -> None:
        super().__init__(
            kernels=(kernel,),
            fusion_mode="equal",
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
        )


class LMSCNetS1(FinalLMSCNet):
    """S1 fixed-equal-weight multi-scale model."""

    def __init__(
        self, *, num_classes: int = 11, dropout: float = 0.2, expansion: float = 1.25
    ) -> None:
        super().__init__(
            kernels=FINAL_LMSCNET_KERNELS,
            fusion_mode="equal",
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
        )


class LMSCNetS2(FinalLMSCNet):
    """S2 content-adaptive multi-scale model using only pooled branch features."""

    def __init__(
        self, *, num_classes: int = 11, dropout: float = 0.2, expansion: float = 1.25
    ) -> None:
        super().__init__(
            kernels=FINAL_LMSCNET_KERNELS,
            fusion_mode="content_adaptive",
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
        )


class LMSCNetS1Static(FinalLMSCNet):
    """S1-static: global trainable branch logits, independent of sample content."""

    def __init__(
        self, *, num_classes: int = 11, dropout: float = 0.2, expansion: float = 1.25
    ) -> None:
        super().__init__(
            kernels=FINAL_LMSCNET_KERNELS,
            fusion_mode="learned_static",
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
        )


class LMSCNetS1WideStatic(LMSCNetS1Static):
    """Parameter-matched static control with a pre-registered wider expansion."""

    PRE_REGISTERED_EXPANSION = 1.8

    def __init__(
        self, *, num_classes: int = 11, dropout: float = 0.2, expansion: float | None = None
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            dropout=dropout,
            expansion=self.PRE_REGISTERED_EXPANSION if expansion is None else expansion,
        )


class LMSCNetS2Mean(LMSCNetS2):
    """S2 inference with a train-only, fixed mean gate."""

    def __init__(self, *, num_classes: int = 11, dropout: float = 0.2, expansion: float = 1.25) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout, expansion=expansion)
        self.register_buffer("mean_gate", torch.full((NUM_FINAL_BLOCKS, 3), 1.0 / 3.0))
        self.mean_gate_fitted = False

    @torch.no_grad()
    def fit_mean_gate(self, train_batches: object) -> None:
        totals = torch.zeros_like(self.mean_gate)
        count = 0
        was_training = self.training
        self.eval()
        for batch in train_batches:  # type: ignore[union-attr]
            if isinstance(batch, dict):
                iq = batch["iq"]
            elif isinstance(batch, (tuple, list)):
                iq = batch[0]
            else:
                raise ValueError("fit_mean_gate batches must be dicts or sequences")
            weights = super().forward(iq)["scale_weights"]
            totals += weights.sum(dim=0).to(self.mean_gate)
            count += int(weights.shape[0])
        if was_training:
            self.train()
        if count <= 0:
            raise ValueError("fit_mean_gate requires at least one train batch")
        self.mean_gate.copy_(totals / count)
        self.mean_gate_fitted = True

    def forward(self, iq: torch.Tensor, gate_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        override = self.mean_gate.unsqueeze(0).expand(iq.shape[0], -1, -1)
        return super().forward(iq, override if gate_override is None else gate_override)


def shuffled_gate_weights(weights: torch.Tensor, seed: int) -> tuple[torch.Tensor, str]:
    """Shuffle sample-to-gate correspondence while preserving gate marginals."""
    if weights.ndim != 3 or weights.shape[-1] != 3:
        raise ValueError("weights must have shape [batch, blocks, 3]")
    generator = torch.Generator(device=weights.device).manual_seed(int(seed))
    permutation = torch.randperm(weights.shape[0], generator=generator, device=weights.device)
    digest = sha256(permutation.detach().cpu().numpy().tobytes()).hexdigest()
    return weights.index_select(0, permutation), digest


class LMSCNetS2Shuffled(LMSCNetS2):
    """S2 inference with deterministic batch-local gate shuffling."""

    def __init__(
        self, *, permutation_seed: int = 13, num_classes: int = 11,
        dropout: float = 0.2, expansion: float = 1.25
    ) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout, expansion=expansion)
        self.permutation_seed = int(permutation_seed)
        self.last_permutation_hash = ""

    def forward(self, iq: torch.Tensor, gate_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        aligned = super().forward(iq, gate_override)
        shuffled, digest = shuffled_gate_weights(aligned["scale_weights"], self.permutation_seed)
        self.last_permutation_hash = digest
        return super().forward(iq, shuffled)


class _ChannelGateBlock(nn.Module):
    """Channel-wise attention gate block used by the neighbor adaptations.

    Source-informed adaptation of SKNet (Li et al., CVPR 2019): split into
    parallel branch convolutions, fuse via summation + global average pooling
    + bottleneck FC, then assign a per-channel softmax weight vector over the
    branches. The weights depend on the sample content, unlike the scalar
    branch gate of S2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        kernels: tuple[int, ...],
        expansion: float,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        width = math.ceil(expansion * out_channels)
        self.kernels = kernels
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
        gate_hidden = max(len(kernels), width // reduction)
        self.gate_down = nn.Linear(width, gate_hidden)
        self.gate_up = nn.Linear(gate_hidden, width * len(kernels))
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
        fused_sum = torch.stack(branches, dim=0).sum(dim=0)
        pooled = fused_sum.mean(dim=-1)
        gate = torch.nn.functional.silu(self.gate_down(pooled))
        logits = self.gate_up(gate).view(projected.shape[0], -1, len(self.kernels))
        weights = torch.softmax(logits, dim=-1)
        fused = sum(
            branch * weights[:, :, index].unsqueeze(-1)
            for index, branch in enumerate(branches)
        )
        output = torch.nn.functional.silu(self.fusion(fused) + self.shortcut(x))
        return output, weights.mean(dim=1)


class _AFNetFusionBlock(nn.Module):
    """Adaptive fusion block with lambda-softmax channel gates.

    Source-informed adaptation of AFNet (Shi et al., VTC2022-Spring): dual
    branch convolutions of kernel (3, 5), a first lambda-softmax fusion module
    with lambda=1 combining the two branches, and a second fusion module with
    lambda=2 acting as an optimized skip connection. All gates are per-channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        kernels: tuple[int, ...] = (3, 5),
        expansion: float,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        width = math.ceil(expansion * out_channels)
        self.kernels = kernels
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
        gate_hidden = max(len(kernels), width // reduction)
        self.fusion1_down = nn.Linear(width, gate_hidden)
        self.fusion1_a = nn.Linear(gate_hidden, width)
        self.fusion1_b = nn.Linear(gate_hidden, width)
        self.fusion1_scale = 1.0
        self.fusion2_down = nn.Linear(out_channels, max(len(kernels), out_channels // reduction))
        self.fusion2_a = nn.Linear(max(len(kernels), out_channels // reduction), out_channels)
        self.fusion2_b = nn.Linear(max(len(kernels), out_channels // reduction), out_channels)
        self.fusion2_scale = 2.0
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

    @staticmethod
    def _lambda_softmax(
        vector_a: torch.Tensor, vector_b: torch.Tensor, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([vector_a, vector_b], dim=1)
        probability = torch.softmax(stacked, dim=1)
        return scale * probability[:, 0], scale * probability[:, 1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(x)
        branches = [branch(projected) for branch in self.branches]
        branch_a, branch_b = branches[0], branches[1]
        pooled = (branch_a + branch_b).mean(dim=-1)
        gate = torch.nn.functional.relu(self.fusion1_down(pooled))
        alpha, beta = self._lambda_softmax(
            self.fusion1_a(gate), self.fusion1_b(gate), self.fusion1_scale
        )
        fused = alpha.unsqueeze(-1) * branch_a + beta.unsqueeze(-1) * branch_b
        fused_out = self.fusion(fused)
        shortcut = self.shortcut(x)
        pooled2 = (fused_out + shortcut).mean(dim=-1)
        gate2 = torch.nn.functional.relu(self.fusion2_down(pooled2))
        gamma, delta = self._lambda_softmax(
            self.fusion2_a(gate2), self.fusion2_b(gate2), self.fusion2_scale
        )
        output = torch.nn.functional.silu(
            gamma.unsqueeze(-1) * fused_out + delta.unsqueeze(-1) * shortcut
        )
        weights = torch.stack([alpha, beta], dim=1).mean(dim=2)
        return output, weights


def _channel_backbone(
    *,
    block_factory: type[_ChannelGateBlock] | type[_AFNetFusionBlock],
    num_classes: int,
    dropout: float,
    expansion: float,
    kernels: tuple[int, ...],
) -> tuple[nn.Module, nn.Module, list[nn.Module]]:
    """Build the shared stem, classifier and staged channel-gate blocks."""
    _validate_common(num_classes, dropout, expansion)
    stem = nn.Sequential(
        nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(32),
        nn.SiLU(inplace=True),
    )
    classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, num_classes))

    def block(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        return block_factory(
            in_channels,
            out_channels,
            stride,
            kernels=kernels,
            expansion=expansion,
        )

    stages = [
        nn.ModuleList([block(32, 32, 1), block(32, 32, 1)]),
        nn.ModuleList([block(32, 64, 2), block(64, 64, 1)]),
        nn.ModuleList([block(64, 96, 2), block(96, 96, 1)]),
    ]
    return stem, classifier, stages


class _ChannelGateNetwork(nn.Module):
    """Shared forward contract for the channel-gate neighbor adaptations."""

    def __init__(
        self,
        *,
        block_factory: type[_ChannelGateBlock] | type[_AFNetFusionBlock],
        num_classes: int = 11,
        dropout: float = 0.2,
        expansion: float = 1.25,
        kernels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.stem, self.classifier, raw_stages = _channel_backbone(
            block_factory=block_factory,
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
            kernels=kernels,
        )
        self.stages = nn.ModuleList(raw_stages)

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_input(iq)
        features = self.stem(iq)
        weights: list[torch.Tensor] = []
        for stage in self.stages:
            for block in stage:
                features, block_weights = block(features)
                weights.append(block_weights)
        return {
            "logits": self.classifier(features.mean(dim=-1)),
            "scale_weights": torch.stack(weights, dim=1),
        }


class SKNet1DAdaptation(_ChannelGateNetwork):
    """1-D selective-kernel adaptation; channel-wise, input-dependent weights.

    Source-informed adaptation of SKNet (Li et al., CVPR 2019). Not a
    reproduction claim: split/preprocessing/training protocol differ.
    """

    def __init__(
        self, *, num_classes: int = 11, dropout: float = 0.2, expansion: float = 1.25
    ) -> None:
        super().__init__(
            block_factory=_ChannelGateBlock,
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
            kernels=(3, 7, 15),
        )


class AFNetAdaptation(_ChannelGateNetwork):
    """1-D adaptive fusion adaptation; lambda-softmax channel gates.

    Source-informed adaptation of AFNet (Shi et al., VTC2022-Spring). Not a
    reproduction claim: 2-D inputs become 1-D I/Q, the original nine AF units
    are mapped onto the shared six-block backbone, and lambda=1/lambda=2
    fusion modules are preserved per block.
    """

    def __init__(
        self, *, num_classes: int = 11, dropout: float = 0.2, expansion: float = 1.25
    ) -> None:
        super().__init__(
            block_factory=_AFNetFusionBlock,
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
            kernels=(3, 5),
        )


__all__ = [
    "FINAL_LMSCNET_KERNELS",
    "FinalLMSCNet",
    "LMSCNetS0",
    "LMSCNetS1",
    "LMSCNetS2",
    "LMSCNetS1Static",
    "LMSCNetS1WideStatic",
    "LMSCNetS2Mean",
    "LMSCNetS2Shuffled",
    "SKNet1DAdaptation",
    "AFNetAdaptation",
    "NUM_FINAL_BLOCKS",
    "shuffled_gate_weights",
]
