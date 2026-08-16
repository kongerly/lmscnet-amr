"""Explicit I/Q preprocessing protocols for controlled AMR experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

from na_lmscnet.data.contracts import ModulationSample

PreprocessingMode = Literal[
    "raw",
    "per_sample_dc_power",
    "global_zscore",
    "per_sample_max_abs",
]
PREPROCESSING_MODES: tuple[PreprocessingMode, ...] = (
    "raw",
    "per_sample_dc_power",
    "global_zscore",
    "per_sample_max_abs",
)
DEFAULT_PREPROCESSING_MODE: PreprocessingMode = "per_sample_max_abs"


class PreprocessingError(ValueError):
    """Raised when an I/Q preprocessing protocol is invalid."""


@dataclass(frozen=True)
class GlobalZScoreStatistics:
    """Population statistics computed only from the frozen training split."""

    channel_mean: tuple[float, float]
    channel_std: tuple[float, float]
    scalar_count_per_channel: int
    split: str = "train"
    estimator: str = "population"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _validate_iq(iq: torch.Tensor) -> None:
    if iq.ndim != 2 or iq.shape[0] != 2 or not iq.dtype.is_floating_point:
        raise PreprocessingError("iq must be a floating tensor with shape [2, length]")
    if not bool(torch.isfinite(iq).all()):
        raise PreprocessingError("iq must contain only finite values")


def preprocess_iq(
    iq: torch.Tensor,
    *,
    mode: PreprocessingMode = DEFAULT_PREPROCESSING_MODE,
    global_zscore: GlobalZScoreStatistics | None = None,
) -> torch.Tensor:
    """Apply one explicitly named preprocessing protocol to one I/Q window."""

    _validate_iq(iq)
    if mode not in PREPROCESSING_MODES:
        raise PreprocessingError(f"Unsupported preprocessing mode: {mode}")
    if mode == "raw":
        return iq
    if mode == "per_sample_dc_power":
        centered = iq - iq.mean(dim=1, keepdim=True)
        mean_power = centered.square().sum(dim=0).mean()
        if not bool(torch.isfinite(mean_power)) or float(mean_power) <= 0.0:
            raise PreprocessingError("iq must have positive finite power after DC removal")
        return centered / mean_power.sqrt()
    if mode == "per_sample_max_abs":
        maximum = iq.square().sum(dim=0).sqrt().max()
        if not bool(torch.isfinite(maximum)) or float(maximum) <= 0.0:
            raise PreprocessingError("iq must have positive finite maximum complex amplitude")
        return iq / maximum
    if global_zscore is None:
        raise PreprocessingError("global_zscore mode requires frozen training statistics")
    mean = iq.new_tensor(global_zscore.channel_mean).view(2, 1)
    std = iq.new_tensor(global_zscore.channel_std).view(2, 1)
    if bool((std <= 0).any()) or not bool(torch.isfinite(std).all()):
        raise PreprocessingError("global z-score standard deviations must be positive and finite")
    return (iq - mean) / std


def compute_global_zscore_statistics(
    dataset: Dataset[ModulationSample],
    *,
    batch_size: int = 4096,
    num_workers: int = 0,
) -> GlobalZScoreStatistics:
    """Compute per-I/Q-channel population statistics from a raw training dataset."""

    if batch_size < 1 or num_workers < 0:
        raise PreprocessingError("batch_size must be positive and num_workers non-negative")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    total = torch.zeros(2, dtype=torch.float64)
    total_square = torch.zeros(2, dtype=torch.float64)
    count = 0
    for batch in loader:
        iq = batch["iq"].to(dtype=torch.float64)
        if iq.ndim != 3 or iq.shape[1] != 2 or not bool(torch.isfinite(iq).all()):
            raise PreprocessingError("statistics dataset must yield finite [batch,2,length] I/Q")
        total += iq.sum(dim=(0, 2))
        total_square += iq.square().sum(dim=(0, 2))
        count += iq.shape[0] * iq.shape[2]
    if count == 0:
        raise PreprocessingError("statistics dataset is empty")
    mean = total / count
    variance = total_square / count - mean.square()
    if bool((variance <= 0).any()) or not bool(torch.isfinite(variance).all()):
        raise PreprocessingError("training statistics have non-positive variance")
    std = variance.sqrt()
    return GlobalZScoreStatistics(
        channel_mean=(float(mean[0]), float(mean[1])),
        channel_std=(float(std[0]), float(std[1])),
        scalar_count_per_channel=count,
    )
