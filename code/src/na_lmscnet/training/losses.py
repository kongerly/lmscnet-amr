"""Losses for noise-aware modulation recognition."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

NA_LMSCNET_SNR_LOSS_WEIGHT = 0.1


class NoiseAwareJointLoss(nn.Module):
    """Cross-entropy plus normalized SNR regression loss."""

    def __init__(self, *, snr_weight: float = NA_LMSCNET_SNR_LOSS_WEIGHT) -> None:
        super().__init__()
        if type(snr_weight) is not float or not 0.0 <= snr_weight <= 1.0:
            raise ValueError("snr_weight must be a float in [0, 1]")
        self.snr_weight = snr_weight

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        modulation: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if set(outputs) != {"logits", "snr_hat", "scale_weights"}:
            raise ValueError("NA-LMSCNet outputs must contain logits, snr_hat, and scale_weights")
        logits = outputs["logits"]
        snr_hat = outputs["snr_hat"]
        if logits.ndim != 2 or modulation.ndim != 1 or logits.shape[0] != modulation.shape[0]:
            raise ValueError("Classification loss shapes are invalid")
        if snr_hat.ndim != 1 or snr_hat.shape[0] != modulation.shape[0]:
            raise ValueError("SNR loss shapes are invalid")
        target_snr_norm = (snr_db.to(dtype=snr_hat.dtype) + 1.0) / 19.0
        predicted_snr_norm = (snr_hat + 1.0) / 19.0
        classification = nn.functional.cross_entropy(logits, modulation)
        snr = nn.functional.smooth_l1_loss(predicted_snr_norm, target_snr_norm)
        return classification + self.snr_weight * snr, classification, snr


__all__ = ["NA_LMSCNET_SNR_LOSS_WEIGHT", "NoiseAwareJointLoss"]
