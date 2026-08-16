"""Deterministic classification metrics for AMR validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_snr_accuracy: dict[str, float]
    sample_count: int
    snr_mae_db: float | None = None


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    num_classes: int,
    snr_prediction_db: torch.Tensor | None = None,
) -> ClassificationMetrics:
    """Compute accuracy, macro F1, and per-SNR accuracy without test-set access."""

    if predictions.ndim != 1 or targets.ndim != 1 or snr_db.ndim != 1:
        raise ValueError("predictions, targets, and snr_db must be one-dimensional")
    if not (len(predictions) == len(targets) == len(snr_db)) or len(targets) == 0:
        raise ValueError("metric inputs must have equal nonzero lengths")
    if snr_prediction_db is not None and (
        snr_prediction_db.ndim != 1 or len(snr_prediction_db) != len(targets)
    ):
        raise ValueError("snr_prediction_db must match the other metric inputs")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    predictions = predictions.to(dtype=torch.int64, device="cpu")
    targets = targets.to(dtype=torch.int64, device="cpu")
    snr_db = snr_db.to(dtype=torch.int64, device="cpu")
    if bool(((predictions < 0) | (predictions >= num_classes)).any()):
        raise ValueError("predictions contain an invalid class index")
    if bool(((targets < 0) | (targets >= num_classes)).any()):
        raise ValueError("targets contain an invalid class index")

    flat = targets * num_classes + predictions
    confusion = torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    true_positive = confusion.diag().to(torch.float64)
    false_positive = confusion.sum(dim=0).to(torch.float64) - true_positive
    false_negative = confusion.sum(dim=1).to(torch.float64) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = torch.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    correct = predictions == targets
    per_snr = {}
    for value in sorted(int(item) for item in snr_db.unique().tolist()):
        mask = snr_db == value
        per_snr[f"{value:+d}"] = float(correct[mask].to(torch.float64).mean())
    snr_mae_db = None
    if snr_prediction_db is not None:
        snr_mae_db = float(
            (snr_prediction_db.to(dtype=torch.float64, device="cpu") - snr_db.to(dtype=torch.float64))
            .abs()
            .mean()
        )
    return ClassificationMetrics(
        accuracy=float(correct.to(torch.float64).mean()),
        macro_f1=float(f1.mean()),
        per_snr_accuracy=per_snr,
        sample_count=len(targets),
        snr_mae_db=snr_mae_db,
    )
