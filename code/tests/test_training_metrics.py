from __future__ import annotations

import pytest
import torch

from na_lmscnet.training.metrics import classification_metrics


def test_classification_metrics_matches_known_confusion_matrix() -> None:
    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 0, 1, 1])
    snr = torch.tensor([-10, -10, 0, 0])

    metrics = classification_metrics(predictions, targets, snr, num_classes=2)

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(0.5)
    assert metrics.per_snr_accuracy == {"-10": 0.5, "+0": 0.5}
    assert metrics.sample_count == 4


def test_macro_f1_includes_missing_classes_as_zero() -> None:
    metrics = classification_metrics(
        torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]), num_classes=2
    )

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 0.5


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        classification_metrics(torch.tensor([]), torch.tensor([]), torch.tensor([]), num_classes=2)
    with pytest.raises(ValueError):
        classification_metrics(
            torch.tensor([2]), torch.tensor([0]), torch.tensor([0]), num_classes=2
        )


def test_metrics_reports_snr_mae_when_predictions_are_provided() -> None:
    metrics = classification_metrics(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([-10, 0]),
        num_classes=2,
        snr_prediction_db=torch.tensor([-8.0, -1.0]),
    )

    assert metrics.snr_mae_db == pytest.approx(1.5)
