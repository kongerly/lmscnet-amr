from __future__ import annotations

import pytest
import torch
from torch import nn

from na_lmscnet.evaluation.baseline_report import (
    BaselineReportError,
    _aggregate_efficiency,
    _best_validation,
    _count_macs,
)


def _validation_metrics() -> dict[str, object]:
    per_snr = {f"{snr:+d}": 0.5 for snr in range(-20, 20, 2)}
    return {
        "best_epoch": 1,
        "history": [
            {
                "validation_loss": 1.0,
                "validation": {
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "per_snr_accuracy": per_snr,
                },
            }
        ],
    }


def test_best_validation_requires_all_canonical_snr_values() -> None:
    metrics = _validation_metrics()
    result = _best_validation(metrics, "fixture")

    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == 0.4
    assert set(result["per_snr_accuracy"]) == set(range(-20, 20, 2))

    del metrics["history"][0]["validation"]["per_snr_accuracy"]["+0"]
    with pytest.raises(BaselineReportError, match="incomplete"):
        _best_validation(metrics, "fixture")


def test_best_validation_rejects_noncanonical_snr_key() -> None:
    metrics = _validation_metrics()
    per_snr = metrics["history"][0]["validation"]["per_snr_accuracy"]
    per_snr["0"] = per_snr.pop("+0")

    with pytest.raises(BaselineReportError, match="incomplete"):
        _best_validation(metrics, "fixture")


def test_count_macs_includes_conv_linear_and_lstm() -> None:
    class FixtureModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 4, kernel_size=3)
            self.lstm = nn.LSTM(input_size=4, hidden_size=3, batch_first=True)
            self.linear = nn.Linear(3, 2)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = self.conv(value).transpose(1, 2)
            value, _ = self.lstm(value)
            return self.linear(value[:, -1])

    model = FixtureModel()
    macs = _count_macs(model, (1, 2, 8), torch.device("cpu"))

    assert macs == (1 * 4 * 6 * 2 * 3) + (1 * 6 * 1 * 4 * 3 * (4 + 3)) + (1 * 2 * 3)


def test_efficiency_aggregate_uses_single_measurement_without_nan() -> None:
    rows = _aggregate_efficiency(
        [
            {
                "model": "cnn2",
                "parameter_count": 10,
                "macs": 20,
                "checkpoint_size_bytes": 30,
                "gpu_latency_ms": 0.5,
                "gpu_throughput_samples_per_s": 2000.0,
                "cpu_latency_ms": 2.0,
            },
            {
                "model": "cldnn",
                "parameter_count": 11,
                "macs": 21,
                "checkpoint_size_bytes": 31,
                "gpu_latency_ms": 0.6,
                "gpu_throughput_samples_per_s": 1666.0,
                "cpu_latency_ms": 2.1,
            },
            {
                "model": "resnet1d",
                "parameter_count": 12,
                "macs": 22,
                "checkpoint_size_bytes": 32,
                "gpu_latency_ms": 0.7,
                "gpu_throughput_samples_per_s": 1428.0,
                "cpu_latency_ms": 2.2,
            },
        ]
    )

    assert len(rows) == 3
    assert all(row["gpu_latency_ms_std"] == 0.0 for row in rows)
    assert all(row["cpu_latency_ms_std"] == 0.0 for row in rows)
