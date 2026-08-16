from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from na_lmscnet.evaluation.na_lmscnet_report import (
    ALL_SNRS,
    EXPECTED_SEEDS,
    KERNELS,
    NALMSCNetReportError,
    _aggregate_metric,
    _aggregate_scale_weights,
    _plot_confusion,
    _plot_scale_weights,
    _plot_snr,
)


def test_aggregate_metric_uses_sample_standard_deviation() -> None:
    mean, std = _aggregate_metric([{"value": 1.0}, {"value": 3.0}], "value")

    assert mean == 2.0
    assert std == pytest.approx(math.sqrt(2.0))


def test_aggregate_scale_weights_covers_all_snrs_blocks_and_kernels() -> None:
    rows = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for snr in ALL_SNRS:
            for block in range(1, 7):
                for kernel in KERNELS:
                    rows.append(
                        {
                            "seed": seed,
                            "snr_db": snr,
                            "block": block,
                            "kernel": kernel,
                            "mean_weight": 0.1 * seed_index,
                            "sample_count": 1100,
                        }
                    )

    result = _aggregate_scale_weights(rows)

    assert len(result) == len(ALL_SNRS) * 6 * len(KERNELS)
    assert result[0]["mean_weight"] == pytest.approx(0.2)
    assert result[0]["std_across_seeds"] == pytest.approx(math.sqrt(0.025))
    assert result[0]["samples_per_seed"] == 1100


def test_aggregate_scale_weights_rejects_missing_seed() -> None:
    with pytest.raises(NALMSCNetReportError, match="incomplete"):
        _aggregate_scale_weights(
            [
                {
                    "seed": 13,
                    "snr_db": -20,
                    "block": 1,
                    "kernel": 3,
                    "mean_weight": 0.3,
                    "sample_count": 1100,
                }
            ]
        )


def test_pillow_report_plots_are_valid_pngs(tmp_path) -> None:
    snr_rows = [
        {"snr_db": snr, "mean_accuracy": (snr + 22) / 42, "std_accuracy": 0.01} for snr in ALL_SNRS
    ]
    scale_rows = [
        {"snr_db": snr, "block": block, "kernel": kernel, "mean_weight": 1 / 3}
        for snr in ALL_SNRS
        for block in (1, 6)
        for kernel in KERNELS
    ]
    outputs = [tmp_path / "snr.png", tmp_path / "scale.png", tmp_path / "confusion.png"]

    _plot_snr(outputs[0], snr_rows)
    _plot_scale_weights(outputs[1], scale_rows)
    _plot_confusion(outputs[2], np.eye(11, dtype=np.int64), "Validation confusion")

    for output in outputs:
        assert output.stat().st_size > 1_000
        with Image.open(output) as image:
            image.verify()
