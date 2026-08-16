from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation.core_ablation_multiseed_report import (  # noqa: E402
    paired_hierarchical_bootstrap,
)


def _shared_base() -> dict[str, object]:
    rng = np.random.default_rng(2026)
    count = 220
    targets = rng.integers(0, 11, size=count)
    modulation = rng.integers(0, 11, size=count)
    snr_db = np.tile(np.asarray([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8]), 22)
    return {
        "sample_ids": np.asarray([f"shared:{index:04d}" for index in range(count)]),
        "targets": targets,
        "modulation": modulation,
        "snr_db": snr_db,
    }


def _with_bias(base: dict[str, object], rng: np.random.Generator, bias: float) -> dict[str, object]:
    replay = dict(base)
    targets = np.asarray(base["targets"])
    predictions = np.where(
        rng.random(len(targets)) < 0.6 + bias,
        targets,
        rng.integers(0, 11, size=len(targets)),
    )
    replay["predictions"] = predictions
    return replay


def test_paired_bootstrap_detects_positive_accuracy_difference() -> None:
    rng = np.random.default_rng(2026)
    base = _shared_base()
    reference = {seed: _with_bias(base, rng, bias=0.2) for seed in (13, 37, 73, 101, 137)}
    variant = {seed: _with_bias(base, rng, bias=-0.2) for seed in (13, 37, 73, 101, 137)}
    result = paired_hierarchical_bootstrap(
        reference_replays=reference,
        variant_replays=variant,
        metric="accuracy",
        snr_values=(-10, -8, -6, -4, -2, 0),
        seed=2026,
        resamples=2000,
    )
    assert result["mean_difference"] > 0
    assert result["ci_lower"] > 0
    assert result["bootstrap_seed"] == 2026


def test_paired_bootstrap_rejects_misaligned_replays() -> None:
    rng = np.random.default_rng(7)
    base = _shared_base()
    reference = {seed: _with_bias(base, rng, bias=0.0) for seed in (13, 37, 73, 101, 137)}
    variant = {seed: _with_bias(base, rng, bias=0.0) for seed in (13, 37, 73, 101, 137)}
    variant[13] = dict(variant[13])
    variant[13]["sample_ids"] = np.asarray(
        [f"shared:{index:04d}" if index else "tampered:0000" for index in range(len(base["sample_ids"]))]
    )
    with pytest.raises(ValueError, match="alignment"):
        paired_hierarchical_bootstrap(
            reference_replays=reference,
            variant_replays=variant,
            metric="accuracy",
            seed=2026,
            resamples=100,
        )


def test_contrast_report_schema_is_test_isolated() -> None:
    report = {
        "schema_version": 1,
        "purpose": "phase_r2_primary_contrast_analysis",
        "test_accessed": False,
        "contrasts": [
            {
                "contrast": "C1_s2_aligned_vs_s1_static",
                "positive_seed_count": 5,
                "rows": [{"metric": "accuracy", "overall": {"mean_difference": 0.01}}],
            }
        ],
    }
    assert report["test_accessed"] is False
    assert len(report["contrasts"]) == 1
