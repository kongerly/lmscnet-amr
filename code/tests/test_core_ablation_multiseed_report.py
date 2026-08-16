from __future__ import annotations

import numpy as np
import pytest

from na_lmscnet.evaluation.core_ablation_multiseed_report import (
    EXPECTED_SEEDS,
    formal_contribution_decision,
    paired_hierarchical_bootstrap,
)


def _replays(variant_predictions: np.ndarray) -> tuple[dict[int, dict], dict[int, dict]]:
    sample_ids = np.asarray(["a", "b", "c", "d"], dtype=object)
    modulation = np.asarray([0, 0, 1, 1], dtype=np.int64)
    snr_db = np.asarray([-10, 0, -10, 0], dtype=np.int64)
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    reference_predictions = targets.copy()
    reference = {}
    variant = {}
    for seed in EXPECTED_SEEDS:
        shared = {
            "sample_ids": sample_ids,
            "targets": targets,
            "modulation": modulation,
            "snr_db": snr_db,
        }
        reference[seed] = {**shared, "predictions": reference_predictions}
        variant[seed] = {**shared, "predictions": variant_predictions}
    return reference, variant


def test_paired_bootstrap_preserves_strata_and_low_snr_scope() -> None:
    reference, variant = _replays(np.asarray([1, 1, 0, 1], dtype=np.int64))

    overall = paired_hierarchical_bootstrap(
        reference_replays=reference,
        variant_replays=variant,
        metric="accuracy",
        seed=2026,
        resamples=100,
        num_classes=2,
    )
    low_snr = paired_hierarchical_bootstrap(
        reference_replays=reference,
        variant_replays=variant,
        metric="accuracy",
        snr_values=(-10,),
        seed=2026,
        resamples=100,
        num_classes=2,
    )

    assert overall["mean_difference"] == pytest.approx(0.25)
    assert overall["ci_lower"] == pytest.approx(0.25)
    assert low_snr["mean_difference"] == pytest.approx(0.5)
    assert low_snr["snr_values"] == [-10]


def test_paired_bootstrap_rejects_sample_misalignment() -> None:
    reference, variant = _replays(np.asarray([1, 1, 0, 1], dtype=np.int64))
    variant[13] = {**variant[13], "sample_ids": np.asarray(["b", "a", "c", "d"])}

    with pytest.raises(ValueError, match="alignment"):
        paired_hierarchical_bootstrap(
            reference_replays=reference,
            variant_replays=variant,
            metric="accuracy",
            resamples=10,
        )


def test_formal_contribution_requires_all_three_frozen_conditions() -> None:
    positive = {"ci_lower": 0.001, "mean_difference": 0.01}
    crossing = {"ci_lower": -0.001, "mean_difference": 0.01}

    accepted = formal_contribution_decision(
        low_snr_ci=positive,
        accuracy_ci=positive,
        macro_f1_ci=crossing,
        positive_low_snr_seed_count=4,
    )
    rejected = formal_contribution_decision(
        low_snr_ci=positive,
        accuracy_ci=positive,
        macro_f1_ci=positive,
        positive_low_snr_seed_count=3,
    )

    assert accepted["qualifies"] is True
    assert accepted["status"] == "stable_independent_contribution"
    assert rejected["qualifies"] is False
    assert rejected["status"] == "evidence_insufficient"
