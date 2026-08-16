from __future__ import annotations

import numpy as np
import pytest

from na_lmscnet.evaluation.na_lmscnet_report import ALL_SNRS
from na_lmscnet.evaluation.snr_auxiliary_ablation_report import (
    _snr_hat_distribution,
    screening_decision,
)


def test_snr_hat_distribution_records_quantiles_for_each_true_snr() -> None:
    true_snr = np.repeat(np.asarray(ALL_SNRS), 5)
    offsets = np.tile(np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]), len(ALL_SNRS))
    predictions = true_snr + offsets

    rows = _snr_hat_distribution(true_snr, predictions)

    assert len(rows) == len(ALL_SNRS)
    assert rows[0]["sample_count"] == 5
    assert rows[0]["mean_snr_hat_db"] == pytest.approx(-20.0)
    assert rows[0]["median_snr_hat_db"] == pytest.approx(-20.0)
    assert rows[0]["mae_db"] == pytest.approx(1.2)


def test_screening_decision_runs_five_seeds_for_clear_drop() -> None:
    result = screening_decision(
        reference_accuracy=0.56,
        reference_macro_f1=0.60,
        reference_low_snr=0.51,
        ablation_accuracy=0.55,
        ablation_macro_f1=0.59,
        ablation_low_snr=0.49,
    )

    assert result["action"] == "run_five_seed_formal_validation"
    assert result["reason"] == "clear_seed13_drop"


def test_screening_decision_narrows_claim_for_unchanged_result() -> None:
    result = screening_decision(
        reference_accuracy=0.56,
        reference_macro_f1=0.60,
        reference_low_snr=0.51,
        ablation_accuracy=0.558,
        ablation_macro_f1=0.598,
        ablation_low_snr=0.508,
    )

    assert result["action"] == "narrow_to_lightweight_multiscale_dynamic_fusion"
    assert result["reason"] == "seed13_basically_unchanged"
