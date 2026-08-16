from __future__ import annotations

from pathlib import Path

import pytest

from na_lmscnet.evaluation.core_ablation_report import (
    FIXED_AVERAGE_MODEL,
    WO_MULTI_SCALE_MODEL,
    core_screening_decision,
    validate_split_audit_pair,
    variant_screening_decision,
)


def test_variant_screening_requests_formal_validation_for_clear_drop() -> None:
    decision = variant_screening_decision(
        reference_accuracy=0.56,
        reference_macro_f1=0.60,
        reference_low_snr=0.51,
        ablation_accuracy=0.55,
        ablation_macro_f1=0.59,
        ablation_low_snr=0.49,
    )

    assert decision["action"] == "run_five_seed_formal_validation"
    assert decision["reason"] == "clear_seed13_drop"


def test_core_screening_tracks_only_variants_requiring_formal_validation() -> None:
    formal = {"action": "run_five_seed_formal_validation", "reason": "clear_seed13_drop"}
    unchanged = {
        "action": "do_not_claim_independent_gain_from_seed13",
        "reason": "seed13_basically_unchanged",
    }

    decision = core_screening_decision(formal, unchanged)

    assert decision["formal_validation_variants"] == [WO_MULTI_SCALE_MODEL]
    assert decision["provisional_scope"] == (
        "lightweight_multiscale_fusion_pending_formal_validation"
    )
    assert decision["final_claim_authorized"] is False


def test_core_screening_keeps_both_variants_when_both_are_unresolved() -> None:
    formal = {"action": "run_five_seed_formal_validation", "reason": "seed13_borderline"}

    decision = core_screening_decision(formal, formal)

    assert decision["formal_validation_variants"] == [
        WO_MULTI_SCALE_MODEL,
        FIXED_AVERAGE_MODEL,
    ]
    assert decision["final_claim_authorized"] is False


def test_report_cli_rejects_mixed_generation_split_artifacts(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    audit_dir = tmp_path / "audit"
    split_dir.mkdir()
    audit_dir.mkdir()
    split = split_dir / "RML2016.10a.split-manifest.json"
    audit = audit_dir / "RML2016.10a.leakage-audit.json"
    split.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="same frozen artifact directory"):
        validate_split_audit_pair(split, audit)
