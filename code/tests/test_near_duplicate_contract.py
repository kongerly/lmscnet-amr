from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from na_lmscnet.data.near_duplicate_contract import (
    MAX_NEAR_DUPLICATE_CONTRACT_BYTES,
    NearDuplicateContractError,
    load_near_duplicate_contract,
    max_abs_normalized_circular_correlation,
    near_duplicate_contract_sha256,
    power_normalized_complex,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
CONTRACT_PATH = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_near_duplicate.yml"


def repository_contract() -> dict[str, object]:
    return load_near_duplicate_contract(CONTRACT_PATH, DATASET_SPEC, CONVERSION_CONTRACT)


def write_contract(path: Path, contract: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def test_repository_contract_is_bound_and_generation_gated() -> None:
    contract = repository_contract()

    assert contract["contract_id"] == "radioml_2016_10a_near_duplicate_v1"
    assert contract["candidate_generation"]["status"] == "reference_only"
    assert contract["threshold_calibration"]["status"] == "pending"
    assert contract["review"]["status"] == "pending"
    assert contract["generation_gate"] == {
        "audit_generation_enabled": False,
        "split_generation_enabled": True,
        "audit_blocked_by": [
            "candidate_recall_evidence",
            "threshold_calibration",
            "manual_review_protocol",
        ],
    }
    assert len(near_duplicate_contract_sha256(CONTRACT_PATH)) == 64


def test_power_normalization_returns_read_only_unit_rms() -> None:
    iq = np.asarray([[3.0, 0.0, -3.0], [0.0, 4.0, 0.0]], dtype=np.float32)

    result = power_normalized_complex(iq)

    assert result.flags.writeable is False
    assert np.mean(np.abs(result) ** 2) == pytest.approx(1.0)
    assert result.dtype == np.dtype(np.complex128)


def test_reference_score_is_invariant_to_complex_gain_and_circular_shift() -> None:
    iq = np.asarray([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0]], dtype=np.float32)
    complex_gain = 2.5 * np.exp(1j * 0.73)
    complex_signal = iq[0] + 1j * iq[1]
    transformed = np.roll(complex_signal * complex_gain, 2)
    transformed_iq = np.vstack([transformed.real, transformed.imag]).astype(np.float32)

    assert max_abs_normalized_circular_correlation(iq, transformed_iq) == pytest.approx(
        1.0, abs=1e-12
    )


def test_reference_score_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        power_normalized_complex(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        power_normalized_complex(np.asarray([[np.nan], [0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        power_normalized_complex(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="same shape"):
        max_abs_normalized_circular_correlation(
            np.ones((2, 4), dtype=np.float32), np.ones((2, 5), dtype=np.float32)
        )


def mutated_contract(tmp_path: Path, mutation: str) -> Path:
    contract = deepcopy(repository_contract())
    if mutation == "extra_field":
        contract["unexpected"] = True
    elif mutation == "source_hash":
        contract["source"]["source_archive_sha256"] = "0" * 64
    elif mutation == "hdf5_logical_hash":
        contract["source"]["hdf5"]["logical_content_sha256"] = "0" * 64
    elif mutation == "representation":
        contract["representations"]["transformed_similarity"]["mean_removal"] = True
    elif mutation == "lag_domain":
        contract["representations"]["transformed_similarity"]["similarity"]["lag_domain"] = [0, 126]
    elif mutation == "candidate_enabled":
        contract["candidate_generation"]["production_enabled"] = True
    elif mutation == "candidate_recall":
        contract["candidate_generation"]["candidate_recall_requirement"] = 0.99
    elif mutation == "candidate_recall_type":
        contract["candidate_generation"]["candidate_recall_requirement"] = 1
    elif mutation == "threshold_status":
        contract["threshold_calibration"]["status"] = "calibrated"
    elif mutation == "threshold_value":
        contract["threshold_calibration"]["threshold"] = 0.999
    elif mutation == "positive_cases":
        contract["threshold_calibration"]["positive_fixture"]["minimum_cases"] = 127
    elif mutation == "manual_review":
        contract["review"]["ambiguous_policy"] = "accept"
    elif mutation == "absolute_paths":
        contract["report"]["absolute_paths"] = True
    elif mutation == "report_version_type":
        contract["report"]["schema_version"] = True
    elif mutation == "publication_bool_type":
        contract["publication"]["overwrite"] = 0
    elif mutation == "audit_enabled":
        contract["generation_gate"]["audit_generation_enabled"] = True
    elif mutation == "audit_enabled_type":
        contract["generation_gate"]["audit_generation_enabled"] = 0
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    output = tmp_path / "near-duplicate.yml"
    write_contract(output, contract)
    return output


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "source_hash",
        "hdf5_logical_hash",
        "representation",
        "lag_domain",
        "candidate_enabled",
        "candidate_recall",
        "candidate_recall_type",
        "threshold_status",
        "threshold_value",
        "positive_cases",
        "manual_review",
        "absolute_paths",
        "report_version_type",
        "publication_bool_type",
        "audit_enabled",
        "audit_enabled_type",
    ],
)
def test_rejects_contract_mutations(tmp_path: Path, mutation: str) -> None:
    with pytest.raises((NearDuplicateContractError, ValueError)):
        load_near_duplicate_contract(
            mutated_contract(tmp_path, mutation), DATASET_SPEC, CONVERSION_CONTRACT
        )


def test_rejects_oversized_contract(tmp_path: Path) -> None:
    path = tmp_path / "near-duplicate.yml"
    path.write_bytes(b"#" * (MAX_NEAR_DUPLICATE_CONTRACT_BYTES + 1))

    with pytest.raises(NearDuplicateContractError, match="exceeds"):
        load_near_duplicate_contract(path, DATASET_SPEC, CONVERSION_CONTRACT)


def test_rejects_contract_symlink(tmp_path: Path) -> None:
    path = tmp_path / "near-duplicate.yml"
    path.write_text("schema_version: 1\n", encoding="utf-8")

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(NearDuplicateContractError, match="symlinks"),
    ):
        load_near_duplicate_contract(path, DATASET_SPEC, CONVERSION_CONTRACT)


def test_near_duplicate_contract_cli_reports_pending_gate() -> None:
    result = subprocess.run(
        [sys.executable, "code/scripts/validate_near_duplicate_contract.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", errors="replace")
    summary = json.loads(result.stdout)
    assert summary["contract_id"] == "radioml_2016_10a_near_duplicate_v1"
    assert summary["split_generation_enabled"] is True
    assert summary["threshold_calibration"] == "pending"
