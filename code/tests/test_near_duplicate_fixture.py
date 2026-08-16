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

from na_lmscnet.data.near_duplicate_fixture import (
    MAX_NEAR_DUPLICATE_FIXTURE_CONTRACT_BYTES,
    NearDuplicateFixtureError,
    build_near_duplicate_fixture_evidence,
    deterministic_fixture_sample,
    load_near_duplicate_fixture_contract,
    near_duplicate_fixture_contract_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
NEAR_DUPLICATE_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_near_duplicate.yml"
FIXTURE_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_near_duplicate_fixture.yml"


def repository_fixture_contract() -> dict[str, object]:
    return load_near_duplicate_fixture_contract(
        FIXTURE_CONTRACT, NEAR_DUPLICATE_CONTRACT, DATASET_SPEC, CONVERSION_CONTRACT
    )


def write_fixture_contract(path: Path, contract: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def test_repository_fixture_contract_is_bound_and_generation_gated() -> None:
    contract = repository_fixture_contract()

    assert contract["fixture_contract_id"] == "radioml_2016_10a_near_duplicate_fixture_v1"
    assert contract["calibration_fixture"]["positive_cases"] == 128
    assert contract["calibration_fixture"]["negative_cases"] == 1024
    assert contract["bounded_reference_audit"] == {
        "sample_count": 64,
        "base_transform_pairs": 16,
        "unrelated_samples": 32,
        "algorithm": "exhaustive_pairwise_reference_v1",
        "expected_pair_count": 2016,
        "candidate_recall_required": 1.0,
        "production_claim": False,
    }
    assert contract["generation_gate"] == {
        "production_candidate_generation_enabled": False,
        "near_duplicate_audit_generation_enabled": False,
        "split_generation_enabled": True,
    }
    assert len(near_duplicate_fixture_contract_sha256(FIXTURE_CONTRACT)) == 64


def test_deterministic_fixture_sample_is_finite_read_only_float32() -> None:
    first = deterministic_fixture_sample("stable-label")
    second = deterministic_fixture_sample("stable-label")
    other = deterministic_fixture_sample("other-label")

    assert first.shape == (2, 128)
    assert first.dtype == np.dtype(np.float32)
    assert first.flags.writeable is False
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.isfinite(first).all()
    assert np.sqrt(np.mean(np.abs(first[0] + 1j * first[1]) ** 2)) > 0.0


def test_fixture_evidence_passes_without_writing_artifacts() -> None:
    evidence = build_near_duplicate_fixture_evidence(
        FIXTURE_CONTRACT, NEAR_DUPLICATE_CONTRACT, DATASET_SPEC, CONVERSION_CONTRACT
    )

    assert evidence["writes_artifacts"] is False
    assert evidence["calibration"]["passed"] is True
    assert evidence["calibration"]["positive_cases"] == 128
    assert evidence["calibration"]["negative_cases"] == 1024
    assert evidence["calibration"]["positive_recall"] == "1"
    assert evidence["calibration"]["negative_false_positive_rate"] == "0"
    assert evidence["bounded_reference_audit"]["passed"] is True
    assert evidence["bounded_reference_audit"]["sample_count"] == 64
    assert evidence["bounded_reference_audit"]["pair_count"] == 2016
    assert evidence["bounded_reference_audit"]["candidate_recall"] == "1"
    assert evidence["bounded_reference_audit"]["production_claim"] is False
    assert len(evidence["evidence_sha256"]) == 64


def mutated_fixture_contract(tmp_path: Path, mutation: str) -> Path:
    contract = deepcopy(repository_fixture_contract())
    if mutation == "near_contract_hash":
        contract["near_duplicate_contract"]["contract_sha256"] = "0" * 64
    elif mutation == "positive_cases":
        contract["calibration_fixture"]["positive_cases"] = 127
    elif mutation == "negative_seed":
        contract["calibration_fixture"]["negative_seed"] = 2027
    elif mutation == "threshold_rule":
        contract["threshold_selection"]["rule"] = "manual"
    elif mutation == "threshold_recall_type":
        contract["threshold_selection"]["positive_recall_required"] = 1
    elif mutation == "sample_count":
        contract["bounded_reference_audit"]["sample_count"] = 65
    elif mutation == "production_claim":
        contract["bounded_reference_audit"]["production_claim"] = True
    elif mutation == "writes_artifacts":
        contract["publication"]["writes_artifacts"] = True
    elif mutation == "absolute_paths":
        contract["publication"]["absolute_paths"] = True
    elif mutation == "split_generation":
        contract["generation_gate"]["split_generation_enabled"] = False
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    output = tmp_path / "near-duplicate-fixture.yml"
    write_fixture_contract(output, contract)
    return output


@pytest.mark.parametrize(
    "mutation",
    [
        "near_contract_hash",
        "positive_cases",
        "negative_seed",
        "threshold_rule",
        "threshold_recall_type",
        "sample_count",
        "production_claim",
        "writes_artifacts",
        "absolute_paths",
        "split_generation",
    ],
)
def test_rejects_fixture_contract_mutations(tmp_path: Path, mutation: str) -> None:
    with pytest.raises(NearDuplicateFixtureError):
        load_near_duplicate_fixture_contract(
            mutated_fixture_contract(tmp_path, mutation),
            NEAR_DUPLICATE_CONTRACT,
            DATASET_SPEC,
            CONVERSION_CONTRACT,
        )


def test_rejects_oversized_fixture_contract(tmp_path: Path) -> None:
    path = tmp_path / "near-duplicate-fixture.yml"
    path.write_bytes(b"#" * (MAX_NEAR_DUPLICATE_FIXTURE_CONTRACT_BYTES + 1))

    with pytest.raises(NearDuplicateFixtureError, match="exceeds"):
        load_near_duplicate_fixture_contract(
            path, NEAR_DUPLICATE_CONTRACT, DATASET_SPEC, CONVERSION_CONTRACT
        )


def test_rejects_fixture_contract_symlink(tmp_path: Path) -> None:
    path = tmp_path / "near-duplicate-fixture.yml"
    path.write_text("schema_version: 1\n", encoding="utf-8")

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(NearDuplicateFixtureError, match="symlink"),
    ):
        load_near_duplicate_fixture_contract(
            path, NEAR_DUPLICATE_CONTRACT, DATASET_SPEC, CONVERSION_CONTRACT
        )


def test_near_duplicate_fixture_cli_reports_evidence() -> None:
    result = subprocess.run(
        [sys.executable, "code/scripts/validate_near_duplicate_fixture.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", errors="replace")
    summary = json.loads(result.stdout)
    assert summary["fixture_contract_id"] == "radioml_2016_10a_near_duplicate_fixture_v1"
    assert summary["calibration"]["passed"] is True
    assert summary["bounded_reference_audit"]["passed"] is True
    assert summary["generation_gate"]["split_generation_enabled"] is True
