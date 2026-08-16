from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from na_lmscnet.data.split_contract import (
    MAX_SPLIT_CONTRACT_BYTES,
    SplitContractError,
    allocation_counts,
    assign_stratum_sample_ids,
    load_split_contract,
    rank_sample_ids,
    split_contract_sha256,
    split_rank_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def repository_contract() -> dict[str, object]:
    return load_split_contract(SPLIT_CONTRACT, DATASET_SPEC, CONVERSION_CONTRACT)


def write_contract(path: Path, contract: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def test_repository_split_contract_is_complete_and_generation_enabled() -> None:
    contract = repository_contract()

    assert contract["contract_id"] == "radioml_2016_10a_split_v1"
    assert contract["assignment"]["seed"] == 2026
    assert contract["source"]["hdf5"]["file_sha256"] == (
        "96120f40a9190bf24697227aaa7377a4e1cf883b3bb1b602b176f2622ebf7c63"
    )
    assert contract["source"]["conversion_manifest"]["file_sha256"] == (
        "de5bcb3dc6c490dca774d18bb7f3d3fd79634b55f9e2c31af244ac55b8ea776e"
    )
    assert contract["stratification"]["expected_totals"] == {
        "train": 154000,
        "validation": 22000,
        "test": 44000,
    }
    assert contract["generation_gate"] == {
        "split_generation_enabled": True,
        "blocked_by": [],
    }
    assert len(split_contract_sha256(SPLIT_CONTRACT)) == 64


@pytest.mark.parametrize(
    ("sample_count", "expected"),
    [
        (0, {"train": 0, "validation": 0, "test": 0}),
        (1, {"train": 1, "validation": 0, "test": 0}),
        (3, {"train": 2, "validation": 0, "test": 1}),
        (7, {"train": 5, "validation": 1, "test": 1}),
        (1000, {"train": 700, "validation": 100, "test": 200}),
    ],
)
def test_largest_remainder_allocation_is_exact(sample_count: int, expected: dict[str, int]) -> None:
    assert allocation_counts(repository_contract(), sample_count) == expected


@pytest.mark.parametrize("sample_count", [True, -1, 1.5, "1000"])
def test_allocation_rejects_invalid_counts(sample_count: object) -> None:
    with pytest.raises(SplitContractError):
        allocation_counts(repository_contract(), sample_count)  # type: ignore[arg-type]


def test_rank_digest_has_a_fixed_cross_implementation_vector() -> None:
    digest = split_rank_digest(repository_contract(), "radioml_2016_10a:QPSK:+00:0999")

    assert digest == "6ba3315ff624ccf05597fcc515766a3a5692c34af59a79e1e15f663f703b67bb"


def test_ranking_is_deterministic_and_independent_of_input_order() -> None:
    contract = repository_contract()
    identifiers = [
        "radioml_2016_10a:8PSK:-20:0002",
        "radioml_2016_10a:8PSK:-20:0000",
        "radioml_2016_10a:8PSK:-20:0001",
    ]

    assert rank_sample_ids(contract, identifiers) == rank_sample_ids(
        contract, list(reversed(identifiers))
    )


def test_ranking_rejects_duplicate_or_invalid_sample_ids() -> None:
    contract = repository_contract()
    with pytest.raises(SplitContractError, match="duplicates"):
        rank_sample_ids(contract, ["sample", "sample"])
    with pytest.raises(SplitContractError, match="non-empty"):
        rank_sample_ids(contract, [""])


def test_assigns_one_complete_stratum_exhaustively_and_disjointly() -> None:
    contract = repository_contract()
    sample_ids = [f"radioml_2016_10a:8PSK:-20:{index:04d}" for index in range(1000)]

    assignments = assign_stratum_sample_ids(contract, sample_ids)

    assert {name: len(values) for name, values in assignments.items()} == {
        "train": 700,
        "validation": 100,
        "test": 200,
    }
    combined = [value for split in assignments.values() for value in split]
    assert len(combined) == len(set(combined)) == 1000
    assert set(combined) == set(sample_ids)
    assert assignments == assign_stratum_sample_ids(contract, list(reversed(sample_ids)))


def test_rejects_incomplete_stratum() -> None:
    with pytest.raises(SplitContractError, match="exactly 1000"):
        assign_stratum_sample_ids(repository_contract(), ["one-sample"])


@pytest.mark.parametrize(
    "mutation", ["mixed_stratum", "unknown_modulation", "missing_index", "negative_zero_snr"]
)
def test_rejects_invalid_full_stratum_sample_ids(mutation: str) -> None:
    sample_ids = [f"radioml_2016_10a:8PSK:-20:{index:04d}" for index in range(1000)]
    if mutation == "mixed_stratum":
        sample_ids[-1] = "radioml_2016_10a:8PSK:-18:0999"
    elif mutation == "unknown_modulation":
        sample_ids[-1] = "radioml_2016_10a:UNKNOWN:-20:0999"
    elif mutation == "missing_index":
        sample_ids[-1] = "radioml_2016_10a:8PSK:-20:1000"
    elif mutation == "negative_zero_snr":
        sample_ids = [f"radioml_2016_10a:8PSK:-00:{index:04d}" for index in range(1000)]

    with pytest.raises(SplitContractError):
        assign_stratum_sample_ids(repository_contract(), sample_ids)


def mutated_contract(tmp_path: Path, mutation: str) -> Path:
    contract = deepcopy(repository_contract())
    if mutation == "extra_field":
        contract["unexpected"] = True
    elif mutation == "dataset_spec_hash":
        contract["source"]["dataset_spec_sha256"] = "0" * 64
    elif mutation == "source_hash":
        contract["source"]["source_archive_sha256"] = "A" * 64
    elif mutation == "absolute_hdf5":
        contract["source"]["hdf5"]["filename"] = "C:/data/RML2016.10a.h5"
    elif mutation == "hdf5_hash":
        contract["source"]["hdf5"]["file_sha256"] = "0" * 64
    elif mutation == "manifest_hash":
        contract["source"]["conversion_manifest"]["file_sha256"] = "0" * 64
    elif mutation == "strata":
        contract["stratification"]["strata"] = 219
    elif mutation == "modulation_order":
        contract["stratification"]["modulation_order"].reverse()
    elif mutation == "ratio":
        contract["stratification"]["ratio_weights"]["train"] = 8
    elif mutation == "rounding":
        contract["stratification"]["rounding"]["algorithm"] = "round"
    elif mutation == "seed":
        contract["assignment"]["seed"] = 2027
    elif mutation == "rng":
        contract["assignment"]["algorithm"] = "numpy-rng"
    elif mutation == "sample_identity":
        contract["assignment"]["sample_identity"]["source_index_width"] = 3
    elif mutation == "exact_policy":
        contract["leakage"]["exact_duplicates"]["cross_split_policy"] = "warn"
    elif mutation == "near_disabled":
        contract["leakage"]["near_duplicates"]["required_for_split_generation"] = True
    elif mutation == "near_contract":
        contract["leakage"]["near_duplicates"]["audit_contract"] = "other_contract"
    elif mutation == "adjacency_claim":
        contract["leakage"]["adjacent_windows"]["source_index_is_window_group"] = True
    elif mutation == "test_training":
        contract["test_isolation"]["training_allowed_splits"] = ["train", "test"]
    elif mutation == "freeze_binding":
        contract["test_isolation"]["freeze_must_bind"].pop()
    elif mutation == "absolute_paths":
        contract["manifest"]["absolute_paths"] = True
    elif mutation == "manifest_binding":
        contract["manifest"]["required_bindings"].pop()
    elif mutation == "inside_repository":
        contract["publication"]["output_outside_repository"] = False
    elif mutation == "generation_enabled":
        contract["generation_gate"]["split_generation_enabled"] = False
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    output = tmp_path / "split.yml"
    write_contract(output, contract)
    return output


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "dataset_spec_hash",
        "source_hash",
        "absolute_hdf5",
        "hdf5_hash",
        "manifest_hash",
        "strata",
        "modulation_order",
        "ratio",
        "rounding",
        "seed",
        "rng",
        "sample_identity",
        "exact_policy",
        "near_disabled",
        "near_contract",
        "adjacency_claim",
        "test_training",
        "freeze_binding",
        "absolute_paths",
        "manifest_binding",
        "inside_repository",
        "generation_enabled",
    ],
)
def test_rejects_inconsistent_or_weakened_contract_mutations(tmp_path: Path, mutation: str) -> None:
    with pytest.raises((SplitContractError, ValueError)):
        load_split_contract(mutated_contract(tmp_path, mutation), DATASET_SPEC, CONVERSION_CONTRACT)


def test_rejects_oversized_contract(tmp_path: Path) -> None:
    path = tmp_path / "split.yml"
    path.write_bytes(b"#" * (MAX_SPLIT_CONTRACT_BYTES + 1))

    with pytest.raises(SplitContractError, match="exceeds"):
        load_split_contract(path, DATASET_SPEC, CONVERSION_CONTRACT)


def test_rejects_contract_symlink(tmp_path: Path) -> None:
    path = tmp_path / "split.yml"
    path.write_text("schema_version: 1\n", encoding="utf-8")

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(SplitContractError, match="must not be a symlink"),
    ):
        load_split_contract(path, DATASET_SPEC, CONVERSION_CONTRACT)


def test_split_contract_cli_reports_validated_summary() -> None:
    result = subprocess.run(
        [sys.executable, "code/scripts/validate_split_contract.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", errors="replace")
    summary = json.loads(result.stdout)
    assert summary["contract_id"] == "radioml_2016_10a_split_v1"
    assert summary["generation_enabled"] is True
    assert summary["per_stratum"] == {
        "train": 700,
        "validation": 100,
        "test": 200,
    }
    assert len(summary["contract_sha256"]) == 64
