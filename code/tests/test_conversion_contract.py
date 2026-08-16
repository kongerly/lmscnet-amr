from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from na_lmscnet.data.conversion_contract import (
    MAX_CONVERSION_CONTRACT_BYTES,
    ConversionContractError,
    conversion_contract_sha256,
    conversion_row_index,
    conversion_sample_id,
    load_conversion_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"


def repository_contract() -> dict[str, object]:
    return load_conversion_contract(CONVERSION_CONTRACT, DATASET_SPEC)


def write_contract(path: Path, contract: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def test_repository_conversion_contract_is_complete_and_bound_to_source() -> None:
    contract = repository_contract()

    assert contract["contract_id"] == "radioml_2016_10a_hdf5_v1"
    assert contract["source"]["archive_sha256"] == (
        "7a1603dd61e557f45b6e113dc0c59be02a14509b77856c31bbb324a993f7974c"
    )
    assert contract["source"]["dataset_content_sha256"] == (
        "bcaf1ea9bca18db5b5e179352b18504e6f92d1db7f4cf5b12673c2e3fba9aef9"
    )
    assert contract["format"]["datasets"]["iq"]["shape"] == [220000, 2, 128]
    assert contract["writer"] == {
        "mode": "single-process",
        "swmr": False,
        "overwrite": False,
        "temporary_same_directory": True,
        "fsync_before_publish": True,
        "manifest_published_last": True,
    }
    assert len(conversion_contract_sha256(CONVERSION_CONTRACT)) == 64


@pytest.mark.parametrize(
    ("modulation", "snr_db", "source_index", "expected_row", "expected_id"),
    [
        ("8PSK", -20, 0, 0, "radioml_2016_10a:8PSK:-20:0000"),
        ("8PSK", -8, 37, 6037, "radioml_2016_10a:8PSK:-08:0037"),
        ("QPSK", 0, 999, 190999, "radioml_2016_10a:QPSK:+00:0999"),
        ("WBFM", 18, 999, 219999, "radioml_2016_10a:WBFM:+18:0999"),
    ],
)
def test_maps_source_coordinates_to_stable_rows_and_ids(
    modulation: str,
    snr_db: int,
    source_index: int,
    expected_row: int,
    expected_id: str,
) -> None:
    contract = repository_contract()

    assert conversion_row_index(contract, modulation, snr_db, source_index) == expected_row
    assert conversion_sample_id(contract, modulation, snr_db, source_index) == expected_id


@pytest.mark.parametrize(
    ("modulation", "snr_db", "source_index", "error"),
    [
        ("UNKNOWN", 0, 0, "Unknown modulation"),
        ("QPSK", 1, 0, "Unknown SNR"),
        ("QPSK", 0, -1, "outside"),
        ("QPSK", 0, 1000, "outside"),
    ],
)
def test_rejects_invalid_source_coordinates(
    modulation: str, snr_db: int, source_index: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        conversion_row_index(repository_contract(), modulation, snr_db, source_index)


@pytest.mark.parametrize(
    ("snr_db", "source_index"),
    [(True, 0), (0, True), (0.5, 0), (0, 1.5)],
)
def test_rejects_non_integer_source_coordinates(snr_db: object, source_index: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        conversion_row_index(repository_contract(), "QPSK", snr_db, source_index)  # type: ignore[arg-type]


def mutated_contract(tmp_path: Path, mutation: str) -> Path:
    contract = deepcopy(repository_contract())
    if mutation == "extra_field":
        contract["unexpected"] = True
    elif mutation == "source_hash":
        contract["source"]["archive_sha256"] = "A" * 64
    elif mutation == "shape":
        contract["format"]["datasets"]["iq"]["shape"] = [219999, 2, 128]
    elif mutation == "chunk":
        contract["format"]["datasets"]["iq"]["chunks"] = [220001, 2, 128]
    elif mutation == "compression":
        contract["format"]["datasets"]["iq"]["compression"] = "gzip"
    elif mutation == "track_times":
        contract["format"]["datasets"]["iq"]["track_times"] = True
    elif mutation == "ordering":
        contract["ordering"]["modulation_order"] = list(
            reversed(contract["ordering"]["modulation_order"])
        )
    elif mutation == "absolute_output":
        contract["format"]["output_filename"] = "C:/data/RML2016.10a.h5"
    elif mutation == "windows_output":
        contract["format"]["output_filename"] = "C:\\data\\RML2016.10a.h5"
    elif mutation == "manifest_digest":
        contract["manifest"]["required_digests"].pop()
    elif mutation == "logical_framing":
        contract["manifest"]["logical_digest"]["record_framing"] = "delimiter-v1"
    elif mutation == "concurrent_writer":
        contract["writer"]["mode"] = "multi-process"
    elif mutation == "overwrite":
        contract["writer"]["overwrite"] = True
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")
    output = tmp_path / "contract.yml"
    write_contract(output, contract)
    return output


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "source_hash",
        "shape",
        "chunk",
        "compression",
        "track_times",
        "ordering",
        "absolute_output",
        "windows_output",
        "manifest_digest",
        "logical_framing",
        "concurrent_writer",
        "overwrite",
    ],
)
def test_rejects_inconsistent_or_unsafe_contract_mutations(tmp_path: Path, mutation: str) -> None:
    with pytest.raises(ConversionContractError):
        load_conversion_contract(mutated_contract(tmp_path, mutation), DATASET_SPEC)


def test_rejects_oversized_contract(tmp_path: Path) -> None:
    path = tmp_path / "contract.yml"
    path.write_bytes(b"#" * (MAX_CONVERSION_CONTRACT_BYTES + 1))

    with pytest.raises(ConversionContractError, match="exceeds"):
        load_conversion_contract(path, DATASET_SPEC)


def test_rejects_contract_symlink(tmp_path: Path) -> None:
    path = tmp_path / "contract.yml"
    path.write_text("schema_version: 1\n", encoding="utf-8")

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(ConversionContractError, match="must not be a symlink"),
    ):
        load_conversion_contract(path, DATASET_SPEC)


def test_conversion_contract_cli_reports_validated_summary() -> None:
    result = subprocess.run(
        [sys.executable, "code/scripts/validate_conversion_contract.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", errors="replace")
    summary = json.loads(result.stdout)
    assert summary["contract_id"] == "radioml_2016_10a_hdf5_v1"
    assert summary["format"] == "hdf5"
    assert summary["output_filename"] == "RML2016.10a.h5"
    assert len(summary["contract_sha256"]) == 64
