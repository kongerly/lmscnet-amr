from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from na_lmscnet.data.numeric_quality import (
    NumericQualityCollector,
    audit_numeric_quality_archive,
)
from na_lmscnet.data.pickle_safety import UnsafePickleError
from na_lmscnet.data.pickle_schema import (
    _ByteString,
    _StaticSchemaInterpreter,
    validate_pickle_schema_stream,
)
from test_pickle_schema import dataset_payload, write_archive, write_spec
from test_pickle_schema import spec as schema_spec


def numeric_spec(
    *,
    modulations: list[str] | None = None,
    samples_per_cell: int = 2,
) -> dict[str, object]:
    expected_modulations = modulations or ["QPSK"]
    return {
        "schema_version": 1,
        "dataset_id": "radioml_2016_10a",
        "archive_filename": "RML2016.10a.tar.bz2",
        "official_page": "https://www.deepsig.ai/datasets/",
        "expected": {
            "modulations": expected_modulations,
            "snr_db": [0],
            "sample_shape": [2, 2],
            "samples_per_cell": samples_per_cell,
            "total_samples": len(expected_modulations) * samples_per_cell,
            "dtype": "float32",
        },
    }


def float32_payload(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes(order="C")


def test_collects_finite_statistics_iq_power_dc_and_fingerprints() -> None:
    collector = NumericQualityCollector.from_spec(numeric_spec())
    payload = float32_payload([1, -1, 2, -2, 0, 0, 0, 0])

    collector.observe(("QPSK", 0), payload)
    report = collector.as_dict()

    assert report["cell_count"] == 1
    assert report["sample_count"] == 2
    assert report["value_count"] == 8
    assert report["finite_count"] == 8
    assert report["nan_count"] == 0
    assert report["positive_infinity_count"] == 0
    assert report["negative_infinity_count"] == 0
    assert report["zero_value_count"] == 4
    assert report["zero_energy_sample_count"] == 1
    assert report["min_value"] == -2.0
    assert report["max_value"] == 2.0
    assert report["mean_value"] == 0.0
    assert report["mean_square"] == pytest.approx(1.25)
    assert report["rms_value"] == pytest.approx(math.sqrt(1.25))
    assert report["mean_iq_power"] == pytest.approx(2.5)
    assert report["channel_dc_means"] == [0.0, 0.0]
    assert report["sample_duplicates"] == {
        "distinct_duplicate_hashes": 0,
        "duplicate_occurrences_beyond_first": 0,
    }
    assert report["cell_duplicates"] == {
        "distinct_duplicate_hashes": 0,
        "duplicate_occurrences_beyond_first": 0,
    }
    assert report["all_values_finite"] is True
    assert report["all_samples_nonzero_energy"] is False
    assert len(report["dataset_content_sha256"]) == 64
    assert len(report["cells"][0]["sha256"]) == 64


def test_counts_nonfinite_values_without_polluting_finite_statistics() -> None:
    collector = NumericQualityCollector.from_spec(numeric_spec(samples_per_cell=1))
    collector.observe(("QPSK", 0), float32_payload([math.nan, 0, math.inf, -math.inf]))

    report = collector.as_dict()

    assert report["finite_count"] == 1
    assert report["nan_count"] == 1
    assert report["positive_infinity_count"] == 1
    assert report["negative_infinity_count"] == 1
    assert report["min_value"] == 0.0
    assert report["max_value"] == 0.0
    assert report["mean_iq_power"] is None
    assert report["all_values_finite"] is False


def test_detects_exact_sample_and_cell_duplicates_across_cells() -> None:
    collector = NumericQualityCollector.from_spec(
        numeric_spec(modulations=["QPSK", "BPSK"], samples_per_cell=1)
    )
    payload = float32_payload([1, 2, 3, 4])

    collector.observe(("QPSK", 0), payload)
    collector.observe(("BPSK", 0), payload)
    report = collector.as_dict()

    assert report["sample_duplicates"] == {
        "distinct_duplicate_hashes": 1,
        "duplicate_occurrences_beyond_first": 1,
    }
    assert report["cell_duplicates"] == {
        "distinct_duplicate_hashes": 1,
        "duplicate_occurrences_beyond_first": 1,
    }


def test_detects_duplicate_samples_within_a_cell() -> None:
    collector = NumericQualityCollector.from_spec(numeric_spec())
    sample = [1, 2, 3, 4]

    collector.observe(("QPSK", 0), float32_payload(sample + sample))
    report = collector.as_dict()

    assert report["sample_duplicates"] == {
        "distinct_duplicate_hashes": 1,
        "duplicate_occurrences_beyond_first": 1,
    }
    assert report["cells"][0]["sample_duplicates"] == report["sample_duplicates"]


def test_rejects_unexpected_duplicate_and_wrong_length_cells() -> None:
    collector = NumericQualityCollector.from_spec(numeric_spec())
    payload = float32_payload([0] * 8)

    with pytest.raises(UnsafePickleError, match="unexpected cell"):
        collector.observe(("BPSK", 0), payload)
    collector.observe(("QPSK", 0), payload)
    with pytest.raises(UnsafePickleError, match="duplicate cell"):
        collector.observe(("QPSK", 0), payload)

    second = NumericQualityCollector.from_spec(numeric_spec())
    with pytest.raises(UnsafePickleError, match="has 4 bytes, expected 32"):
        second.observe(("QPSK", 0), b"\0" * 4)


def test_rejects_incomplete_audit_and_resource_limits() -> None:
    collector = NumericQualityCollector.from_spec(
        numeric_spec(modulations=["QPSK", "BPSK"], samples_per_cell=1)
    )
    collector.observe(("QPSK", 0), float32_payload([0] * 4))
    with pytest.raises(UnsafePickleError, match="observed 1 cells, expected 2"):
        collector.as_dict()

    with (
        patch("na_lmscnet.data.numeric_quality.MAX_AUDIT_SAMPLES", 1),
        pytest.raises(ValueError, match="dataset limit"),
    ):
        NumericQualityCollector.from_spec(numeric_spec())
    limited = NumericQualityCollector.from_spec(numeric_spec())
    with (
        patch("na_lmscnet.data.numeric_quality.MAX_AUDIT_CELLS", 0),
        pytest.raises(UnsafePickleError, match="cell limit"),
    ):
        limited.observe(("QPSK", 0), float32_payload([0] * 8))


def test_schema_observer_receives_validated_key_and_exact_buffer() -> None:
    observed: list[tuple[tuple[str, int], bytes]] = []

    report = validate_pickle_schema_stream(
        io.BytesIO(dataset_payload()),
        schema_spec(),
        buffer_observer=lambda key, payload: observed.append((key, payload)),
    )

    assert report["cell_count"] == 1
    assert observed == [(("QPSK", 0), b"\0" * 4)]


def test_schema_observer_releases_memoized_raw_buffer() -> None:
    interpreter = _StaticSchemaInterpreter(buffer_observer=lambda _key, _payload: None)

    interpreter.run(io.BytesIO(dataset_payload()))

    assert all(
        not isinstance(value, _ByteString) or value.raw is None
        for value in interpreter.memo.values()
    )


def test_audits_archive_without_deserialization_or_conversion(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, dataset_payload())
    write_spec(spec_path)

    report = audit_numeric_quality_archive(archive_path, spec_path)

    assert report["numeric_quality"]["sample_count"] == 1
    assert report["numeric_quality"]["all_values_finite"] is True
    assert report["verification"]["numeric_values_inspected"] is True
    assert report["verification"]["exact_sample_fingerprints_computed"] is True
    assert report["verification"]["near_duplicate_detection_performed"] is False
    assert report["security"]["pickle_deserialized"] is False
    assert report["security"]["pickle_globals_imported"] is False
    assert report["security"]["numpy_read_only_views_constructed"] is True
    assert report["security"]["converted_dataset_written"] is False
    assert not (tmp_path / "RML2016.10a_dict.pkl").exists()


def test_numeric_audit_cli_writes_atomic_json(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    output_path = tmp_path / "RML2016.10a.numeric-audit.json"
    write_archive(archive_path, dataset_payload())
    write_spec(spec_path)
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "code/scripts/audit_numeric_quality.py",
            str(archive_path),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", errors="replace")
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["numeric_quality"]["sample_count"] == 1
    assert list(tmp_path.glob(".*.tmp")) == []


def test_numeric_audit_cli_rejects_wrong_output_suffix(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    output_path = tmp_path / "audit.json"
    write_archive(archive_path, dataset_payload())
    write_spec(spec_path)
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "code/scripts/audit_numeric_quality.py",
            str(archive_path),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
    )

    assert result.returncode != 0
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert "must end with '.numeric-audit.json'" in output
    assert not output_path.exists()
