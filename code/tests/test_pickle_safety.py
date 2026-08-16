from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from na_lmscnet.data.pickle_safety import (
    UnsafePickleError,
    inspect_pickle_archive,
    scan_pickle_stream,
)

MODULATIONS = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]
SNR_VALUES = list(range(-20, 20, 2))


def make_protocol_zero_payload() -> bytes:
    literals = [f"S'{value}'\n".encode() for value in MODULATIONS]
    integers = [f"I{value}\n".encode() for value in SNR_VALUES]
    return b"(" + b"".join(literals + integers) + b"."


def write_archive(path: Path, pickle_payload: bytes, *, extra_member: bool = False) -> None:
    with tarfile.open(path, "w:bz2") as archive:
        for name, payload in [
            ("RML2016.10a_dict.pkl", pickle_payload),
            (
                "LICENSE.TXT",
                b"DeepSig Attribution-NonCommercial-ShareAlike 4.0",
            ),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if extra_member:
            info = tarfile.TarInfo("unexpected.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))


def write_spec(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_id": "radioml_2016_10a",
                "archive_filename": "RML2016.10a.tar.bz2",
                "official_page": "https://www.deepsig.ai/datasets/",
                "expected": {"modulations": MODULATIONS, "snr_db": SNR_VALUES},
            }
        ),
        encoding="utf-8",
    )


def test_scans_python2_protocol_zero_without_decoding_binary_string() -> None:
    stream = io.BytesIO(b"(dp1\nS'QPSK'\np2\nI-20\nsS'\\xc1\\xbb'\np3\ns.")

    report = scan_pickle_stream(stream)

    assert report.protocol_versions == {0}
    assert report.stop_count == 1
    assert report.literal_strings["QPSK"] == 1
    assert report.integer_values[-20] == 1
    assert report.restricted_opcodes == {}


def test_records_global_and_reduce_as_unexecuted_restricted_opcodes() -> None:
    stream = io.BytesIO(b"cnumpy.core.multiarray\n_reconstruct\np0\nR.")

    report = scan_pickle_stream(stream)

    assert report.global_references["numpy.core.multiarray._reconstruct"] == 1
    assert report.restricted_opcodes["GLOBAL"] == 1
    assert report.restricted_opcodes["REDUCE"] == 1


def test_rejects_oversized_global_reference() -> None:
    stream = io.BytesIO(b"cpackage\nfunction\n.")

    with (
        patch("na_lmscnet.data.pickle_safety.MAX_GLOBAL_REFERENCE_TEXT_BYTES", 4),
        pytest.raises(UnsafePickleError, match="global reference exceeds"),
    ):
        scan_pickle_stream(stream)


def test_rejects_stream_without_stop() -> None:
    with pytest.raises(UnsafePickleError, match="before STOP"):
        scan_pickle_stream(io.BytesIO(b"S'QPSK'\n"))


def test_rejects_oversized_opcode_argument() -> None:
    with (
        patch("na_lmscnet.data.pickle_safety.MAX_OPCODE_ARGUMENT_BYTES", 2),
        pytest.raises(UnsafePickleError, match="argument exceeds"),
    ):
        scan_pickle_stream(io.BytesIO(b"B\x03\x00\x00\x00abc."))


def test_inspects_expected_archive_without_deserializing(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, make_protocol_zero_payload())
    write_spec(spec_path)

    report = inspect_pickle_archive(archive_path, spec_path)

    assert report["expected_grid"]["modulations_observed"] == sorted(MODULATIONS)
    assert report["expected_grid"]["snr_db_observed"] == SNR_VALUES
    assert report["security"] == {
        "archive_extracted": False,
        "pickle_deserialized": False,
        "globals_executed": False,
    }
    assert report["verification_scope"] == {
        "description": "opcode metadata and expected modulation/SNR literal presence only",
        "sample_count_verified": False,
        "dtype_verified": False,
        "shape_verified": False,
    }
    assert not (tmp_path / "RML2016.10a_dict.pkl").exists()


def test_rejects_archive_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, make_protocol_zero_payload())
    write_spec(spec_path)

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(ValueError, match="not a symlink"),
    ):
        inspect_pickle_archive(archive_path, spec_path)


def test_rejects_unexpected_archive_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, make_protocol_zero_payload(), extra_member=True)
    write_spec(spec_path)

    with pytest.raises(UnsafePickleError, match="exactly match"):
        inspect_pickle_archive(archive_path, spec_path)


def test_rejects_unexpected_pickle_protocol(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, b"\x80\x04.")
    write_spec(spec_path)

    with pytest.raises(UnsafePickleError, match="protocol 0"):
        inspect_pickle_archive(archive_path, spec_path)


def test_rejects_trailing_pickle_data(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, make_protocol_zero_payload() + b"second-object")
    write_spec(spec_path)

    with pytest.raises(UnsafePickleError, match="trailing data"):
        inspect_pickle_archive(archive_path, spec_path)


def test_pickle_inspection_cli_writes_json(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    output_path = tmp_path / "RML2016.10a.pickle-scan.json"
    write_archive(archive_path, make_protocol_zero_payload())
    write_spec(spec_path)
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "code/scripts/inspect_pickle_payload.py",
            str(archive_path),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert list(tmp_path.glob(".*.tmp")) == []
