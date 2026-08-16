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

from na_lmscnet.data.pickle_safety import UnsafePickleError
from na_lmscnet.data.pickle_schema import (
    MAX_SCHEMA_CELLS,
    validate_pickle_schema_archive,
    validate_pickle_schema_stream,
)


def pickle_string(value: bytes) -> bytes:
    escaped = b"".join(f"\\x{byte:02x}".encode() for byte in value)
    return b"S'" + escaped + b"'\n"


def array_payload(
    *,
    shape: tuple[int, ...] = (1, 1, 1),
    dtype_code: str = "f4",
    buffer: bytes = b"\x00\x00\x00\x00",
    ndarray_global: bytes = b"cnumpy\nndarray\n",
) -> bytes:
    shape_pickle = b"(" + b"".join(f"I{value}\n".encode() for value in shape) + b"t"
    dtype = (
        b"cnumpy\ndtype\n"
        b"(" + pickle_string(dtype_code.encode()) + b"I0\nI1\ntR"
        b"(I3\n" + pickle_string(b"<") + b"NNNI-1\nI-1\nI0\ntb"
    )
    return b"".join(
        (
            b"cnumpy.core.multiarray\n_reconstruct\n",
            b"(" + ndarray_global + b"(I0\nt" + pickle_string(b"b") + b"tR",
            b"(I1\n" + shape_pickle,
            dtype,
            b"I00\n" + pickle_string(buffer) + b"tb",
        )
    )


def dataset_payload(
    cells: list[tuple[str, int, bytes]] | None = None,
) -> bytes:
    if cells is None:
        cells = [("QPSK", 0, array_payload())]
    encoded_cells = []
    for modulation, snr, array in cells:
        key = b"(" + pickle_string(modulation.encode()) + f"I{snr}\n".encode() + b"t"
        encoded_cells.append(key + array + b"s")
    return b"(d" + b"".join(encoded_cells) + b"."


def spec(
    *,
    modulations: list[str] | None = None,
    snr_values: list[int] | None = None,
) -> dict[str, object]:
    expected_modulations = modulations or ["QPSK"]
    expected_snr = snr_values or [0]
    return {
        "schema_version": 1,
        "dataset_id": "radioml_2016_10a",
        "archive_filename": "RML2016.10a.tar.bz2",
        "official_page": "https://www.deepsig.ai/datasets/",
        "expected": {
            "modulations": expected_modulations,
            "snr_db": expected_snr,
            "sample_shape": [1, 1],
            "samples_per_cell": 1,
            "total_samples": len(expected_modulations) * len(expected_snr),
            "dtype": "float32",
        },
    }


def write_archive(
    path: Path,
    payload: bytes,
    *,
    include_extra_member: bool = False,
) -> None:
    with tarfile.open(path, "w:bz2") as archive:
        for name, content in [
            ("RML2016.10a_dict.pkl", payload),
            ("LICENSE.TXT", b"DeepSig Attribution-NonCommercial-ShareAlike 4.0"),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if include_extra_member:
            member = tarfile.TarInfo("unexpected.txt")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))


def write_spec(path: Path, dataset_spec: dict[str, object] | None = None) -> None:
    path.write_text(yaml.safe_dump(dataset_spec or spec()), encoding="utf-8")


def test_validates_complete_static_schema_without_constructing_numpy() -> None:
    report = validate_pickle_schema_stream(io.BytesIO(dataset_payload()), spec())

    assert report["cell_count"] == 1
    assert report["array_shape"] == [1, 1, 1]
    assert report["sample_shape"] == [1, 1]
    assert report["dtype_encoding"] == "<f4"
    assert report["memory_order"] == "C"
    assert report["array_buffer_bytes_per_cell"] == 4
    assert report["total_samples"] == 1
    assert report["allowed_globals"] == [
        "numpy.core.multiarray._reconstruct",
        "numpy.dtype",
        "numpy.ndarray",
    ]


def test_rejects_unauthorized_global_reference() -> None:
    payload = dataset_payload([("QPSK", 0, array_payload(ndarray_global=b"cos\nsystem\n"))])

    with pytest.raises(UnsafePickleError, match="GLOBAL reference is not allowed"):
        validate_pickle_schema_stream(io.BytesIO(payload), spec())


def test_rejects_opcode_outside_strict_protocol_subset() -> None:
    with pytest.raises(UnsafePickleError, match="Opcode 'POP' is not allowed"):
        validate_pickle_schema_stream(io.BytesIO(b"(0"), spec())


def test_rejects_unknown_memo_reference() -> None:
    with pytest.raises(UnsafePickleError, match="unknown memo index"):
        validate_pickle_schema_stream(io.BytesIO(b"g1\n."), spec())


def test_rejects_resource_limits() -> None:
    with (
        patch("na_lmscnet.data.pickle_schema.MAX_SCHEMA_STACK_ITEMS", 1),
        pytest.raises(UnsafePickleError, match="stack exceeds"),
    ):
        validate_pickle_schema_stream(io.BytesIO(b"(("), spec())
    with (
        patch("na_lmscnet.data.pickle_schema.MAX_SCHEMA_MEMO_ITEMS", 1),
        pytest.raises(UnsafePickleError, match="Memo index exceeds"),
    ):
        validate_pickle_schema_stream(io.BytesIO(b"I1\np1\n."), spec())
    with (
        patch("na_lmscnet.data.pickle_schema.MAX_SCHEMA_TUPLE_ITEMS", 1),
        pytest.raises(UnsafePickleError, match="tuple exceeds"),
    ):
        validate_pickle_schema_stream(io.BytesIO(b"(I1\nI2\nt."), spec())
    with (
        patch("na_lmscnet.data.pickle_schema.MAX_SCHEMA_OPCODES", 1),
        pytest.raises(UnsafePickleError, match="opcode limit"),
    ):
        validate_pickle_schema_stream(io.BytesIO(b"(d."), spec())
    with (
        patch("na_lmscnet.data.pickle_schema.MAX_SCHEMA_CELLS", 0),
        pytest.raises(ValueError, match="cell limit"),
    ):
        validate_pickle_schema_stream(io.BytesIO(dataset_payload()), spec())
    with (
        patch("na_lmscnet.data.pickle_schema.MAX_DECODED_STRING_BYTES", 3),
        pytest.raises(ValueError, match="buffer exceeds"),
    ):
        validate_pickle_schema_stream(io.BytesIO(dataset_payload()), spec())


def test_rejects_noncanonical_python2_string_escape() -> None:
    payload = dataset_payload().replace(b"S'\\x51\\x50\\x53\\x4b'", b"S'\\q'")

    with pytest.raises(UnsafePickleError, match="Invalid Python 2 STRING"):
        validate_pickle_schema_stream(io.BytesIO(payload), spec())


def test_rejects_wrong_dtype_code() -> None:
    payload = dataset_payload([("QPSK", 0, array_payload(dtype_code="f8"))])

    with pytest.raises(UnsafePickleError, match="Only float32"):
        validate_pickle_schema_stream(io.BytesIO(payload), spec())


def test_rejects_wrong_array_shape() -> None:
    payload = dataset_payload([("QPSK", 0, array_payload(shape=(1, 1, 2), buffer=b"\x00" * 8))])

    with pytest.raises(UnsafePickleError, match="has shape"):
        validate_pickle_schema_stream(io.BytesIO(payload), spec())


def test_rejects_wrong_array_buffer_length() -> None:
    payload = dataset_payload([("QPSK", 0, array_payload(buffer=b"\x00" * 3))])

    with pytest.raises(UnsafePickleError, match="buffer has 3 bytes, expected 4"):
        validate_pickle_schema_stream(io.BytesIO(payload), spec())


def test_rejects_duplicate_dataset_cell() -> None:
    cells = [("QPSK", 0, array_payload()), ("QPSK", 0, array_payload())]

    with pytest.raises(UnsafePickleError, match="Duplicate dataset cell"):
        validate_pickle_schema_stream(io.BytesIO(dataset_payload(cells)), spec())


def test_rejects_incomplete_expected_grid() -> None:
    expected = spec(modulations=["QPSK", "BPSK"])

    with pytest.raises(UnsafePickleError, match="Dataset grid mismatch"):
        validate_pickle_schema_stream(io.BytesIO(dataset_payload()), expected)


def test_rejects_trailing_data_in_direct_stream() -> None:
    with pytest.raises(UnsafePickleError, match="trailing data"):
        validate_pickle_schema_stream(io.BytesIO(dataset_payload() + b"second-object"), spec())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modulations", ["QPSK", "QPSK"], "unique string list"),
        ("snr_db", [False], "unique integer list"),
        ("sample_shape", [1, 0], "two positive integers"),
        ("samples_per_cell", 0, "positive integer"),
        ("total_samples", True, "positive integer"),
        ("total_samples", 2, "complete grid size"),
        ("dtype", "float64", "requires expected dtype float32"),
    ],
)
def test_rejects_invalid_expected_spec(field: str, value: object, message: str) -> None:
    invalid_spec = spec()
    expected = invalid_spec["expected"]
    assert isinstance(expected, dict)
    expected[field] = value

    with pytest.raises(ValueError, match=message):
        validate_pickle_schema_stream(io.BytesIO(dataset_payload()), invalid_spec)


def test_rejects_expected_grid_above_cell_limit() -> None:
    oversized_spec = spec(
        modulations=[f"M{index}" for index in range(MAX_SCHEMA_CELLS + 1)],
    )

    with pytest.raises(ValueError, match="cell limit"):
        validate_pickle_schema_stream(io.BytesIO(dataset_payload()), oversized_spec)


def test_validates_archive_and_reports_security_boundary(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, dataset_payload())
    write_spec(spec_path)

    report = validate_pickle_schema_archive(archive_path, spec_path)

    assert report["verification"] == {
        "complete_grid_verified": True,
        "sample_count_verified": True,
        "dtype_verified": True,
        "shape_verified": True,
        "buffer_sizes_verified": True,
        "numeric_values_inspected": False,
    }
    assert report["security"] == {
        "archive_extracted": False,
        "pickle_deserialized": False,
        "pickle_globals_imported": False,
        "globals_executed": False,
        "static_interpreter": True,
    }
    assert not (tmp_path / "RML2016.10a_dict.pkl").exists()


def test_rejects_archive_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, dataset_payload())
    write_spec(spec_path)

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(ValueError, match="not a symlink"),
    ):
        validate_pickle_schema_archive(archive_path, spec_path)


def test_rejects_unexpected_archive_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, dataset_payload(), include_extra_member=True)
    write_spec(spec_path)

    with pytest.raises(UnsafePickleError, match="exactly match"):
        validate_pickle_schema_archive(archive_path, spec_path)


def test_rejects_trailing_pickle_data_in_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    write_archive(archive_path, dataset_payload() + b"second-object")
    write_spec(spec_path)

    with pytest.raises(UnsafePickleError, match="trailing data"):
        validate_pickle_schema_archive(archive_path, spec_path)


def test_schema_cli_writes_atomic_json_report(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    output_path = tmp_path / "RML2016.10a.pickle-schema.json"
    write_archive(archive_path, dataset_payload())
    write_spec(spec_path)
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "code/scripts/validate_pickle_schema.py",
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
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["validated_schema"]["cell_count"] == 1
    assert list(tmp_path.glob(".*.tmp")) == []


def test_schema_cli_rejects_wrong_output_suffix(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    spec_path = tmp_path / "spec.yml"
    output_path = tmp_path / "schema.json"
    write_archive(archive_path, dataset_payload())
    write_spec(spec_path)
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "code/scripts/validate_pickle_schema.py",
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
    assert "must end with '.pickle-schema.json'" in output
    assert not output_path.exists()
