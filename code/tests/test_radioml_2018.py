from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import na_lmscnet.data.radioml_2018 as rml2018
from na_lmscnet.data.radioml_2018 import (
    RadioML2018Error,
    RadioML2018HDF5Dataset,
    _assignment_sha256,
    _build_split_rows,
    _fsync_file,
    _split_codes,
    _write_json_atomic,
    sample_id,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_spec(hdf5_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "radioml_2018_01a",
        "source": {
            "archive_filename": "fixture.tar.gz",
            "archive_size_bytes": 1,
            "archive_sha256": "1" * 64,
            "hdf5_filename": hdf5_path.name,
            "hdf5_size_bytes": hdf5_path.stat().st_size,
            "hdf5_sha256": _sha256(hdf5_path),
        },
        "expected": {
            "x_shape": [6, 4, 2],
            "y_shape": [6, 2],
            "z_shape": [6, 1],
            "samples_per_stratum": 3,
            "modulations": ["A", "B"],
            "snr_db": [0],
        },
    }


def _write_adapter_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    hdf5_path = tmp_path / "source.h5"
    x = np.arange(48, dtype=np.float32).reshape(6, 4, 2) + 1.0
    with h5py.File(hdf5_path, "w") as file:
        file.create_dataset("X", data=x)
        file.create_dataset("Y", data=np.eye(2, dtype=np.int64)[[0, 0, 0, 1, 1, 1]])
        file.create_dataset("Z", data=np.zeros((6, 1), dtype=np.int64))
    spec = _fixture_spec(hdf5_path)
    spec_path = tmp_path / "spec.yml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    source_manifest_path = tmp_path / "source.json"
    source_manifest_path.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    rows = {
        "train": np.asarray([0, 3], dtype="<i8"),
        "validation": np.asarray([2, 4], dtype="<i8"),
        "test": np.asarray([1, 5], dtype="<i8"),
    }
    assignment = _assignment_sha256(rows)
    split_path = tmp_path / "split.h5"
    with h5py.File(split_path, "w") as file:
        file.attrs["assignment_sha256"] = assignment
        for name, values in rows.items():
            file.create_dataset(name, data=values)
    split_manifest = {
        "counts": {name: len(values) for name, values in rows.items()},
        "assignment": {"sha256": assignment},
        "artifact": {
            "filename": split_path.name,
            "size_bytes": split_path.stat().st_size,
            "sha256": _sha256(split_path),
        },
        "bindings": {
            "source_manifest_sha256": _sha256(source_manifest_path),
            "dataset_spec_sha256": _sha256(spec_path),
        },
    }
    split_manifest_path = tmp_path / "split.json"
    split_manifest_path.write_text(json.dumps(split_manifest), encoding="utf-8")

    monkeypatch.setattr(rml2018, "_validate_dataset_spec", lambda _: None)
    monkeypatch.setattr(rml2018, "_validate_source_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(rml2018, "_validate_split_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(rml2018, "_validate_hdf5_schema", lambda *args, **kwargs: {})
    monkeypatch.setattr(rml2018, "_verified_sha256", lambda path, expected: expected)
    return {
        "hdf5_path": hdf5_path,
        "source_manifest_path": source_manifest_path,
        "split_artifact_path": split_path,
        "split_manifest_path": split_manifest_path,
        "dataset_spec_path": spec_path,
    }


def test_test_split_is_rejected_before_any_file_access(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(RadioML2018Error, match="Only train and validation"):
        RadioML2018HDF5Dataset(
            split="test",  # type: ignore[arg-type]
            hdf5_path=missing,
            source_manifest_path=missing,
            split_artifact_path=missing,
            split_manifest_path=missing,
            dataset_spec_path=missing,
        )


def test_sample_identity_and_split_ranking_are_stable_and_complete() -> None:
    contract = {
        "stratification": {
            "class_count": 2,
            "snr_db": [-2, 0],
            "samples_per_stratum": 10,
            "expected_per_stratum": {"train": 7, "validation": 1, "test": 2},
        },
        "assignment": {"seed": 2026},
    }
    first = _build_split_rows(contract)
    second = _build_split_rows(contract)

    assert sample_id(1, -2, 9) == "radioml_2018_01a:01:-02:0009"
    assert all(np.array_equal(first[name], second[name]) for name in rml2018.SPLITS)
    assert {name: len(first[name]) for name in rml2018.SPLITS} == {
        "train": 28,
        "validation": 4,
        "test": 8,
    }
    codes = _split_codes(first, 40)
    assert sorted(np.bincount(codes).tolist()) == [4, 8, 28]
    assert _assignment_sha256(first) == _assignment_sha256(second)


def test_adapter_transposes_normalizes_and_restores_shuffled_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_adapter_fixture(tmp_path, monkeypatch)
    with RadioML2018HDF5Dataset(split="validation", **paths) as dataset:
        single = dataset[0]
        batch = dataset.__getitems__([1, 0])

    expected = np.arange(16, 24, dtype=np.float32).reshape(4, 2).T + 1.0
    expected /= np.sqrt(np.square(expected).sum(axis=0)).max()
    assert single["iq"].shape == (2, 4)
    assert single["iq"].numpy() == pytest.approx(expected)
    assert single["sample_id"] == "radioml_2018_01a:00:+00:0002"
    assert [sample["sample_id"] for sample in batch] == [
        "radioml_2018_01a:01:+00:0001",
        "radioml_2018_01a:00:+00:0002",
    ]
    assert all(float(sample["iq"].square().sum(dim=0).sqrt().max()) == pytest.approx(1.0) for sample in batch)


def test_adapter_rejects_split_artifact_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_adapter_fixture(tmp_path, monkeypatch)
    manifest = json.loads(paths["split_manifest_path"].read_text(encoding="utf-8"))
    manifest["artifact"]["sha256"] = "0" * 64
    paths["split_manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RadioML2018Error, match="manifest bindings"):
        RadioML2018HDF5Dataset(split="train", **paths)


def test_split_codes_reject_overlap_and_missing_rows() -> None:
    with pytest.raises(RadioML2018Error, match="overlap"):
        _split_codes(
            {
                "train": np.asarray([0, 1]),
                "validation": np.asarray([1]),
                "test": np.asarray([2]),
            },
            3,
        )
    with pytest.raises(RadioML2018Error, match="cover"):
        _split_codes(
            {
                "train": np.asarray([0]),
                "validation": np.asarray([1]),
                "test": np.asarray([], dtype=np.int64),
            },
            3,
        )


def test_atomic_manifest_writer_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    _write_json_atomic({"version": 1}, destination)
    with pytest.raises(RadioML2018Error, match="overwrite"):
        _write_json_atomic({"version": 2}, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 1}


def test_fsync_file_uses_a_windows_compatible_writable_handle(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"frozen")
    _fsync_file(path)
    assert path.read_bytes() == b"frozen"
