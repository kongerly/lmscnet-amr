from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import yaml

from na_lmscnet.data.conversion_contract import load_conversion_contract
from na_lmscnet.data.hdf5_conversion import (
    ConversionError,
    _output_directory,
    convert_archive,
    verify_conversion,
)
from test_pickle_schema import array_payload, dataset_payload, spec, write_archive, write_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
REPOSITORY_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"


def source_content_digest(modulation: str, snr: int, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(modulation.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(snr).encode("ascii"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def prepare_fixture(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    cell = np.asarray([1.25], dtype="<f4").tobytes()
    archive = tmp_path / "RML2016.10a.tar.bz2"
    write_archive(
        archive,
        dataset_payload([("QPSK", 0, array_payload(shape=(1, 1, 1), buffer=cell))]),
    )
    spec_path = tmp_path / "spec.yml"
    write_spec(spec_path, spec())

    contract = deepcopy(load_conversion_contract(REPOSITORY_CONTRACT, REPOSITORY_SPEC))
    contract["source"].update(
        {
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "dataset_content_sha256": source_content_digest("QPSK", 0, cell),
        }
    )
    contract["format"]["datasets"] = {
        "iq": {
            "path": "/iq",
            "dtype": "<f4",
            "shape": [1, 1, 1],
            "chunks": [1, 1, 1],
            "compression": None,
            "shuffle": False,
            "fletcher32": True,
            "track_times": False,
        },
        "modulation_index": {
            "path": "/modulation_index",
            "dtype": "|u1",
            "shape": [1],
            "chunks": [1],
            "compression": None,
            "shuffle": False,
            "fletcher32": True,
            "track_times": False,
        },
        "snr_db": {
            "path": "/snr_db",
            "dtype": "|i1",
            "shape": [1],
            "chunks": [1],
            "compression": None,
            "shuffle": False,
            "fletcher32": True,
            "track_times": False,
        },
        "source_index": {
            "path": "/source_index",
            "dtype": "<u2",
            "shape": [1],
            "chunks": [1],
            "compression": None,
            "shuffle": False,
            "fletcher32": True,
            "track_times": False,
        },
        "modulation_names": {
            "path": "/modulation_names",
            "dtype": "|S4",
            "shape": [1],
            "chunks": None,
            "compression": None,
            "shuffle": False,
            "fletcher32": False,
            "track_times": False,
        },
    }
    contract["ordering"] = {
        "dimensions": ["modulation", "snr_db", "source_index"],
        "modulation_order": ["QPSK"],
        "snr_db_order": [0],
        "source_index": {"start": 0, "stop": 1, "step": 1},
    }
    contract["sample_id"]["source_index_width"] = 1
    contract_path = tmp_path / "conversion.yml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    load_conversion_contract(contract_path, spec_path)
    return archive, spec_path, contract_path, cell


def artifact_paths(tmp_path: Path) -> tuple[Path, Path]:
    return (
        tmp_path / "RML2016.10a.h5",
        tmp_path / "RML2016.10a.conversion-manifest.json",
    )


def test_converts_and_independently_verifies_real_static_pickle_fixture(tmp_path: Path) -> None:
    archive, spec_path, contract_path, cell = prepare_fixture(tmp_path)
    archive_before = hashlib.sha256(archive.read_bytes()).hexdigest()

    manifest = convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)
    hdf5_path, manifest_path = artifact_paths(tmp_path)
    result = verify_conversion(
        hdf5_path, manifest_path, archive, spec_path, contract_path, PROJECT_ROOT
    )

    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_before
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["inspection"]["sample_count"] == 1
    assert result["inspection"]["cell_count"] == 1
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")
    assert list(tmp_path.glob(".*.tmp")) == []
    with h5py.File(hdf5_path, "r") as file:
        assert file["iq"][...].tobytes() == cell
        assert file["modulation_index"][...].tolist() == [0]
        assert file["snr_db"][...].tolist() == [0]
        assert file["source_index"][...].tolist() == [0]
        assert file["modulation_names"][...].tolist() == [b"QPSK"]

    with pytest.raises(ConversionError, match="already exists"):
        convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)


def test_verifier_rejects_manifest_schema_tampering(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)
    hdf5_path, manifest_path = artifact_paths(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConversionError, match="fields"):
        verify_conversion(hdf5_path, manifest_path, archive, spec_path, contract_path, PROJECT_ROOT)


def test_verifier_rejects_manifest_digest_tampering(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)
    hdf5_path, manifest_path = artifact_paths(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["digests"]["iq_dataset_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConversionError, match="digest"):
        verify_conversion(hdf5_path, manifest_path, archive, spec_path, contract_path, PROJECT_ROOT)


def test_verifier_rejects_hdf5_iq_tampering(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)
    hdf5_path, manifest_path = artifact_paths(tmp_path)
    with h5py.File(hdf5_path, "r+") as file:
        file["iq"][0, 0, 0] = np.float32(9.0)

    with pytest.raises(ConversionError, match="changed source cell"):
        verify_conversion(hdf5_path, manifest_path, archive, spec_path, contract_path, PROJECT_ROOT)


def test_rejects_output_inside_declared_project_root(tmp_path: Path) -> None:
    inside = tmp_path / "project" / "artifacts"
    inside.mkdir(parents=True)

    with pytest.raises(ConversionError, match="outside the repository"):
        _output_directory(inside, tmp_path / "project")


def test_rejects_wrong_source_hash_before_creating_outputs(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with pytest.raises(ConversionError, match="size|SHA-256"):
        convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)

    assert artifact_paths(tmp_path) == (
        tmp_path / "RML2016.10a.h5",
        tmp_path / "RML2016.10a.conversion-manifest.json",
    )
    assert not any(path.exists() for path in artifact_paths(tmp_path))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_rejects_same_size_source_tampering_before_creating_outputs(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    payload = bytearray(archive.read_bytes())
    payload[-1] ^= 1
    archive.write_bytes(payload)

    with pytest.raises(ConversionError, match="SHA-256"):
        convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)

    assert not any(path.exists() for path in artifact_paths(tmp_path))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_rolls_back_hdf5_when_manifest_publication_fails(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    from na_lmscnet.data import hdf5_conversion

    original = hdf5_conversion._publish_noreplace
    calls = 0

    def fail_second_publication(temporary: Path, final: Path) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConversionError("injected manifest publication failure")
        return original(temporary, final)

    with (
        patch.object(hdf5_conversion, "_publish_noreplace", fail_second_publication),
        pytest.raises(ConversionError, match="injected"),
    ):
        convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)

    assert not any(path.exists() for path in artifact_paths(tmp_path))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_rejects_existing_writer_lock_without_removing_it(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    lock = tmp_path / ".RML2016.10a.h5.conversion.lock"
    lock.write_text("other writer", encoding="ascii")

    with pytest.raises(ConversionError, match="lock already exists"):
        convert_archive(archive, spec_path, contract_path, tmp_path, PROJECT_ROOT)

    assert lock.read_text(encoding="ascii") == "other writer"
    assert not any(path.exists() for path in artifact_paths(tmp_path))


def test_conversion_and_verifier_clis_use_custom_fixture_contract(tmp_path: Path) -> None:
    archive, spec_path, contract_path, _ = prepare_fixture(tmp_path)
    convert = subprocess.run(
        [
            sys.executable,
            "code/scripts/convert_radioml_2016_10a.py",
            str(archive),
            "--output-dir",
            str(tmp_path),
            "--spec",
            str(spec_path),
            "--contract",
            str(contract_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    assert convert.returncode == 0, (convert.stdout + convert.stderr).decode(
        "utf-8", errors="replace"
    )
    hdf5_path, manifest_path = artifact_paths(tmp_path)
    verify = subprocess.run(
        [
            sys.executable,
            "code/scripts/verify_radioml_2016_10a_conversion.py",
            str(archive),
            str(hdf5_path),
            str(manifest_path),
            "--spec",
            str(spec_path),
            "--contract",
            str(contract_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    assert verify.returncode == 0, (verify.stdout + verify.stderr).decode("utf-8", errors="replace")
    assert json.loads(verify.stdout)["verified"] is True
    assert str(tmp_path).encode() not in convert.stdout + verify.stdout
