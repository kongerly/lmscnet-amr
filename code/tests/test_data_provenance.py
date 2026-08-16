from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from na_lmscnet.data.provenance import (
    MAX_MEMBERS,
    MAX_SPEC_BYTES,
    UnsafeArchiveError,
    build_archive_inventory,
    inspect_tar_bz2,
    load_dataset_spec,
    write_inventory,
)


def write_archive(path: Path, members: list[tuple[str, bytes, bytes | None]]) -> None:
    with tarfile.open(path, "w:bz2") as archive:
        for name, payload, member_type in members:
            info = tarfile.TarInfo(name)
            if member_type is not None:
                info.type = member_type
            if info.isfile():
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                archive.addfile(info)


def write_spec(path: Path, archive_filename: str = "RML2016.10a.tar.bz2") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_id": "radioml_2016_10a",
                "archive_filename": archive_filename,
                "official_page": "https://www.deepsig.ai/datasets/",
            }
        ),
        encoding="utf-8",
    )


def test_repository_spec_matches_verified_generator_grid() -> None:
    project_root = Path(__file__).resolve().parents[2]
    spec = load_dataset_spec(project_root / "code/configs/data/radioml_2016_10a.yml")
    expected = spec["expected"]

    assert spec["archive_filename"] == "RML2016.10a.tar.bz2"
    assert spec["license"] == "CC-BY-NC-SA-4.0"
    assert spec["generator"]["commit"] == "4ecf612cfbc5bfc80eb8b0dbe63ed685d0a73c44"
    assert expected["sample_shape"] == [2, 128]
    assert expected["dtype"] == "float32"
    assert expected["snr_db"] == list(range(-20, 20, 2))
    assert set(expected["modulations"]) == {
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
    }
    assert expected["total_samples"] == (
        len(expected["modulations"]) * len(expected["snr_db"]) * expected["samples_per_cell"]
    )


def test_builds_path_redacted_inventory_without_extracting(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    payload = b"not-a-real-pickle"
    write_archive(archive_path, [("RML2016.10a_dict.pkl", payload, None)])
    spec_path = tmp_path / "spec.yml"
    write_spec(spec_path)

    inventory = build_archive_inventory(archive_path, spec_path)

    assert inventory["archive"]["sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert inventory["members"] == [
        {"name": "RML2016.10a_dict.pkl", "size_bytes": len(payload), "type": "file"}
    ]
    assert inventory["security"] == {
        "archive_extracted": False,
        "payload_deserialized": False,
        "absolute_source_path_recorded": False,
    }
    assert str(tmp_path) not in json.dumps(inventory)
    assert not (tmp_path / "RML2016.10a_dict.pkl").exists()


@pytest.mark.parametrize("name", ["../payload.pkl", "/payload.pkl", "C:/payload.pkl"])
def test_rejects_unsafe_member_paths(tmp_path: Path, name: str) -> None:
    archive_path = tmp_path / "unsafe.tar.bz2"
    write_archive(archive_path, [(name, b"payload", None)])

    with pytest.raises(UnsafeArchiveError, match="unsafe|drive-qualified"):
        inspect_tar_bz2(archive_path)


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_rejects_archive_links(tmp_path: Path, member_type: bytes) -> None:
    archive_path = tmp_path / "link.tar.bz2"
    write_archive(archive_path, [("payload-link", b"", member_type)])

    with pytest.raises(UnsafeArchiveError, match="type is not allowed"):
        inspect_tar_bz2(archive_path)


def test_rejects_unexpected_archive_filename(tmp_path: Path) -> None:
    archive_path = tmp_path / "renamed.tar.bz2"
    write_archive(archive_path, [("payload.pkl", b"payload", None)])
    spec_path = tmp_path / "spec.yml"
    write_spec(spec_path)

    with pytest.raises(ValueError, match="Expected archive filename"):
        build_archive_inventory(archive_path, spec_path)


def test_rejects_archive_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    write_archive(archive_path, [("payload.pkl", b"payload", None)])
    spec_path = tmp_path / "spec.yml"
    write_spec(spec_path)

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(ValueError, match="not a symlink"),
    ):
        build_archive_inventory(archive_path, spec_path)


def test_rejects_excessive_archive_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "too-many.tar.bz2"
    members = [(f"item-{index}.bin", b"x", None) for index in range(MAX_MEMBERS + 1)]
    write_archive(archive_path, members)

    with pytest.raises(UnsafeArchiveError, match="more than"):
        inspect_tar_bz2(archive_path)


def test_rejects_invalid_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "invalid.tar.bz2"
    archive_path.write_bytes(b"not an archive")

    with pytest.raises(UnsafeArchiveError, match="Invalid"):
        inspect_tar_bz2(archive_path)


def test_rejects_oversized_dataset_spec(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    write_archive(archive_path, [("payload.pkl", b"payload", None)])
    spec_path = tmp_path / "spec.yml"
    spec_path.write_bytes(b"#" * (MAX_SPEC_BYTES + 1))

    with pytest.raises(ValueError, match="specification exceeds"):
        build_archive_inventory(archive_path, spec_path)


def test_write_inventory_replaces_output_atomically(tmp_path: Path) -> None:
    output_path = tmp_path / "inventory.json"
    output_path.write_text("stale", encoding="utf-8")

    write_inventory({"schema_version": 1}, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_write_inventory_rejects_output_symlink(tmp_path: Path) -> None:
    output_path = tmp_path / "inventory.json"

    with (
        patch.object(Path, "is_symlink", return_value=True),
        pytest.raises(ValueError, match="must not be a symlink"),
    ):
        write_inventory({"schema_version": 1}, output_path)


def test_inventory_cli_runs_from_repository_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "RML2016.10a.tar.bz2"
    write_archive(archive_path, [("payload.pkl", b"payload", None)])
    spec_path = tmp_path / "spec.yml"
    write_spec(spec_path)
    output_path = tmp_path / "RML2016.10a.dataset-inventory.json"
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "code/scripts/inventory_dataset.py",
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
    assert output_path.is_file()
    assert "SHA-256" in result.stdout
