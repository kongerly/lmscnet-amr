"""Controlled, atomic conversion of the verified RadioML pickle archive to HDF5."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from na_lmscnet.data.conversion_contract import (
    conversion_contract_sha256,
    conversion_row_index,
    load_conversion_contract,
)
from na_lmscnet.data.pickle_schema import validate_pickle_schema_archive
from na_lmscnet.data.provenance import _sha256_stream, load_dataset_spec

_DATASET_NAMES = ("iq", "modulation_index", "snr_db", "source_index", "modulation_names")
_MANIFEST_VERSION = 1
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_HASH_RECORDS = 4096
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class ConversionError(ValueError):
    """Raised when conversion or artifact verification fails closed."""


def _is_reparse_point(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _regular_file(path: Path, field: str) -> Path:
    if path.is_symlink() or _is_reparse_point(path):
        raise ConversionError(f"{field} must not be a symlink or reparse point")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ConversionError(f"{field} does not exist") from error
    if not resolved.is_file():
        raise ConversionError(f"{field} must be a regular file")
    return resolved


def _output_directory(path: Path, project_root: Path) -> Path:
    if path.exists():
        if path.is_symlink() or _is_reparse_point(path):
            raise ConversionError("Output directory must not be a symlink or reparse point")
        if not path.is_dir():
            raise ConversionError("Output directory must be a directory")
    else:
        raise ConversionError("Output directory must already exist")
    resolved = path.resolve(strict=True)
    root = project_root.resolve(strict=True)
    if resolved == root or root in resolved.parents:
        raise ConversionError("Output directory must be outside the repository")
    return resolved


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    info = os.stat(path, follow_symlinks=False)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def logical_content_sha256(
    records: Mapping[str, Mapping[str, Any]], contract: dict[str, Any]
) -> str:
    """Hash dataset identities using the contract's unambiguous framing."""

    digest = hashlib.sha256()
    order = contract["manifest"]["logical_digest"]["dataset_order"]
    for name in order:
        record = records[name]
        dtype = str(record["dtype"]).encode("ascii")
        shape = [int(value) for value in record["shape"]]
        digest.update(_length_prefixed(str(name).encode("utf-8")))
        digest.update(_length_prefixed(dtype))
        digest.update(len(shape).to_bytes(8, "big"))
        for dimension in shape:
            digest.update(dimension.to_bytes(8, "big"))
        digest.update(bytes.fromhex(record["sha256"]))
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str:
    git_entry = project_root / ".git"
    if git_entry.is_dir():
        git = git_entry
    elif git_entry.is_file():
        pointer = git_entry.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise ConversionError("Repository .git file is invalid")
        git = (project_root / pointer[8:]).resolve(strict=True)
    else:
        raise ConversionError("Project root is not a Git worktree")
    common = git
    common_pointer = git / "commondir"
    if common_pointer.is_file():
        common = (git / common_pointer.read_text(encoding="utf-8").strip()).resolve(strict=True)
    head = (git / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        ref = head[5:]
        ref_path = git / ref
        if not ref_path.is_file():
            ref_path = common / ref
        if ref_path.is_file():
            value = ref_path.read_text(encoding="ascii").strip()
        else:
            packed = common / "packed-refs"
            entries = {}
            if packed.is_file():
                for line in packed.read_text(encoding="ascii").splitlines():
                    if line and not line.startswith(("#", "^")):
                        commit, name = line.split(" ", 1)
                        entries[name] = commit
            value = entries.get(ref, "")
    else:
        value = head
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ConversionError("Could not resolve a full hexadecimal project commit")
    return value


def _environment(project_root: Path) -> dict[str, str]:
    return {
        "project_commit": _git_commit(project_root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
    }


def _contract_layout(contract: dict[str, Any]) -> dict[str, Any]:
    return contract["format"]["datasets"]


def _expected_keys(contract: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (modulation, int(snr))
        for modulation in contract["ordering"]["modulation_order"]
        for snr in contract["ordering"]["snr_db_order"]
    }


def _source_content_digest(records: Mapping[tuple[str, int], bytes]) -> str:
    if len(records) > _MAX_SOURCE_HASH_RECORDS:
        raise ConversionError("Too many source cell hash records")
    digest = hashlib.sha256()
    for modulation, snr in sorted(records):
        digest.update(modulation.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(snr).encode("ascii"))
        digest.update(b"\0")
        digest.update(records[(modulation, snr)])
    return digest.hexdigest()


def _create_dataset(file: h5py.File, name: str, layout: dict[str, Any]) -> h5py.Dataset:
    dtype = np.dtype(layout["dtype"])
    kwargs: dict[str, Any] = {
        "shape": tuple(layout["shape"]),
        "dtype": dtype,
        "chunks": None if layout["chunks"] is None else tuple(layout["chunks"]),
        "compression": layout["compression"],
        "shuffle": layout["shuffle"],
        "fletcher32": layout["fletcher32"],
        "track_times": layout["track_times"],
    }
    return file.create_dataset(layout["path"], **kwargs)


class _HDF5Writer:
    def __init__(self, file: h5py.File, contract: dict[str, Any]) -> None:
        self.file = file
        self.contract = contract
        layout = _contract_layout(contract)
        self.datasets = {name: _create_dataset(file, name, layout[name]) for name in _DATASET_NAMES}
        names = [name.encode("ascii") for name in contract["ordering"]["modulation_order"]]
        self.datasets["modulation_names"][:] = np.asarray(
            names, dtype=np.dtype(layout["modulation_names"]["dtype"])
        )
        self.seen: set[tuple[str, int]] = set()
        self.cell_hashes: dict[tuple[str, int], bytes] = {}

    def observe(self, key: tuple[str, int], payload: bytes) -> None:
        if key not in _expected_keys(self.contract) or key in self.seen:
            raise ConversionError(f"Unexpected or duplicate source cell: {key!r}")
        samples_per_cell = int(self.contract["ordering"]["source_index"]["stop"])
        channels, length = (
            int(value) for value in self.contract["format"]["datasets"]["iq"]["shape"][1:]
        )
        expected_size = samples_per_cell * channels * length * 4
        if len(payload) != expected_size:
            raise ConversionError(f"Cell {key!r} has an unexpected byte length")
        values = np.frombuffer(payload, dtype=np.dtype("<f4"))
        if not bool(np.isfinite(values).all()):
            raise ConversionError(f"Cell {key!r} contains non-finite values")
        array = values.reshape(samples_per_cell, channels, length)
        row = conversion_row_index(self.contract, key[0], key[1], 0)
        rows = slice(row, row + samples_per_cell)
        self.datasets["iq"][rows] = array
        modulation_index = self.contract["ordering"]["modulation_order"].index(key[0])
        self.datasets["modulation_index"][rows] = np.full(
            samples_per_cell, modulation_index, dtype=np.dtype("|u1")
        )
        self.datasets["snr_db"][rows] = np.full(samples_per_cell, key[1], dtype=np.dtype("|i1"))
        source_start = int(self.contract["ordering"]["source_index"]["start"])
        self.datasets["source_index"][rows] = np.arange(
            source_start, source_start + samples_per_cell, dtype=np.dtype("<u2")
        )
        self.seen.add(key)
        self.cell_hashes[key] = hashlib.sha256(payload).digest()

    def finish(self) -> None:
        expected = _expected_keys(self.contract)
        if self.seen != expected:
            raise ConversionError("Static schema did not provide the complete expected cell grid")
        expected_digest = self.contract["source"]["dataset_content_sha256"]
        if _source_content_digest(self.cell_hashes) != expected_digest:
            raise ConversionError("Source dataset content digest does not match the contract")
        self.file.flush()


def _dataset_record(dataset: h5py.Dataset, digest: str) -> dict[str, Any]:
    return {
        "path": dataset.name,
        "dtype": dataset.dtype.str,
        "shape": [int(value) for value in dataset.shape],
        "sha256": digest,
    }


def _hash_dataset(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    if dataset.ndim == 0:
        chunks = [dataset[()]]
    elif dataset.shape[0] == 0:
        chunks = []
    else:
        chunks = (
            dataset[start : min(start + 4096, dataset.shape[0])]
            for start in range(0, dataset.shape[0], 4096)
        )
    for values in chunks:
        digest.update(np.asarray(values, dtype=dataset.dtype).tobytes(order="C"))
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def inspect_hdf5(
    path: Path,
    contract: dict[str, Any],
    source_cells: Mapping[tuple[str, int], bytes] | None = None,
) -> dict[str, Any]:
    """Verify HDF5 structure, labels, finite values, and canonical dataset hashes."""

    path = _regular_file(path, "HDF5 output")
    layout = _contract_layout(contract)
    expected_names = set(_DATASET_NAMES)
    with h5py.File(path, "r", libver="earliest", swmr=False) as file:
        if set(file.keys()) != expected_names:
            raise ConversionError("HDF5 root datasets do not match the conversion contract")
        if file.attrs:
            raise ConversionError("HDF5 file must not contain attributes in schema version 1")
        records: dict[str, dict[str, Any]] = {}
        for name in _DATASET_NAMES:
            link = file.get(layout[name]["path"], getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ConversionError(f"HDF5 dataset {name} must be a hard link")
            dataset = file[layout[name]["path"]]
            if not isinstance(dataset, h5py.Dataset) or dataset.attrs:
                raise ConversionError(f"HDF5 object {name} is not a plain dataset")
            if dataset.dtype.str != layout[name]["dtype"]:
                raise ConversionError(f"HDF5 dtype mismatch for {name}")
            if list(dataset.shape) != list(layout[name]["shape"]):
                raise ConversionError(f"HDF5 shape mismatch for {name}")
            expected_chunks = layout[name]["chunks"]
            if dataset.chunks != (None if expected_chunks is None else tuple(expected_chunks)):
                raise ConversionError(f"HDF5 chunk mismatch for {name}")
            if (
                dataset.compression != layout[name]["compression"]
                or dataset.shuffle != layout[name]["shuffle"]
            ):
                raise ConversionError(f"HDF5 filter mismatch for {name}")
            if dataset.fletcher32 != layout[name]["fletcher32"]:
                raise ConversionError(f"HDF5 checksum filter mismatch for {name}")
            if dataset.scaleoffset is not None or dataset.is_virtual or dataset.external:
                raise ConversionError(f"HDF5 dataset {name} uses a forbidden storage feature")
            if name == "iq":
                for start in range(0, dataset.shape[0], 4096):
                    if not bool(np.isfinite(dataset[start : start + 4096]).all()):
                        raise ConversionError("HDF5 IQ dataset contains non-finite values")
            records[name] = _dataset_record(dataset, _hash_dataset(dataset))
        names = [
            bytes(value).decode("ascii") for value in file[layout["modulation_names"]["path"]][...]
        ]
        if names != contract["ordering"]["modulation_order"]:
            raise ConversionError("HDF5 modulation names do not match the contract")
        modulation_index = file[layout["modulation_index"]["path"]][...]
        snr_db = file[layout["snr_db"]["path"]][...]
        source_index = file[layout["source_index"]["path"]][...]
        expected_modulation = np.repeat(
            np.arange(len(names), dtype=np.dtype("|u1")),
            len(contract["ordering"]["snr_db_order"])
            * int(contract["ordering"]["source_index"]["stop"]),
        )
        expected_snr = np.tile(
            np.repeat(
                np.asarray(contract["ordering"]["snr_db_order"], dtype=np.dtype("|i1")),
                int(contract["ordering"]["source_index"]["stop"]),
            ),
            len(names),
        )
        expected_source = np.tile(
            np.arange(int(contract["ordering"]["source_index"]["stop"]), dtype=np.dtype("<u2")),
            len(names) * len(contract["ordering"]["snr_db_order"]),
        )
        if not (
            np.array_equal(modulation_index, expected_modulation)
            and np.array_equal(snr_db, expected_snr)
            and np.array_equal(source_index, expected_source)
        ):
            raise ConversionError("HDF5 row metadata does not match canonical ordering")
        if source_cells is not None:
            iq = file[layout["iq"]["path"]]
            for (modulation, snr), source_digest in source_cells.items():
                row = conversion_row_index(contract, modulation, snr, 0)
                actual = hashlib.sha256(
                    np.asarray(
                        iq[row : row + int(contract["ordering"]["source_index"]["stop"])]
                    ).tobytes(order="C")
                ).digest()
                if actual != source_digest:
                    raise ConversionError(f"HDF5 IQ rows changed source cell {modulation!r}, {snr}")
    logical = logical_content_sha256(records, contract)
    return {
        "datasets": records,
        "logical_content_sha256": logical,
        "sample_count": int(layout["iq"]["shape"][0]),
        "cell_count": len(_expected_keys(contract)),
    }


def _manifest(
    *,
    contract: dict[str, Any],
    contract_path: Path,
    spec_path: Path,
    source_archive: Path,
    source_report: dict[str, Any],
    output_path: Path,
    inspection: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    output_file_sha256 = _sha256_file(output_path)
    return {
        "schema_version": _MANIFEST_VERSION,
        "contract_id": contract["contract_id"],
        "dataset_id": contract["dataset_id"],
        "source": {
            "archive_filename": source_archive.name,
            "archive_size_bytes": int(source_report["archive"]["size_bytes"]),
            "archive_sha256": source_report["archive"]["sha256"],
            "dataset_content_sha256": contract["source"]["dataset_content_sha256"],
        },
        "artifacts": {
            "hdf5_filename": contract["format"]["output_filename"],
            "manifest_filename": contract["manifest"]["filename"],
            "output_file_sha256": output_file_sha256,
            "output_logical_content_sha256": inspection["logical_content_sha256"],
        },
        "datasets": inspection["datasets"],
        "environment": _environment(project_root),
        "digests": {
            "conversion_contract_sha256": conversion_contract_sha256(contract_path),
            "dataset_spec_sha256": _sha256_file(spec_path),
            "source_archive_sha256": source_report["archive"]["sha256"],
            "source_dataset_content_sha256": contract["source"]["dataset_content_sha256"],
            "output_file_sha256": output_file_sha256,
            "output_logical_content_sha256": inspection["logical_content_sha256"],
            **{
                f"{name}_dataset_sha256": inspection["datasets"][name]["sha256"]
                for name in _DATASET_NAMES
            },
        },
        "verification": {
            "complete_grid_verified": True,
            "source_schema_verified": True,
            "hdf5_reopened_read_only": True,
            "canonical_order_verified": True,
            "numeric_values_finite_verified": True,
        },
        "security": {
            "archive_extracted": False,
            "pickle_deserialized": False,
            "pickle_globals_imported": False,
            "globals_executed": False,
            "output_absolute_path_recorded": False,
        },
    }


def _write_json_exclusive(payload: dict[str, Any], directory: Path, filename: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _publish_noreplace(temporary: Path, final: Path) -> tuple[int, int]:
    if final.exists() or final.is_symlink() or _is_reparse_point(final):
        raise ConversionError(f"Refusing to overwrite existing output: {final.name}")
    created = False
    try:
        os.link(temporary, final)
        created = True
        published = os.stat(final, follow_symlinks=False)
        temporary.unlink()
    except FileExistsError as error:
        raise ConversionError(f"Output appeared during publication: {final.name}") from error
    except BaseException:
        if created:
            final.unlink(missing_ok=True)
        raise
    return published.st_dev, published.st_ino


def _temporary_hdf5_path(directory: Path, filename: str) -> Path:
    for _ in range(16):
        candidate = directory / f".{filename}.{secrets.token_hex(16)}.tmp"
        if (
            not candidate.exists()
            and not candidate.is_symlink()
            and not _is_reparse_point(candidate)
        ):
            return candidate
    raise ConversionError("Could not allocate an unpredictable HDF5 temporary path")


def _unlink_if_identity_matches(path: Path, identity: tuple[int, int]) -> None:
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (info.st_dev, info.st_ino) == identity:
        path.unlink()


def _acquire_writer_lock(directory: Path, filename: str) -> tuple[Path, tuple[int, int]]:
    path = directory / f".{filename}.conversion.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise ConversionError("Another conversion lock already exists") from error
    try:
        payload = f"pid={os.getpid()}\n".encode("ascii")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        return path, (info.st_dev, info.st_ino)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    finally:
        with suppress(OSError):
            os.close(descriptor)


def convert_archive(
    archive_path: Path,
    spec_path: Path,
    contract_path: Path,
    output_directory: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Convert one verified archive and publish HDF5 plus manifest atomically."""

    contract = load_conversion_contract(contract_path, spec_path)
    spec = load_dataset_spec(spec_path)
    archive = _regular_file(archive_path, "Source archive")
    if archive.name != spec["archive_filename"]:
        raise ConversionError("Source archive filename does not match the dataset specification")
    if archive.stat().st_size != contract["source"]["archive_size_bytes"]:
        raise ConversionError("Source archive size does not match the conversion contract")
    output_dir = _output_directory(output_directory, project_root)
    output_path = output_dir / contract["format"]["output_filename"]
    manifest_path = output_dir / contract["manifest"]["filename"]
    if (
        output_path.exists()
        or output_path.is_symlink()
        or _is_reparse_point(output_path)
        or manifest_path.exists()
        or manifest_path.is_symlink()
        or _is_reparse_point(manifest_path)
    ):
        raise ConversionError("Final output or manifest already exists; overwrite is disabled")

    lock_path, lock_identity = _acquire_writer_lock(output_dir, output_path.name)
    source_report: dict[str, Any] = {}
    h5_temporary: Path | None = None
    manifest_temporary: Path | None = None
    published_h5_identity: tuple[int, int] | None = None
    try:
        with archive.open("rb") as stream:
            source_stat = os.fstat(stream.fileno())
            source_digest = _sha256_stream(stream)
        if source_digest != contract["source"]["archive_sha256"]:
            raise ConversionError("Source archive SHA-256 does not match the conversion contract")

        h5_temporary = _temporary_hdf5_path(output_dir, output_path.name)
        with h5py.File(
            h5_temporary, "x", libver=contract["format"]["libver"], track_order=False
        ) as file:
            writer = _HDF5Writer(file, contract)
            source_report = validate_pickle_schema_archive(
                archive, spec_path, buffer_observer=writer.observe
            )
            writer.finish()
        if (
            source_report["archive"]["sha256"] != source_digest
            or source_report["archive"]["size_bytes"] != source_stat.st_size
        ):
            raise ConversionError("Source archive changed between preflight and conversion")
        _fsync_file(h5_temporary)
        inspection = inspect_hdf5(h5_temporary, contract, writer.cell_hashes)
        source_report = {
            **source_report,
            "archive": {
                **source_report["archive"],
                "size_bytes": int(source_stat.st_size),
                "sha256": source_digest,
            },
        }
        manifest = _manifest(
            contract=contract,
            contract_path=contract_path,
            spec_path=spec_path,
            source_archive=archive,
            source_report=source_report,
            output_path=h5_temporary,
            inspection=inspection,
            project_root=project_root,
        )
        manifest_temporary = _write_json_exclusive(manifest, output_dir, manifest_path.name)
        published_h5_identity = _publish_noreplace(h5_temporary, output_path)
        h5_temporary = None
        _publish_noreplace(manifest_temporary, manifest_path)
        manifest_temporary = None
        return manifest
    except BaseException:
        if published_h5_identity is not None:
            _unlink_if_identity_matches(output_path, published_h5_identity)
        raise
    finally:
        if h5_temporary is not None:
            h5_temporary.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
        _unlink_if_identity_matches(lock_path, lock_identity)


def _load_manifest(path: Path) -> dict[str, Any]:
    path = _regular_file(path, "Manifest")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ConversionError("Manifest exceeds the configured size limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"Could not read manifest: {error}") from error
    if not isinstance(value, dict):
        raise ConversionError("Manifest root must be an object")
    return value


class _SourceVerifier:
    def __init__(self, contract: dict[str, Any]) -> None:
        self.contract = contract
        self.cells: dict[tuple[str, int], bytes] = {}

    def observe(self, key: tuple[str, int], payload: bytes) -> None:
        if key not in _expected_keys(self.contract) or key in self.cells:
            raise ConversionError(f"Unexpected or duplicate verification cell: {key!r}")
        self.cells[key] = hashlib.sha256(payload).digest()

    def finish(self) -> None:
        if set(self.cells) != _expected_keys(self.contract):
            raise ConversionError("Source verification did not observe the complete cell grid")
        if _source_content_digest(self.cells) != self.contract["source"]["dataset_content_sha256"]:
            raise ConversionError("Source dataset content digest does not match the contract")


def _validate_manifest(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    contract_path: Path,
    spec_path: Path,
    archive: Path,
    hdf5: Path,
    inspection: dict[str, Any],
) -> None:
    expected_top = {
        "schema_version",
        "contract_id",
        "dataset_id",
        "source",
        "artifacts",
        "datasets",
        "environment",
        "digests",
        "verification",
        "security",
    }
    if set(manifest) != expected_top:
        raise ConversionError("Manifest fields do not match schema version 1")
    if (
        manifest["schema_version"] != _MANIFEST_VERSION
        or manifest["contract_id"] != contract["contract_id"]
        or manifest["dataset_id"] != contract["dataset_id"]
    ):
        raise ConversionError("Manifest identity does not match the contract")
    expected_source = {
        "archive_filename": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": contract["source"]["archive_sha256"],
        "dataset_content_sha256": contract["source"]["dataset_content_sha256"],
    }
    if manifest["source"] != expected_source:
        raise ConversionError("Manifest source binding does not match the contract")
    expected_artifacts = {
        "hdf5_filename": contract["format"]["output_filename"],
        "manifest_filename": contract["manifest"]["filename"],
        "output_file_sha256": _sha256_file(hdf5),
        "output_logical_content_sha256": inspection["logical_content_sha256"],
    }
    if manifest["artifacts"] != expected_artifacts:
        raise ConversionError("Manifest artifact binding does not match verified output")
    if manifest["datasets"] != inspection["datasets"]:
        raise ConversionError("Manifest dataset records do not match verified output")
    environment = manifest["environment"]
    if (
        not isinstance(environment, dict)
        or set(environment) != set(contract["manifest"]["required_environment"])
        or not all(isinstance(value, str) and value for value in environment.values())
    ):
        raise ConversionError("Manifest environment record is incomplete")
    project_commit = environment["project_commit"]
    if len(project_commit) != 40 or any(
        character not in "0123456789abcdef" for character in project_commit
    ):
        raise ConversionError("Manifest project commit is not a full hexadecimal commit ID")
    expected_digests = {
        "conversion_contract_sha256": conversion_contract_sha256(contract_path),
        "dataset_spec_sha256": _sha256_file(spec_path),
        "source_archive_sha256": contract["source"]["archive_sha256"],
        "source_dataset_content_sha256": contract["source"]["dataset_content_sha256"],
        "output_file_sha256": expected_artifacts["output_file_sha256"],
        "output_logical_content_sha256": inspection["logical_content_sha256"],
        **{
            f"{name}_dataset_sha256": inspection["datasets"][name]["sha256"]
            for name in _DATASET_NAMES
        },
    }
    if manifest["digests"] != expected_digests or set(expected_digests) != set(
        contract["manifest"]["required_digests"]
    ):
        raise ConversionError("Manifest digest set does not match verified artifacts")
    expected_verification = {
        "complete_grid_verified": True,
        "source_schema_verified": True,
        "hdf5_reopened_read_only": True,
        "canonical_order_verified": True,
        "numeric_values_finite_verified": True,
    }
    expected_security = {
        "archive_extracted": False,
        "pickle_deserialized": False,
        "pickle_globals_imported": False,
        "globals_executed": False,
        "output_absolute_path_recorded": False,
    }
    if (
        manifest["verification"] != expected_verification
        or manifest["security"] != expected_security
    ):
        raise ConversionError("Manifest verification or security boundary is invalid")


def verify_conversion(
    hdf5_path: Path,
    manifest_path: Path,
    archive_path: Path,
    spec_path: Path,
    contract_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Verify both artifacts and their source binding without modifying them."""

    contract = load_conversion_contract(contract_path, spec_path)
    archive = _regular_file(archive_path, "Source archive")
    manifest = _load_manifest(manifest_path)
    expected_output = Path(contract["format"]["output_filename"]).name
    if (
        manifest.get("artifacts", {}).get("hdf5_filename") != expected_output
        or hdf5_path.name != expected_output
    ):
        raise ConversionError("Manifest or HDF5 filename does not match the contract")
    if manifest.get("artifacts", {}).get("manifest_filename") != manifest_path.name:
        raise ConversionError("Manifest filename binding mismatch")
    if (
        archive.stat().st_size != contract["source"]["archive_size_bytes"]
        or _sha256_file(archive) != contract["source"]["archive_sha256"]
    ):
        raise ConversionError("Source archive hash mismatch")
    source_verifier = _SourceVerifier(contract)
    source_report = validate_pickle_schema_archive(
        archive, spec_path, buffer_observer=source_verifier.observe
    )
    if (
        source_report["archive"]["size_bytes"] != contract["source"]["archive_size_bytes"]
        or source_report["archive"]["sha256"] != contract["source"]["archive_sha256"]
    ):
        raise ConversionError("Source archive changed during independent verification")
    source_verifier.finish()
    hdf5 = _regular_file(hdf5_path, "HDF5 output")
    hdf5_identity = _file_identity(hdf5)
    inspection = inspect_hdf5(hdf5, contract, source_verifier.cells)
    _validate_manifest(manifest, contract, contract_path, spec_path, archive, hdf5, inspection)
    if _file_identity(hdf5) != hdf5_identity:
        raise ConversionError("HDF5 output changed during verification")
    return {"manifest": manifest, "inspection": inspection}
