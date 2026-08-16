"""Safe, metadata-only inventory helpers for dataset archives."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import yaml

BUFFER_SIZE = 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024**3
MAX_MEMBERS = 64
MAX_DECLARED_CONTENT_BYTES = 4 * 1024**3
MAX_MEMBER_NAME_LENGTH = 255
MAX_SPEC_BYTES = 64 * 1024


class UnsafeArchiveError(ValueError):
    """Raised when archive metadata violates the ingestion boundary."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest without loading the file into memory."""

    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    while chunk := stream.read(BUFFER_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _validated_member_name(name: str) -> str:
    if len(name) > MAX_MEMBER_NAME_LENGTH:
        raise UnsafeArchiveError(f"Archive member name exceeds {MAX_MEMBER_NAME_LENGTH} characters")
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or path.is_absolute():
        raise UnsafeArchiveError(f"Archive member has an unsafe absolute path: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveError(f"Archive member has an unsafe path component: {name!r}")
    if path.parts[0].endswith(":"):
        raise UnsafeArchiveError(f"Archive member has a drive-qualified path: {name!r}")
    return path.as_posix()


def inspect_tar_bz2(path: Path) -> list[dict[str, int | str]]:
    """Inspect declared tar metadata without extraction or payload deserialization."""

    with path.open("rb") as stream:
        return _inspect_tar_bz2_stream(stream)


def _inspect_tar_bz2_stream(stream: BinaryIO) -> list[dict[str, int | str]]:
    members: list[dict[str, int | str]] = []
    declared_content_bytes = 0
    try:
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r:bz2") as archive:
            for index, member in enumerate(archive, start=1):
                if index > MAX_MEMBERS:
                    raise UnsafeArchiveError(f"Archive contains more than {MAX_MEMBERS} members")
                name = _validated_member_name(member.name)
                if member.isdir():
                    member_type = "directory"
                elif member.isfile():
                    member_type = "file"
                    declared_content_bytes += member.size
                    if declared_content_bytes > MAX_DECLARED_CONTENT_BYTES:
                        raise UnsafeArchiveError(
                            "Archive declares more uncompressed content than the configured limit"
                        )
                else:
                    raise UnsafeArchiveError(
                        f"Archive member type is not allowed: {name!r} ({member.type!r})"
                    )
                members.append({"name": name, "size_bytes": member.size, "type": member_type})
    except tarfile.TarError as error:
        raise UnsafeArchiveError(f"Invalid bzip2-compressed tar archive: {error}") from error

    if not members or not any(member["type"] == "file" for member in members):
        raise UnsafeArchiveError("Archive does not contain a regular file")
    return members


def load_dataset_spec(path: Path) -> dict[str, Any]:
    """Load and minimally validate the repository-controlled dataset specification."""

    if path.stat().st_size > MAX_SPEC_BYTES:
        raise ValueError(f"Dataset specification exceeds {MAX_SPEC_BYTES} bytes")
    with path.open(encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict):
        raise ValueError("Dataset specification must be a mapping")
    required = {"schema_version", "dataset_id", "archive_filename", "official_page"}
    missing = sorted(required - spec.keys())
    if missing:
        raise ValueError(f"Dataset specification is missing fields: {missing}")
    if spec["schema_version"] != 1:
        raise ValueError(f"Unsupported dataset specification schema: {spec['schema_version']!r}")
    return spec


def build_archive_inventory(archive_path: Path, spec_path: Path) -> dict[str, Any]:
    """Build a reproducible, path-redacted inventory for an official dataset archive."""

    if archive_path.is_symlink():
        raise ValueError("Dataset archive must be a regular file, not a symlink")
    archive_path = archive_path.resolve(strict=True)
    if not archive_path.is_file():
        raise ValueError("Dataset archive must be a regular file")
    spec = load_dataset_spec(spec_path)
    expected_filename = spec["archive_filename"]
    if archive_path.name != expected_filename:
        raise ValueError(
            f"Expected archive filename {expected_filename!r}, got {archive_path.name!r}"
        )

    with archive_path.open("rb") as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("Dataset archive must be a regular file")
        archive_size = file_stat.st_size
        if archive_size <= 0 or archive_size > MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"Dataset archive size must be between 1 and {MAX_ARCHIVE_BYTES} bytes"
            )
        archive_sha256 = _sha256_stream(stream)
        members = _inspect_tar_bz2_stream(stream)

    return {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "source_page": spec["official_page"],
        "archive": {
            "filename": archive_path.name,
            "size_bytes": archive_size,
            "sha256": archive_sha256,
        },
        "members": members,
        "security": {
            "archive_extracted": False,
            "payload_deserialized": False,
            "absolute_source_path_recorded": False,
        },
    }


def write_inventory(inventory: dict[str, Any], output_path: Path) -> None:
    """Atomically write an inventory without exposing a partial JSON file."""

    if output_path.is_symlink():
        raise ValueError("Inventory output must not be a symlink")
    output_path = output_path.absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(inventory, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
