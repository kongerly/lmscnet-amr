"""No-execution inspection for legacy Python 2 protocol-0 pickle streams."""

from __future__ import annotations

import codecs
import hashlib
import os
import pickletools
import stat
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from na_lmscnet.data.provenance import (
    MAX_ARCHIVE_BYTES,
    _inspect_tar_bz2_stream,
    _sha256_stream,
    load_dataset_spec,
)

MAX_STRING_LINE_BYTES = 32 * 1024 * 1024
MAX_OPCODE_ARGUMENT_BYTES = 32 * 1024 * 1024
MAX_PICKLE_PAYLOAD_BYTES = 2 * 1024**3
MAX_LITERAL_TEXT_BYTES = 4096
MAX_GLOBAL_REFERENCE_TEXT_BYTES = 4096
MAX_PICKLE_OPCODES = 2_000_000
MAX_REPORTED_LITERALS = 10_000
MAX_REPORTED_GLOBALS = 1_000


class UnsafePickleError(ValueError):
    """Raised when a pickle stream cannot be inspected safely."""


@dataclass
class PickleScanReport:
    """Bounded metadata collected without constructing pickle objects."""

    opcode_count: int = 0
    stop_count: int = 0
    protocol_versions: set[int] = field(default_factory=set)
    opcode_counts: Counter[str] = field(default_factory=Counter)
    literal_strings: Counter[str] = field(default_factory=Counter)
    integer_values: Counter[int] = field(default_factory=Counter)
    global_references: Counter[str] = field(default_factory=Counter)
    restricted_opcodes: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "opcode_count": self.opcode_count,
            "stop_count": self.stop_count,
            "protocol_versions": sorted(self.protocol_versions),
            "opcode_counts": dict(sorted(self.opcode_counts.items())),
            "literal_strings": dict(sorted(self.literal_strings.items())),
            "integer_values": {
                str(key): value for key, value in sorted(self.integer_values.items())
            },
            "global_references": dict(sorted(self.global_references.items())),
            "restricted_opcodes": dict(sorted(self.restricted_opcodes.items())),
        }


def _readline_limited(stream: BinaryIO) -> bytes:
    data = stream.readline(MAX_STRING_LINE_BYTES + 1)
    if len(data) > MAX_STRING_LINE_BYTES:
        raise UnsafePickleError("Pickle text argument exceeds the configured size limit")
    if not data.endswith(b"\n"):
        raise UnsafePickleError("Pickle text argument is not newline terminated")
    return data


class _BoundedPickleReader:
    """Expose only bounded reads to pickletools argument readers."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise UnsafePickleError("Unbounded pickle read is not allowed")
        if size > MAX_OPCODE_ARGUMENT_BYTES:
            raise UnsafePickleError("Pickle opcode argument exceeds the configured size limit")
        return self._stream.read(size)

    def readline(self, size: int = -1) -> bytes:
        limit = MAX_STRING_LINE_BYTES + 1
        if size >= 0:
            limit = min(size, limit)
        return self._stream.readline(limit)

    def tell(self) -> int:
        return self._stream.tell()


def _read_stringnl_raw(stream: BinaryIO) -> bytes:
    data = _readline_limited(stream)[:-1]
    if len(data) < 2 or data[:1] not in {b"'", b'"'} or data[-1:] != data[:1]:
        raise UnsafePickleError("Python 2 STRING opcode has invalid quoting")
    return data[1:-1]


def _decode_literal(raw: bytes) -> str | None:
    if len(raw) > MAX_LITERAL_TEXT_BYTES:
        return None
    try:
        decoded = codecs.decode(raw, "unicode_escape")
    except (UnicodeDecodeError, ValueError):
        return None
    if not decoded.isprintable() and decoded not in {""}:
        return None
    return decoded


def _iter_safe_ops(stream: BinaryIO):
    reader = _BoundedPickleReader(stream)
    while True:
        position = reader.tell()
        code = reader.read(1)
        if not code:
            raise UnsafePickleError("Pickle stream ended before STOP")
        opcode = pickletools.code2op.get(code.decode("latin-1"))
        if opcode is None:
            raise UnsafePickleError(f"Unknown pickle opcode {code!r} at offset {position}")
        if opcode.name == "STRING":
            argument = _read_stringnl_raw(reader)
        elif opcode.arg is None:
            argument = None
        else:
            try:
                argument = opcode.arg.reader(reader)
            except (EOFError, UnicodeDecodeError, ValueError, UnicodeError) as error:
                raise UnsafePickleError(
                    f"Could not read {opcode.name} argument at offset {position}: {error}"
                ) from error
        yield opcode, argument, position
        if opcode.name == "STOP":
            return


def scan_pickle_stream(stream: BinaryIO) -> PickleScanReport:
    """Scan pickle opcodes while never invoking pickle deserialization."""

    report = PickleScanReport()
    restricted = {
        "BUILD",
        "EXT1",
        "EXT2",
        "EXT4",
        "GLOBAL",
        "INST",
        "NEWOBJ",
        "NEWOBJ_EX",
        "OBJ",
        "PERSID",
        "REDUCE",
        "STACK_GLOBAL",
    }
    for opcode, argument, _position in _iter_safe_ops(stream):
        report.opcode_count += 1
        if report.opcode_count > MAX_PICKLE_OPCODES:
            raise UnsafePickleError("Pickle stream exceeds the configured opcode limit")
        report.opcode_counts[opcode.name] += 1
        if opcode.name == "PROTO":
            report.protocol_versions.add(int(argument))
        elif opcode.name == "STRING" and isinstance(argument, bytes):
            literal = _decode_literal(argument)
            if literal is not None:
                if (
                    literal not in report.literal_strings
                    and len(report.literal_strings) >= MAX_REPORTED_LITERALS
                ):
                    raise UnsafePickleError("Pickle literal report exceeds the configured limit")
                report.literal_strings[literal] += 1
        elif opcode.name in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}:
            if isinstance(argument, int) and -1024 <= argument <= 1024:
                report.integer_values[argument] += 1
        elif opcode.name in {"GLOBAL", "INST"} and argument is not None:
            reference = str(argument).replace(" ", ".")
            if len(reference) > MAX_GLOBAL_REFERENCE_TEXT_BYTES:
                raise UnsafePickleError("Pickle global reference exceeds the configured limit")
            if (
                reference not in report.global_references
                and len(report.global_references) >= MAX_REPORTED_GLOBALS
            ):
                raise UnsafePickleError("Pickle global report exceeds the configured limit")
            report.global_references[reference] += 1
        if opcode.name in restricted:
            report.restricted_opcodes[opcode.name] += 1
        if opcode.name == "STOP":
            report.stop_count += 1
    if not report.protocol_versions:
        report.protocol_versions.add(0)
    if report.stop_count != 1:
        raise UnsafePickleError(f"Expected exactly one STOP opcode, got {report.stop_count}")
    return report


def _read_member_bytes(archive: tarfile.TarFile, name: str, max_bytes: int) -> bytes:
    member = archive.getmember(name)
    if not member.isfile() or member.size > max_bytes:
        raise UnsafePickleError(f"Archive member {name!r} is not a bounded regular file")
    stream = archive.extractfile(member)
    if stream is None:
        raise UnsafePickleError(f"Could not open archive member {name!r}")
    with stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) != member.size:
        raise UnsafePickleError(f"Archive member {name!r} is truncated")
    return payload


def inspect_pickle_archive(archive_path: Path, spec_path: Path) -> dict[str, Any]:
    """Inspect the legacy pickle member and package license without deserializing."""

    spec = load_dataset_spec(spec_path)
    if archive_path.is_symlink():
        raise ValueError("Dataset archive must be a regular file, not a symlink")
    archive_path = archive_path.resolve(strict=True)
    if archive_path.name != spec["archive_filename"] or not archive_path.is_file():
        raise ValueError("Archive path does not match the dataset specification")
    with archive_path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Dataset archive must be a regular file")
        if before.st_size <= 0 or before.st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Dataset archive size is outside the configured safety limit")
        digest = _sha256_stream(stream)
        stream.seek(0)
        members = _inspect_tar_bz2_stream(stream)
        member_names = [str(member["name"]) for member in members]
        if sorted(member_names) != ["LICENSE.TXT", "RML2016.10a_dict.pkl"]:
            raise UnsafePickleError("Archive members do not exactly match the expected payload")
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r:bz2") as archive:
            try:
                pickle_member = archive.getmember("RML2016.10a_dict.pkl")
            except KeyError as error:
                raise UnsafePickleError("Could not find the pickle member") from error
            if not pickle_member.isfile() or pickle_member.size > MAX_PICKLE_PAYLOAD_BYTES:
                raise UnsafePickleError("Pickle member exceeds the configured safety limit")
            pickle_stream = archive.extractfile(pickle_member)
            if pickle_stream is None:
                raise UnsafePickleError("Could not open the pickle member")
            with pickle_stream:
                scan = scan_pickle_stream(pickle_stream)
                if pickle_stream.read(1):
                    raise UnsafePickleError("Pickle member contains trailing data after STOP")
            if scan.protocol_versions != {0}:
                raise UnsafePickleError(
                    f"Expected legacy pickle protocol 0, got {sorted(scan.protocol_versions)}"
                )
            license_bytes = _read_member_bytes(archive, "LICENSE.TXT", 128 * 1024)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise UnsafePickleError("Dataset archive changed during inspection")
    license_text = license_bytes.decode("utf-8", errors="replace")
    expected_modulations = set(spec["expected"]["modulations"])
    expected_snr = set(spec["expected"]["snr_db"])
    observed_modulations = expected_modulations & set(scan.literal_strings)
    observed_snr = expected_snr & set(scan.integer_values)
    missing_modulations = sorted(expected_modulations - observed_modulations)
    missing_snr = sorted(expected_snr - observed_snr)
    if missing_modulations or missing_snr:
        raise UnsafePickleError(
            f"Expected grid literals missing: modulations={missing_modulations}, snr={missing_snr}"
        )
    return {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "archive": {
            "filename": archive_path.name,
            "sha256": digest,
            "pickle_member_size_bytes": pickle_member.size,
            "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
        },
        "pickle_scan": scan.as_dict(),
        "expected_grid": {
            "modulations_observed": sorted(observed_modulations),
            "snr_db_observed": sorted(observed_snr),
            "modulation_literal_counts": {
                key: scan.literal_strings[key] for key in sorted(observed_modulations)
            },
            "snr_integer_counts": {
                str(key): scan.integer_values[key] for key in sorted(observed_snr)
            },
        },
        "license": {
            "member": "LICENSE.TXT",
            "contains_cc_by_nc_sa_4": "Attribution-NonCommercial-ShareAlike 4.0" in license_text,
            "contains_deepsig": "DeepSig" in license_text,
        },
        "security": {
            "archive_extracted": False,
            "pickle_deserialized": False,
            "globals_executed": False,
        },
        "verification_scope": {
            "description": "opcode metadata and expected modulation/SNR literal presence only",
            "sample_count_verified": False,
            "dtype_verified": False,
            "shape_verified": False,
        },
    }
