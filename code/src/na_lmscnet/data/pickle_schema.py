"""Static schema validation for the legacy RADIOML 2016.10A pickle."""

from __future__ import annotations

import codecs
import os
import stat
import tarfile
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from na_lmscnet.data.pickle_safety import (
    MAX_PICKLE_PAYLOAD_BYTES,
    UnsafePickleError,
    _iter_safe_ops,
)
from na_lmscnet.data.provenance import (
    MAX_ARCHIVE_BYTES,
    _inspect_tar_bz2_stream,
    _sha256_stream,
    load_dataset_spec,
)

MAX_SCHEMA_OPCODES = 20_000
MAX_SCHEMA_STACK_ITEMS = 256
MAX_SCHEMA_MEMO_ITEMS = 2_000
MAX_SCHEMA_TUPLE_ITEMS = 16
MAX_SCHEMA_CELLS = 4_096
MAX_DECODED_STRING_BYTES = 16 * 1024 * 1024
MAX_RETAINED_STRING_BYTES = 4_096

ALLOWED_SCHEMA_OPCODES = {
    "BUILD",
    "DICT",
    "GET",
    "GLOBAL",
    "INT",
    "MARK",
    "NONE",
    "PUT",
    "REDUCE",
    "SETITEM",
    "STOP",
    "STRING",
    "TUPLE",
}
ALLOWED_GLOBALS = {
    "numpy.core.multiarray._reconstruct",
    "numpy.dtype",
    "numpy.ndarray",
}


@dataclass(frozen=True)
class _GlobalReference:
    name: str


@dataclass
class _ByteString:
    value: bytes | None
    size_bytes: int
    raw: bytes | None = None


@dataclass
class _DTypeMetadata:
    code: str
    byte_order: str | None = None
    item_size: int | None = None
    built: bool = False


@dataclass
class _ArrayMetadata:
    shape: tuple[int, ...] | None = None
    dtype: _DTypeMetadata | None = None
    fortran_order: bool | None = None
    buffer_size_bytes: int | None = None
    raw_buffer: bytes | None = None
    built: bool = False


_MARK = object()


def _decode_python2_string(raw: bytes, *, retain_raw: bool = False) -> _ByteString:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            decoded, consumed = codecs.escape_decode(raw)
    except (DeprecationWarning, ValueError, UnicodeDecodeError) as error:
        raise UnsafePickleError(f"Invalid Python 2 STRING escape sequence: {error}") from error
    if consumed != len(raw):
        raise UnsafePickleError("Python 2 STRING was not decoded completely")
    if len(decoded) > MAX_DECODED_STRING_BYTES:
        raise UnsafePickleError("Decoded pickle string exceeds the configured size limit")
    retained = decoded if len(decoded) <= MAX_RETAINED_STRING_BYTES else None
    return _ByteString(value=retained, size_bytes=len(decoded), raw=decoded if retain_raw else None)


def _small_ascii(value: object, field: str) -> str:
    if not isinstance(value, _ByteString) or value.value is None:
        raise UnsafePickleError(f"{field} must be a bounded byte string")
    try:
        return value.value.decode("ascii")
    except UnicodeDecodeError as error:
        raise UnsafePickleError(f"{field} must contain ASCII text") from error


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsafePickleError(f"{field} must be an integer")
    return value


def _pop_mark(stack: list[object]) -> list[object]:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is _MARK:
            items = stack[index + 1 :]
            del stack[index:]
            return items
    raise UnsafePickleError("Pickle stack operation is missing MARK")


class _StaticSchemaInterpreter:
    """Interpret a strict metadata-only subset of pickle protocol 0."""

    def __init__(
        self,
        buffer_observer: Callable[[tuple[str, int], bytes], None] | None = None,
    ) -> None:
        self.stack: list[object] = []
        self.memo: dict[int, object] = {}
        self.opcode_counts: Counter[str] = Counter()
        self.buffer_observer = buffer_observer

    def _push(self, value: object) -> None:
        self.stack.append(value)
        if len(self.stack) > MAX_SCHEMA_STACK_ITEMS:
            raise UnsafePickleError("Pickle schema stack exceeds the configured limit")

    def _reduce(self) -> None:
        if len(self.stack) < 2:
            raise UnsafePickleError("REDUCE requires a callable symbol and argument tuple")
        arguments = self.stack.pop()
        function = self.stack.pop()
        if not isinstance(function, _GlobalReference) or not isinstance(arguments, tuple):
            raise UnsafePickleError("REDUCE operands do not match the allowed symbolic form")

        if function.name == "numpy.core.multiarray._reconstruct":
            expected_ndarray = _GlobalReference("numpy.ndarray")
            if (
                len(arguments) != 3
                or arguments[0] != expected_ndarray
                or arguments[1] != (0,)
                or _small_ascii(arguments[2], "ndarray subtype") != "b"
            ):
                raise UnsafePickleError("ndarray reconstruction metadata is not allowed")
            self._push(_ArrayMetadata())
            return

        if function.name == "numpy.dtype":
            if (
                len(arguments) != 3
                or _integer(arguments[1], "dtype alignment flag") != 0
                or _integer(arguments[2], "dtype copy flag") != 1
            ):
                raise UnsafePickleError("dtype reconstruction metadata is not allowed")
            code = _small_ascii(arguments[0], "dtype code")
            if code != "f4":
                raise UnsafePickleError(f"Only float32 dtype code 'f4' is allowed, got {code!r}")
            self._push(_DTypeMetadata(code=code))
            return

        raise UnsafePickleError(f"REDUCE target is not allowed: {function.name!r}")

    def _build(self) -> None:
        if len(self.stack) < 2:
            raise UnsafePickleError("BUILD requires an instance and state")
        state = self.stack.pop()
        instance = self.stack[-1]
        if not isinstance(state, tuple):
            raise UnsafePickleError("BUILD state must be a tuple")

        if isinstance(instance, _DTypeMetadata):
            if instance.built:
                raise UnsafePickleError("dtype metadata cannot be built more than once")
            if len(state) != 8:
                raise UnsafePickleError("dtype state must contain eight fields")
            version = _integer(state[0], "dtype state version")
            byte_order = _small_ascii(state[1], "dtype byte order")
            if (
                version != 3
                or byte_order != "<"
                or state[2:5] != (None, None, None)
                or tuple(_integer(item, "dtype state integer") for item in state[5:]) != (-1, -1, 0)
            ):
                raise UnsafePickleError("dtype state does not describe little-endian float32")
            instance.byte_order = byte_order
            instance.item_size = 4
            instance.built = True
            return

        if isinstance(instance, _ArrayMetadata):
            if instance.built:
                raise UnsafePickleError("ndarray metadata cannot be built more than once")
            if len(state) != 5:
                raise UnsafePickleError("ndarray state must contain five fields")
            version = _integer(state[0], "ndarray state version")
            shape = state[1]
            dtype = state[2]
            fortran_order = state[3]
            buffer = state[4]
            if version != 1:
                raise UnsafePickleError(f"Unsupported ndarray state version: {version}")
            if not isinstance(shape, tuple) or not shape:
                raise UnsafePickleError("ndarray shape must be a non-empty tuple")
            normalized_shape = tuple(_integer(item, "ndarray dimension") for item in shape)
            if any(dimension <= 0 for dimension in normalized_shape):
                raise UnsafePickleError("ndarray dimensions must be positive")
            if not isinstance(dtype, _DTypeMetadata) or not dtype.built or dtype.item_size != 4:
                raise UnsafePickleError("ndarray dtype must be built float32 metadata")
            if fortran_order is not False:
                raise UnsafePickleError("Only C-contiguous ndarray metadata is allowed")
            if not isinstance(buffer, _ByteString):
                raise UnsafePickleError("ndarray state must contain an inline byte buffer")
            expected_bytes = dtype.item_size
            for dimension in normalized_shape:
                expected_bytes *= dimension
            if buffer.size_bytes != expected_bytes:
                raise UnsafePickleError(
                    f"ndarray buffer has {buffer.size_bytes} bytes, expected {expected_bytes}"
                )
            instance.shape = normalized_shape
            instance.dtype = dtype
            instance.fortran_order = fortran_order
            instance.buffer_size_bytes = buffer.size_bytes
            instance.raw_buffer = buffer.raw
            buffer.raw = None
            instance.built = True
            return

        raise UnsafePickleError("BUILD target is not an allowed symbolic object")

    def _setitem(self) -> None:
        if len(self.stack) < 3 or not isinstance(self.stack[-3], dict):
            raise UnsafePickleError("SETITEM target must be the root dictionary")
        value = self.stack.pop()
        key = self.stack.pop()
        target = self.stack[-1]
        if not isinstance(key, tuple) or len(key) != 2:
            raise UnsafePickleError("Dataset dictionary key must be a two-item tuple")
        modulation = _small_ascii(key[0], "modulation key")
        snr = _integer(key[1], "SNR key")
        normalized_key = (modulation, snr)
        if not isinstance(value, _ArrayMetadata) or not value.built:
            raise UnsafePickleError("Dataset dictionary value must be built ndarray metadata")
        if normalized_key in target:
            raise UnsafePickleError(f"Duplicate dataset cell: {normalized_key!r}")
        if len(target) >= MAX_SCHEMA_CELLS:
            raise UnsafePickleError("Dataset dictionary exceeds the configured cell limit")
        if self.buffer_observer is not None:
            if value.raw_buffer is None:
                raise UnsafePickleError("Numeric audit buffer was not retained")
            self.buffer_observer(normalized_key, value.raw_buffer)
            value.raw_buffer = None
        target[normalized_key] = value

    def run(self, stream: BinaryIO) -> dict[tuple[str, int], _ArrayMetadata]:
        for index, (opcode, argument, position) in enumerate(_iter_safe_ops(stream), start=1):
            if index > MAX_SCHEMA_OPCODES:
                raise UnsafePickleError("Pickle schema exceeds the configured opcode limit")
            name = opcode.name
            if name not in ALLOWED_SCHEMA_OPCODES:
                raise UnsafePickleError(f"Opcode {name!r} is not allowed at offset {position}")
            self.opcode_counts[name] += 1

            if name == "MARK":
                self._push(_MARK)
            elif name == "DICT":
                if _pop_mark(self.stack):
                    raise UnsafePickleError("Only an initially empty dictionary is allowed")
                self._push({})
            elif name == "STRING":
                if not isinstance(argument, bytes):
                    raise UnsafePickleError("STRING opcode did not yield raw bytes")
                self._push(
                    _decode_python2_string(argument, retain_raw=self.buffer_observer is not None)
                )
            elif name == "INT":
                if not isinstance(argument, (bool, int)):
                    raise UnsafePickleError("INT opcode did not yield an integer")
                self._push(argument)
            elif name == "NONE":
                self._push(None)
            elif name == "TUPLE":
                items = _pop_mark(self.stack)
                if len(items) > MAX_SCHEMA_TUPLE_ITEMS:
                    raise UnsafePickleError("Pickle tuple exceeds the configured item limit")
                self._push(tuple(items))
            elif name == "GLOBAL":
                reference = str(argument).replace(" ", ".")
                if reference not in ALLOWED_GLOBALS:
                    raise UnsafePickleError(f"GLOBAL reference is not allowed: {reference!r}")
                self._push(_GlobalReference(reference))
            elif name == "PUT":
                memo_index = _integer(argument, "memo index")
                if not self.stack:
                    raise UnsafePickleError("PUT requires a stack value")
                if memo_index < 0 or memo_index >= MAX_SCHEMA_MEMO_ITEMS:
                    raise UnsafePickleError("Memo index exceeds the configured limit")
                if memo_index in self.memo:
                    raise UnsafePickleError(f"Memo index is assigned twice: {memo_index}")
                self.memo[memo_index] = self.stack[-1]
                if len(self.memo) > MAX_SCHEMA_MEMO_ITEMS:
                    raise UnsafePickleError("Pickle memo exceeds the configured item limit")
            elif name == "GET":
                memo_index = _integer(argument, "memo index")
                if memo_index not in self.memo:
                    raise UnsafePickleError(f"GET references an unknown memo index: {memo_index}")
                self._push(self.memo[memo_index])
            elif name == "REDUCE":
                self._reduce()
            elif name == "BUILD":
                self._build()
            elif name == "SETITEM":
                self._setitem()
            elif name == "STOP":
                if len(self.stack) != 1 or not isinstance(self.stack[0], dict):
                    raise UnsafePickleError("STOP must leave exactly one dataset dictionary")
                return self.stack[0]

        raise UnsafePickleError("Pickle schema ended without STOP")


def _validated_expected_schema(spec: dict[str, Any]) -> dict[str, Any]:
    expected = spec.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("Dataset specification must define an expected mapping")
    modulations = expected.get("modulations")
    snr_values = expected.get("snr_db")
    sample_shape = expected.get("sample_shape")
    samples_per_cell = expected.get("samples_per_cell")
    total_samples = expected.get("total_samples")
    dtype = expected.get("dtype")
    if (
        not isinstance(modulations, list)
        or not modulations
        or any(not isinstance(value, str) or not value for value in modulations)
        or len(set(modulations)) != len(modulations)
    ):
        raise ValueError("Expected modulations must be a non-empty unique string list")
    if (
        not isinstance(snr_values, list)
        or not snr_values
        or any(isinstance(value, bool) or not isinstance(value, int) for value in snr_values)
        or len(set(snr_values)) != len(snr_values)
    ):
        raise ValueError("Expected SNR values must be a non-empty unique integer list")
    if (
        not isinstance(sample_shape, list)
        or len(sample_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in sample_shape
        )
    ):
        raise ValueError("Expected sample shape must contain two positive integers")
    if (
        isinstance(samples_per_cell, bool)
        or not isinstance(samples_per_cell, int)
        or samples_per_cell <= 0
    ):
        raise ValueError("Expected samples per cell must be a positive integer")
    calculated_total = len(modulations) * len(snr_values) * samples_per_cell
    expected_cell_count = len(modulations) * len(snr_values)
    if expected_cell_count > MAX_SCHEMA_CELLS:
        raise ValueError(f"Expected grid exceeds the configured cell limit: {expected_cell_count}")
    expected_buffer_bytes = 4 * samples_per_cell
    for dimension in sample_shape:
        expected_buffer_bytes *= dimension
    if expected_buffer_bytes > MAX_DECODED_STRING_BYTES:
        raise ValueError(
            f"Expected cell buffer exceeds the configured size limit: {expected_buffer_bytes}"
        )
    if isinstance(total_samples, bool) or not isinstance(total_samples, int) or total_samples <= 0:
        raise ValueError("Expected total samples must be a positive integer")
    if total_samples != calculated_total:
        raise ValueError(
            f"Expected total samples must equal the complete grid size: {calculated_total}"
        )
    if dtype != "float32":
        raise ValueError("Static schema validation currently requires expected dtype float32")
    return {
        "modulations": modulations,
        "snr_db": snr_values,
        "sample_shape": sample_shape,
        "samples_per_cell": samples_per_cell,
        "total_samples": total_samples,
        "dtype": dtype,
        "buffer_bytes_per_cell": expected_buffer_bytes,
    }


def validate_pickle_schema_stream(
    stream: BinaryIO,
    spec: dict[str, Any],
    *,
    buffer_observer: Callable[[tuple[str, int], bytes], None] | None = None,
) -> dict[str, Any]:
    """Validate the complete dictionary and ndarray metadata without object construction."""

    expected = _validated_expected_schema(spec)
    interpreter = _StaticSchemaInterpreter(buffer_observer=buffer_observer)
    cells = interpreter.run(stream)
    if stream.read(1):
        raise UnsafePickleError("Pickle stream contains trailing data after STOP")
    expected_keys = {
        (modulation, snr) for modulation in expected["modulations"] for snr in expected["snr_db"]
    }
    observed_keys = set(cells)
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise UnsafePickleError(
            f"Dataset grid mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
        )

    array_shape = (expected["samples_per_cell"], *expected["sample_shape"])
    buffer_bytes_per_cell = expected["buffer_bytes_per_cell"]
    array_ids: set[int] = set()
    for key, array in cells.items():
        if array.shape != array_shape:
            raise UnsafePickleError(
                f"Dataset cell {key!r} has shape {array.shape}, expected {array_shape}"
            )
        if (
            array.dtype is None
            or not array.dtype.built
            or array.dtype.code != "f4"
            or array.dtype.byte_order != "<"
            or array.dtype.item_size != 4
        ):
            raise UnsafePickleError(f"Dataset cell {key!r} is not little-endian float32")
        if array.fortran_order is not False:
            raise UnsafePickleError(f"Dataset cell {key!r} is not C-contiguous")
        if array.buffer_size_bytes != buffer_bytes_per_cell:
            raise UnsafePickleError(f"Dataset cell {key!r} has an invalid buffer size")
        array_ids.add(id(array))
    if len(array_ids) != len(cells):
        raise UnsafePickleError("Dataset cells must not alias the same ndarray metadata object")

    return {
        "pickle_protocol": 0,
        "container": "dict",
        "cell_count": len(cells),
        "modulations": sorted(expected["modulations"]),
        "snr_db": sorted(expected["snr_db"]),
        "samples_per_cell": expected["samples_per_cell"],
        "total_samples": expected["total_samples"],
        "array_shape": list(array_shape),
        "sample_shape": expected["sample_shape"],
        "dtype": expected["dtype"],
        "dtype_encoding": "<f4",
        "memory_order": "C",
        "array_buffer_bytes_per_cell": buffer_bytes_per_cell,
        "total_array_buffer_bytes": buffer_bytes_per_cell * len(cells),
        "opcode_count": sum(interpreter.opcode_counts.values()),
        "opcode_counts": dict(sorted(interpreter.opcode_counts.items())),
        "allowed_globals": sorted(ALLOWED_GLOBALS),
    }


def validate_pickle_schema_archive(
    archive_path: Path,
    spec_path: Path,
    *,
    buffer_observer: Callable[[tuple[str, int], bytes], None] | None = None,
) -> dict[str, Any]:
    """Validate the archive payload schema through the static protocol-0 interpreter."""

    spec = load_dataset_spec(spec_path)
    if archive_path.is_symlink():
        raise ValueError("Dataset archive must be a regular file, not a symlink")
    archive_path = archive_path.resolve(strict=True)
    if archive_path.name != spec["archive_filename"] or not archive_path.is_file():
        raise ValueError("Archive path does not match the dataset specification")

    try:
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
                pickle_member = archive.getmember("RML2016.10a_dict.pkl")
                if not pickle_member.isfile() or pickle_member.size > MAX_PICKLE_PAYLOAD_BYTES:
                    raise UnsafePickleError("Pickle member exceeds the configured safety limit")
                pickle_stream = archive.extractfile(pickle_member)
                if pickle_stream is None:
                    raise UnsafePickleError("Could not open the pickle member")
                with pickle_stream:
                    schema = validate_pickle_schema_stream(
                        pickle_stream,
                        spec,
                        buffer_observer=buffer_observer,
                    )
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ):
                raise UnsafePickleError("Dataset archive changed during schema validation")
    except (KeyError, tarfile.TarError) as error:
        raise UnsafePickleError(f"Could not validate pickle archive schema: {error}") from error

    return {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "archive": {
            "filename": archive_path.name,
            "size_bytes": before.st_size,
            "sha256": digest,
            "pickle_member_size_bytes": pickle_member.size,
        },
        "validated_schema": schema,
        "verification": {
            "complete_grid_verified": True,
            "sample_count_verified": True,
            "dtype_verified": True,
            "shape_verified": True,
            "buffer_sizes_verified": True,
            "numeric_values_inspected": False,
        },
        "security": {
            "archive_extracted": False,
            "pickle_deserialized": False,
            "pickle_globals_imported": False,
            "globals_executed": False,
            "static_interpreter": True,
        },
    }
