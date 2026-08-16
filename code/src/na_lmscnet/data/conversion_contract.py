"""Validated contract for deterministic RadioML HDF5 conversion."""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from na_lmscnet.data.provenance import load_dataset_spec

MAX_CONVERSION_CONTRACT_BYTES = 64 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_DATASET_NAMES = (
    "iq",
    "modulation_index",
    "snr_db",
    "source_index",
    "modulation_names",
)
_REQUIRED_DIGESTS = {
    "conversion_contract_sha256",
    "dataset_spec_sha256",
    "source_archive_sha256",
    "source_dataset_content_sha256",
    "output_file_sha256",
    "output_logical_content_sha256",
    *{f"{name}_dataset_sha256" for name in _DATASET_NAMES},
}
_REQUIRED_ENVIRONMENT = {"project_commit", "python", "numpy", "h5py", "hdf5"}


class ConversionContractError(ValueError):
    """Raised when the repository conversion contract is inconsistent or unsafe."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConversionContractError(f"{field} must be a string-keyed mapping")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    keys = set(mapping)
    if keys != expected:
        raise ConversionContractError(
            f"{field} fields differ: missing={sorted(expected - keys)}, "
            f"unexpected={sorted(keys - expected)}"
        )


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConversionContractError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ConversionContractError(f"{field} must be at least {minimum}")
    return result


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConversionContractError(f"{field} must be a non-empty trimmed string")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ConversionContractError(f"{field} must be a list")
    result = [_string(item, f"{field} item") for item in value]
    if len(result) != len(set(result)):
        raise ConversionContractError(f"{field} must not contain duplicates")
    return result


def _integer_list(value: object, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ConversionContractError(f"{field} must be a list")
    return [_integer(item, f"{field} item") for item in value]


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConversionContractError(f"{field} must be a boolean")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ConversionContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _basename(value: object, field: str, suffix: str) -> str:
    filename = _string(value, field)
    if (
        PurePosixPath(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ConversionContractError(f"{field} must be a path-free filename")
    if not filename.endswith(suffix):
        raise ConversionContractError(f"{field} must end with {suffix!r}")
    return filename


def _validate_dataset_layout(
    datasets: dict[str, Any],
    *,
    total_samples: int,
    sample_shape: list[int],
    modulation_count: int,
    modulation_width: int,
) -> None:
    _exact_keys(datasets, set(_DATASET_NAMES), "format.datasets")
    expected = {
        "iq": ("/iq", "<f4", [total_samples, *sample_shape]),
        "modulation_index": ("/modulation_index", "|u1", [total_samples]),
        "snr_db": ("/snr_db", "|i1", [total_samples]),
        "source_index": ("/source_index", "<u2", [total_samples]),
        "modulation_names": (
            "/modulation_names",
            f"|S{modulation_width}",
            [modulation_count],
        ),
    }
    fields = {
        "path",
        "dtype",
        "shape",
        "chunks",
        "compression",
        "shuffle",
        "fletcher32",
        "track_times",
    }
    for name in _DATASET_NAMES:
        layout = _mapping(datasets[name], f"format.datasets.{name}")
        _exact_keys(layout, fields, f"format.datasets.{name}")
        expected_path, expected_dtype, expected_shape = expected[name]
        if layout["path"] != expected_path:
            raise ConversionContractError(f"format.datasets.{name}.path must be {expected_path!r}")
        if layout["dtype"] != expected_dtype:
            raise ConversionContractError(
                f"format.datasets.{name}.dtype must be {expected_dtype!r}"
            )
        shape = _integer_list(layout["shape"], f"format.datasets.{name}.shape")
        if shape != expected_shape:
            raise ConversionContractError(
                f"format.datasets.{name}.shape must be {expected_shape!r}"
            )
        chunks_value = layout["chunks"]
        if chunks_value is None:
            chunks = None
        else:
            chunks = _integer_list(chunks_value, f"format.datasets.{name}.chunks")
            if len(chunks) != len(shape) or any(
                chunk < 1 or chunk > dimension
                for chunk, dimension in zip(chunks, shape, strict=True)
            ):
                raise ConversionContractError(
                    f"format.datasets.{name}.chunks must fit the dataset shape"
                )
        if name == "modulation_names" and chunks is not None:
            raise ConversionContractError("modulation_names must use contiguous storage")
        if name != "modulation_names" and chunks is None:
            raise ConversionContractError(f"format.datasets.{name} must be chunked")
        if layout["compression"] is not None:
            raise ConversionContractError(
                f"format.datasets.{name}.compression must be null in schema version 1"
            )
        if _boolean(layout["shuffle"], f"format.datasets.{name}.shuffle"):
            raise ConversionContractError(
                f"format.datasets.{name}.shuffle must be false without compression"
            )
        fletcher32 = _boolean(layout["fletcher32"], f"format.datasets.{name}.fletcher32")
        if fletcher32 != (chunks is not None):
            raise ConversionContractError(
                f"format.datasets.{name}.fletcher32 must match chunked storage"
            )
        if _boolean(layout["track_times"], f"format.datasets.{name}.track_times"):
            raise ConversionContractError(f"format.datasets.{name}.track_times must be false")


def _validate_contract(contract: dict[str, Any], dataset_spec: dict[str, Any]) -> None:
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "dataset_id",
            "source",
            "format",
            "ordering",
            "sample_id",
            "manifest",
            "writer",
        },
        "contract",
    )
    if contract["schema_version"] != 1:
        raise ConversionContractError("Unsupported conversion contract schema version")
    if _string(contract["contract_id"], "contract_id") != "radioml_2016_10a_hdf5_v1":
        raise ConversionContractError("Unexpected conversion contract identifier")
    if contract["dataset_id"] != dataset_spec["dataset_id"]:
        raise ConversionContractError(
            "Conversion contract dataset_id does not match the dataset spec"
        )

    expected = _mapping(dataset_spec.get("expected"), "dataset spec expected")
    modulations = _string_list(expected.get("modulations"), "dataset spec modulations")
    snr_values = _integer_list(expected.get("snr_db"), "dataset spec snr_db")
    sample_shape = _integer_list(expected.get("sample_shape"), "dataset spec sample_shape")
    samples_per_cell = _integer(
        expected.get("samples_per_cell"), "dataset spec samples_per_cell", minimum=1
    )
    total_samples = _integer(expected.get("total_samples"), "dataset spec total_samples", minimum=1)
    if total_samples != len(modulations) * len(snr_values) * samples_per_cell:
        raise ConversionContractError("Dataset spec total_samples is inconsistent with its grid")

    source = _mapping(contract["source"], "source")
    _exact_keys(
        source,
        {
            "archive_filename",
            "archive_size_bytes",
            "archive_sha256",
            "dataset_content_sha256",
        },
        "source",
    )
    if source["archive_filename"] != dataset_spec["archive_filename"]:
        raise ConversionContractError("Source archive filename does not match the dataset spec")
    _integer(source["archive_size_bytes"], "source.archive_size_bytes", minimum=1)
    _sha256(source["archive_sha256"], "source.archive_sha256")
    _sha256(source["dataset_content_sha256"], "source.dataset_content_sha256")

    format_spec = _mapping(contract["format"], "format")
    _exact_keys(
        format_spec,
        {"name", "library", "output_filename", "libver", "datasets"},
        "format",
    )
    if (format_spec["name"], format_spec["library"], format_spec["libver"]) != (
        "hdf5",
        "h5py",
        "earliest",
    ):
        raise ConversionContractError("Format must be HDF5 via h5py with libver='earliest'")
    output_filename = _basename(format_spec["output_filename"], "format.output_filename", ".h5")
    modulation_width = max(len(name.encode("ascii")) for name in modulations)
    _validate_dataset_layout(
        _mapping(format_spec["datasets"], "format.datasets"),
        total_samples=total_samples,
        sample_shape=sample_shape,
        modulation_count=len(modulations),
        modulation_width=modulation_width,
    )

    ordering = _mapping(contract["ordering"], "ordering")
    _exact_keys(
        ordering,
        {"dimensions", "modulation_order", "snr_db_order", "source_index"},
        "ordering",
    )
    if ordering["dimensions"] != ["modulation", "snr_db", "source_index"]:
        raise ConversionContractError("ordering.dimensions is not the canonical row order")
    if _string_list(ordering["modulation_order"], "ordering.modulation_order") != modulations:
        raise ConversionContractError("Modulation order must exactly match the dataset spec")
    if _integer_list(ordering["snr_db_order"], "ordering.snr_db_order") != snr_values:
        raise ConversionContractError("SNR order must exactly match the dataset spec")
    source_index = _mapping(ordering["source_index"], "ordering.source_index")
    _exact_keys(source_index, {"start", "stop", "step"}, "ordering.source_index")
    if (source_index["start"], source_index["stop"], source_index["step"]) != (
        0,
        samples_per_cell,
        1,
    ):
        raise ConversionContractError("Source index range must cover each cell exactly once")

    sample_id = _mapping(contract["sample_id"], "sample_id")
    _exact_keys(
        sample_id,
        {"scheme", "separator", "fields", "snr_format", "source_index_width"},
        "sample_id",
    )
    if sample_id["scheme"] != "source-coordinate-v1":
        raise ConversionContractError("Unsupported sample_id scheme")
    separator = _string(sample_id["separator"], "sample_id.separator")
    if len(separator) != 1 or any(
        separator in value for value in [dataset_spec["dataset_id"], *modulations]
    ):
        raise ConversionContractError("sample_id separator must be one collision-free character")
    if sample_id["fields"] != ["dataset_id", "modulation", "snr_db", "source_index"]:
        raise ConversionContractError("sample_id fields must use canonical source coordinates")
    if sample_id["snr_format"] != "signed-width-3":
        raise ConversionContractError("sample_id SNR format must be signed-width-3")
    width = _integer(sample_id["source_index_width"], "sample_id.source_index_width", minimum=1)
    if width < len(str(samples_per_cell - 1)):
        raise ConversionContractError("sample_id source index width is too small")

    manifest = _mapping(contract["manifest"], "manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "filename",
            "hash_algorithm",
            "absolute_paths",
            "required_digests",
            "required_environment",
            "logical_digest",
        },
        "manifest",
    )
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise ConversionContractError("Manifest must use schema version 1 and SHA-256")
    manifest_filename = _basename(
        manifest["filename"], "manifest.filename", ".conversion-manifest.json"
    )
    if manifest_filename == output_filename:
        raise ConversionContractError("Manifest and HDF5 filenames must differ")
    if _boolean(manifest["absolute_paths"], "manifest.absolute_paths"):
        raise ConversionContractError("Manifest must redact absolute paths")
    if (
        set(_string_list(manifest["required_digests"], "manifest.required_digests"))
        != _REQUIRED_DIGESTS
    ):
        raise ConversionContractError("Manifest digest requirements are incomplete")
    if (
        set(_string_list(manifest["required_environment"], "manifest.required_environment"))
        != _REQUIRED_ENVIRONMENT
    ):
        raise ConversionContractError("Manifest environment requirements are incomplete")
    logical_digest = _mapping(manifest["logical_digest"], "manifest.logical_digest")
    _exact_keys(
        logical_digest,
        {
            "dataset_order",
            "dataset_bytes",
            "record_framing",
            "length_encoding",
            "digest_encoding",
        },
        "manifest.logical_digest",
    )
    if logical_digest != {
        "dataset_order": list(_DATASET_NAMES),
        "dataset_bytes": "canonical-c-order",
        "record_framing": "length-prefixed-v1",
        "length_encoding": "unsigned-big-endian-8",
        "digest_encoding": "raw-32-bytes",
    }:
        raise ConversionContractError("Manifest logical digest framing is not canonical")

    writer = _mapping(contract["writer"], "writer")
    _exact_keys(
        writer,
        {
            "mode",
            "swmr",
            "overwrite",
            "temporary_same_directory",
            "fsync_before_publish",
            "manifest_published_last",
        },
        "writer",
    )
    if writer["mode"] != "single-process":
        raise ConversionContractError("Writer mode must be single-process")
    required_flags = {
        "swmr": False,
        "overwrite": False,
        "temporary_same_directory": True,
        "fsync_before_publish": True,
        "manifest_published_last": True,
    }
    for name, required in required_flags.items():
        if _boolean(writer[name], f"writer.{name}") is not required:
            raise ConversionContractError(f"writer.{name} must be {required}")


def load_conversion_contract(path: Path, dataset_spec_path: Path) -> dict[str, Any]:
    """Load and strictly cross-check the repository HDF5 conversion contract."""

    if path.is_symlink():
        raise ConversionContractError("Conversion contract must not be a symlink")
    if path.stat().st_size > MAX_CONVERSION_CONTRACT_BYTES:
        raise ConversionContractError(
            f"Conversion contract exceeds {MAX_CONVERSION_CONTRACT_BYTES} bytes"
        )
    with path.open(encoding="utf-8") as stream:
        contract = _mapping(yaml.safe_load(stream), "contract")
    dataset_spec = load_dataset_spec(dataset_spec_path)
    _validate_contract(contract, dataset_spec)
    return contract


def conversion_contract_sha256(path: Path) -> str:
    """Hash the exact committed contract bytes for later manifest binding."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def conversion_row_index(
    contract: dict[str, Any], modulation: str, snr_db: int, source_index: int
) -> int:
    """Map canonical source coordinates to one stable HDF5 row."""

    if not isinstance(modulation, str):
        raise TypeError("modulation must be a string")
    if isinstance(snr_db, bool) or not isinstance(snr_db, Integral):
        raise TypeError("snr_db must be an integer")
    if isinstance(source_index, bool) or not isinstance(source_index, Integral):
        raise TypeError("source_index must be an integer")
    modulations = contract["ordering"]["modulation_order"]
    snr_values = contract["ordering"]["snr_db_order"]
    source_range = contract["ordering"]["source_index"]
    try:
        modulation_index = modulations.index(modulation)
    except ValueError as error:
        raise ValueError(f"Unknown modulation: {modulation!r}") from error
    try:
        snr_index = snr_values.index(int(snr_db))
    except ValueError as error:
        raise ValueError(f"Unknown SNR: {snr_db!r}") from error
    source_value = int(source_index)
    if not source_range["start"] <= source_value < source_range["stop"]:
        raise ValueError(f"source_index is outside the configured cell range: {source_value}")
    samples_per_cell = source_range["stop"] - source_range["start"]
    return (modulation_index * len(snr_values) + snr_index) * samples_per_cell + source_value


def conversion_sample_id(
    contract: dict[str, Any], modulation: str, snr_db: int, source_index: int
) -> str:
    """Return the stable source-coordinate identifier for one converted sample."""

    conversion_row_index(contract, modulation, snr_db, source_index)
    separator = contract["sample_id"]["separator"]
    width = contract["sample_id"]["source_index_width"]
    return separator.join(
        (
            contract["dataset_id"],
            modulation,
            f"{int(snr_db):+03d}",
            f"{int(source_index):0{width}d}",
        )
    )
