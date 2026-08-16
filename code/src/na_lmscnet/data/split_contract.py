"""Validated contract for deterministic, leakage-gated dataset splitting."""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from na_lmscnet.data.conversion_contract import load_conversion_contract
from na_lmscnet.data.provenance import load_dataset_spec

MAX_SPLIT_CONTRACT_BYTES = 64 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_SPLITS = ("train", "validation", "test")
_SOURCE_KEYS = {
    "dataset_spec_sha256",
    "conversion_contract_sha256",
    "source_archive_sha256",
    "source_dataset_content_sha256",
    "hdf5",
    "conversion_manifest",
}
_REQUIRED_BINDINGS = {
    "split_contract_sha256",
    "dataset_spec_sha256",
    "conversion_contract_sha256",
    "conversion_manifest_sha256",
    "source_archive_sha256",
    "source_dataset_content_sha256",
    "hdf5_file_sha256",
    "hdf5_logical_content_sha256",
    "assignment_sha256",
}
_REQUIRED_ENVIRONMENT = {"project_commit", "python", "numpy", "h5py", "hdf5"}
_FREEZE_BINDINGS = {
    "split_manifest_sha256",
    "experiment_config_sha256",
    "project_commit",
    "run_seeds",
    "selected_checkpoints",
}
_EXPECTED_HDF5 = {
    "filename": "RML2016.10a.h5",
    "size_bytes": 226409344,
    "file_sha256": "96120f40a9190bf24697227aaa7377a4e1cf883b3bb1b602b176f2622ebf7c63",
    "logical_content_sha256": "0713dd71751ff18fa0f0de26e570afb0f18a8e00191748a3c4a10f9a3271bce4",
}
_EXPECTED_CONVERSION_MANIFEST = {
    "filename": "RML2016.10a.conversion-manifest.json",
    "file_sha256": "de5bcb3dc6c490dca774d18bb7f3d3fd79634b55f9e2c31af244ac55b8ea776e",
    "implementation_commit": "3d836ca356b2a78aa9b94bd54a2468db9bca24b9",
}


class SplitContractError(ValueError):
    """Raised when a split contract is inconsistent or weakens an isolation gate."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SplitContractError(f"{field} must be a string-keyed mapping")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    keys = set(mapping)
    if keys != expected:
        raise SplitContractError(
            f"{field} fields differ: missing={sorted(expected - keys)}, "
            f"unexpected={sorted(keys - expected)}"
        )


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SplitContractError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise SplitContractError(f"{field} must be at least {minimum}")
    return result


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SplitContractError(f"{field} must be a non-empty trimmed string")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise SplitContractError(f"{field} must be a list")
    result = [_string(item, f"{field} item") for item in value]
    if len(result) != len(set(result)):
        raise SplitContractError(f"{field} must not contain duplicates")
    return result


def _integer_list(value: object, field: str) -> list[int]:
    if not isinstance(value, list):
        raise SplitContractError(f"{field} must be a list")
    result = [_integer(item, f"{field} item") for item in value]
    if len(result) != len(set(result)):
        raise SplitContractError(f"{field} must not contain duplicates")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise SplitContractError(f"{field} must be a boolean")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise SplitContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _git_commit(value: object, field: str) -> str:
    commit = _string(value, field)
    if len(commit) != 40 or any(character not in _HEX_DIGITS for character in commit):
        raise SplitContractError(f"{field} must be a full lowercase Git SHA-1 commit")
    return commit


def _basename(value: object, field: str, suffix: str) -> str:
    filename = _string(value, field)
    if (
        PurePosixPath(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or filename in {".", ".."}
        or not filename.endswith(suffix)
    ):
        raise SplitContractError(f"{field} must be a path-free {suffix} filename")
    return filename


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocation_counts(contract: dict[str, Any], sample_count: int) -> dict[str, int]:
    """Allocate one stratum using the contracted largest-remainder rule."""

    count = _integer(sample_count, "sample_count", minimum=0)
    stratification = contract["stratification"]
    splits = stratification["split_order"]
    weights = stratification["ratio_weights"]
    denominator = sum(weights[name] for name in splits)
    allocated = {name: count * weights[name] // denominator for name in splits}
    remainders = {name: (count * weights[name]) % denominator for name in splits}
    remaining = count - sum(allocated.values())
    priority = sorted(splits, key=lambda name: (-remainders[name], splits.index(name)))
    for name in priority[:remaining]:
        allocated[name] += 1
    return allocated


def split_rank_digest(contract: dict[str, Any], sample_id: str) -> str:
    """Return the version-independent ranking digest for one stable sample ID."""

    identifier = _string(sample_id, "sample_id")
    try:
        identifier_bytes = identifier.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SplitContractError("sample_id must be UTF-8 encodable") from error
    assignment = contract["assignment"]
    record = assignment["ranking_record"]
    values = (
        record["domain"].encode("utf-8"),
        str(assignment["seed"]).encode("ascii"),
        identifier_bytes,
    )
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def rank_sample_ids(contract: dict[str, Any], sample_ids: list[str]) -> list[str]:
    """Rank unique stable IDs with an explicit collision tie-break."""

    if not isinstance(sample_ids, list):
        raise SplitContractError("sample_ids must be a list")
    identifiers = [_string(value, "sample_ids item") for value in sample_ids]
    if len(identifiers) != len(set(identifiers)):
        raise SplitContractError("sample_ids must not contain duplicates")
    return sorted(
        identifiers,
        key=lambda value: (
            bytes.fromhex(split_rank_digest(contract, value)),
            value.encode("utf-8"),
        ),
    )


def assign_stratum_sample_ids(
    contract: dict[str, Any], sample_ids: list[str]
) -> dict[str, list[str]]:
    """Assign one complete stratum without relying on a library RNG."""

    expected = contract["stratification"]["samples_per_stratum"]
    if len(sample_ids) != expected:
        raise SplitContractError(
            f"One stratum must contain exactly {expected} sample IDs, got {len(sample_ids)}"
        )
    coordinates = [_sample_id_coordinates(contract, sample_id) for sample_id in sample_ids]
    strata = {(modulation, snr_db) for modulation, snr_db, _ in coordinates}
    if len(strata) != 1:
        raise SplitContractError(
            "One assignment call must contain exactly one modulation/SNR stratum"
        )
    source_indices = [source_index for _, _, source_index in coordinates]
    if set(source_indices) != set(range(expected)):
        raise SplitContractError("A stratum must contain every source index exactly once")
    ranked = rank_sample_ids(contract, sample_ids)
    counts = allocation_counts(contract, len(ranked))
    assignments: dict[str, list[str]] = {}
    start = 0
    for name in contract["stratification"]["split_order"]:
        stop = start + counts[name]
        assignments[name] = ranked[start:stop]
        start = stop
    if start != len(ranked):
        raise SplitContractError("Split assignment did not consume the complete stratum")
    return assignments


def _sample_id_coordinates(contract: dict[str, Any], sample_id: str) -> tuple[str, int, int]:
    identifier = _string(sample_id, "sample_id")
    identity = contract["assignment"]["sample_identity"]
    parts = identifier.split(identity["separator"])
    if len(parts) != 4 or parts[0] != contract["dataset_id"]:
        raise SplitContractError("sample_id does not use the contracted source-coordinate format")
    _, modulation, snr_text, source_text = parts
    if modulation not in contract["stratification"]["modulation_order"]:
        raise SplitContractError(f"sample_id contains an unknown modulation: {modulation!r}")
    if len(snr_text) != 3 or snr_text[0] not in "+-" or not snr_text[1:].isdigit():
        raise SplitContractError("sample_id SNR must use signed-width-3 formatting")
    snr_db = int(snr_text)
    if snr_db not in contract["stratification"]["snr_db_order"]:
        raise SplitContractError(f"sample_id contains an unknown SNR: {snr_db}")
    if snr_text != f"{snr_db:+03d}":
        raise SplitContractError("sample_id SNR is not canonically formatted")
    width = identity["source_index_width"]
    if len(source_text) != width or not source_text.isdigit():
        raise SplitContractError("sample_id source index has invalid formatting")
    source_index = int(source_text)
    if not 0 <= source_index < contract["stratification"]["samples_per_stratum"]:
        raise SplitContractError("sample_id source index is outside the contracted stratum")
    return modulation, snr_db, source_index


def _validate_source(
    contract: dict[str, Any],
    dataset_spec: dict[str, Any],
    conversion: dict[str, Any],
    dataset_spec_path: Path,
    conversion_contract_path: Path,
) -> None:
    source = _mapping(contract["source"], "source")
    _exact_keys(source, _SOURCE_KEYS, "source")
    if _sha256(source["dataset_spec_sha256"], "source.dataset_spec_sha256") != _file_sha256(
        dataset_spec_path
    ):
        raise SplitContractError("Split contract does not bind the exact dataset specification")
    if _sha256(
        source["conversion_contract_sha256"], "source.conversion_contract_sha256"
    ) != _file_sha256(conversion_contract_path):
        raise SplitContractError("Split contract does not bind the exact conversion contract")
    conversion_source = conversion["source"]
    if source["source_archive_sha256"] != conversion_source["archive_sha256"]:
        raise SplitContractError("Source archive digest differs from the conversion contract")
    if source["source_dataset_content_sha256"] != conversion_source["dataset_content_sha256"]:
        raise SplitContractError("Source content digest differs from the conversion contract")
    _sha256(source["source_archive_sha256"], "source.source_archive_sha256")
    _sha256(source["source_dataset_content_sha256"], "source.source_dataset_content_sha256")

    hdf5 = _mapping(source["hdf5"], "source.hdf5")
    _exact_keys(
        hdf5, {"filename", "size_bytes", "file_sha256", "logical_content_sha256"}, "source.hdf5"
    )
    if (
        _basename(hdf5["filename"], "source.hdf5.filename", ".h5")
        != conversion["format"]["output_filename"]
    ):
        raise SplitContractError("HDF5 filename differs from the conversion contract")
    _integer(hdf5["size_bytes"], "source.hdf5.size_bytes", minimum=1)
    _sha256(hdf5["file_sha256"], "source.hdf5.file_sha256")
    _sha256(hdf5["logical_content_sha256"], "source.hdf5.logical_content_sha256")
    if hdf5 != _EXPECTED_HDF5:
        raise SplitContractError("HDF5 identity differs from the verified local artifact")

    conversion_manifest = _mapping(source["conversion_manifest"], "source.conversion_manifest")
    _exact_keys(
        conversion_manifest,
        {"filename", "file_sha256", "implementation_commit"},
        "source.conversion_manifest",
    )
    if (
        _basename(
            conversion_manifest["filename"],
            "source.conversion_manifest.filename",
            ".conversion-manifest.json",
        )
        != conversion["manifest"]["filename"]
    ):
        raise SplitContractError("Conversion manifest filename differs from its contract")
    _sha256(conversion_manifest["file_sha256"], "source.conversion_manifest.file_sha256")
    _git_commit(
        conversion_manifest["implementation_commit"],
        "source.conversion_manifest.implementation_commit",
    )
    if conversion_manifest != _EXPECTED_CONVERSION_MANIFEST:
        raise SplitContractError(
            "Conversion manifest identity differs from the verified local artifact"
        )
    if contract["dataset_id"] != dataset_spec["dataset_id"]:
        raise SplitContractError("Split contract dataset_id differs from the dataset specification")


def _validate_stratification(contract: dict[str, Any], dataset_spec: dict[str, Any]) -> None:
    stratification = _mapping(contract["stratification"], "stratification")
    _exact_keys(
        stratification,
        {
            "fields",
            "modulation_order",
            "snr_db_order",
            "strata",
            "samples_per_stratum",
            "split_order",
            "ratio_weights",
            "rounding",
            "expected_per_stratum",
            "expected_totals",
        },
        "stratification",
    )
    if stratification["fields"] != ["modulation", "snr_db"]:
        raise SplitContractError("Splits must be stratified by modulation and SNR")
    expected = dataset_spec["expected"]
    if (
        _string_list(stratification["modulation_order"], "stratification.modulation_order")
        != expected["modulations"]
    ):
        raise SplitContractError("Modulation order must match the dataset specification")
    if (
        _integer_list(stratification["snr_db_order"], "stratification.snr_db_order")
        != expected["snr_db"]
    ):
        raise SplitContractError("SNR order must match the dataset specification")
    strata = len(expected["modulations"]) * len(expected["snr_db"])
    if _integer(stratification["strata"], "stratification.strata", minimum=1) != strata:
        raise SplitContractError("Configured stratum count differs from the dataset grid")
    samples_per_stratum = _integer(
        stratification["samples_per_stratum"],
        "stratification.samples_per_stratum",
        minimum=1,
    )
    if samples_per_stratum != expected["samples_per_cell"]:
        raise SplitContractError("Configured stratum size differs from the dataset specification")
    if _string_list(stratification["split_order"], "stratification.split_order") != list(_SPLITS):
        raise SplitContractError("Split order must be train, validation, test")
    weights = _mapping(stratification["ratio_weights"], "stratification.ratio_weights")
    _exact_keys(weights, set(_SPLITS), "stratification.ratio_weights")
    if {name: _integer(weights[name], f"ratio weight {name}", minimum=1) for name in _SPLITS} != {
        "train": 7,
        "validation": 1,
        "test": 2,
    }:
        raise SplitContractError("Split ratio must remain 70/10/20")
    rounding = _mapping(stratification["rounding"], "stratification.rounding")
    if rounding != {"algorithm": "largest-remainder-v1", "tie_break": "split-order"}:
        raise SplitContractError("Rounding must use largest-remainder-v1 with split-order ties")
    counts = allocation_counts(contract, samples_per_stratum)
    for field, multiplier in (("expected_per_stratum", 1), ("expected_totals", strata)):
        configured = _mapping(stratification[field], f"stratification.{field}")
        _exact_keys(configured, set(_SPLITS), f"stratification.{field}")
        if configured != {name: counts[name] * multiplier for name in _SPLITS}:
            raise SplitContractError(f"stratification.{field} is inconsistent with the ratio")
    if sum(counts.values()) * strata != expected["total_samples"]:
        raise SplitContractError("Split totals do not cover the dataset exactly once")


def _validate_assignment(contract: dict[str, Any]) -> None:
    assignment = _mapping(contract["assignment"], "assignment")
    _exact_keys(
        assignment,
        {
            "seed",
            "algorithm",
            "independently_rank_each_stratum",
            "sample_identity",
            "ranking_record",
            "ordering",
        },
        "assignment",
    )
    if _integer(assignment["seed"], "assignment.seed", minimum=0) != 2026:
        raise SplitContractError("Split seed must remain 2026")
    if assignment["algorithm"] != "sha256-rank-v1":
        raise SplitContractError("Assignment must use sha256-rank-v1")
    if not _boolean(
        assignment["independently_rank_each_stratum"],
        "assignment.independently_rank_each_stratum",
    ):
        raise SplitContractError("Every stratum must be ranked independently")
    identity = _mapping(assignment["sample_identity"], "assignment.sample_identity")
    if identity != {
        "scheme": "source-coordinate-v1",
        "separator": ":",
        "fields": ["dataset_id", "modulation", "snr_db", "source_index"],
        "snr_format": "signed-width-3",
        "source_index_width": 4,
    }:
        raise SplitContractError("Assignment must use canonical stable source-coordinate IDs")
    record = _mapping(assignment["ranking_record"], "assignment.ranking_record")
    if record != {
        "fields": ["domain", "seed_decimal", "sample_id"],
        "domain": "na-lmscnet/radioml_2016_10a/split-ranking/v1",
        "text_encoding": "utf-8",
        "framing": "length-prefixed-v1",
        "length_encoding": "unsigned-big-endian-8",
    }:
        raise SplitContractError("Ranking record framing is not canonical")
    ordering = _mapping(assignment["ordering"], "assignment.ordering")
    if ordering != {
        "rank_by": ["sha256_digest_bytes", "sample_id_utf8_bytes"],
        "assignment_by": "contiguous-split-order",
        "manifest_rows": "ascending-hdf5-row-index",
    }:
        raise SplitContractError("Assignment ordering is not canonical")


def _validate_leakage_and_isolation(contract: dict[str, Any]) -> None:
    leakage = _mapping(contract["leakage"], "leakage")
    _exact_keys(leakage, {"exact_duplicates", "near_duplicates", "adjacent_windows"}, "leakage")
    exact = _mapping(leakage["exact_duplicates"], "leakage.exact_duplicates")
    if exact != {
        "representation": "canonical-little-endian-float32-iq-bytes",
        "digest": "sha256",
        "scope": "all-samples",
        "cross_split_policy": "reject",
        "within_split_policy": "report",
    }:
        raise SplitContractError("Exact duplicate policy must remain fail-closed across splits")
    near = _mapping(leakage["near_duplicates"], "leakage.near_duplicates")
    if near != {
        "required_for_split_generation": False,
        "audit_contract": "radioml_2016_10a_near_duplicate_v1",
        "bounded_fixture_verified": True,
        "global_production_audit_completed": False,
        "policy": "record-unverified-limitation",
    }:
        raise SplitContractError("Near-duplicate limitations must be recorded without false claims")
    adjacent = _mapping(leakage["adjacent_windows"], "leakage.adjacent_windows")
    if adjacent != {
        "provenance": "unavailable-in-radioml-2016-10a",
        "source_index_semantics": "within-cell-array-position-only",
        "source_index_is_window_group": False,
        "policy": "record-limitation-and-never-claim-verified",
    }:
        raise SplitContractError("Adjacent-window limitations must not be weakened")

    isolation = _mapping(contract["test_isolation"], "test_isolation")
    _exact_keys(
        isolation,
        {
            "training_allowed_splits",
            "tuning_allowed_splits",
            "test_access_requires",
            "freeze_must_bind",
            "test_metrics_before_freeze",
        },
        "test_isolation",
    )
    if isolation["training_allowed_splits"] != ["train"]:
        raise SplitContractError("Training must not read validation or test samples")
    if isolation["tuning_allowed_splits"] != ["train", "validation"]:
        raise SplitContractError("Tuning must not read test samples")
    if isolation["test_access_requires"] != "experiment-freeze-manifest-v1":
        raise SplitContractError("Test access must require a freeze manifest")
    if (
        set(_string_list(isolation["freeze_must_bind"], "test_isolation.freeze_must_bind"))
        != _FREEZE_BINDINGS
    ):
        raise SplitContractError("Experiment freeze bindings are incomplete")
    if isolation["test_metrics_before_freeze"] != "forbidden":
        raise SplitContractError("Test metrics must remain forbidden before configuration freeze")


def _validate_manifest_and_gate(contract: dict[str, Any]) -> None:
    manifest = _mapping(contract["manifest"], "manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "filename",
            "leakage_audit_filename",
            "hash_algorithm",
            "absolute_paths",
            "assignment_encoding",
            "required_bindings",
            "required_environment",
        },
        "manifest",
    )
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise SplitContractError("Split manifest must use schema version 1 and SHA-256")
    split_filename = _basename(manifest["filename"], "manifest.filename", ".split-manifest.json")
    audit_filename = _basename(
        manifest["leakage_audit_filename"],
        "manifest.leakage_audit_filename",
        ".leakage-audit.json",
    )
    if split_filename == audit_filename:
        raise SplitContractError("Split and leakage manifests must use different filenames")
    if _boolean(manifest["absolute_paths"], "manifest.absolute_paths"):
        raise SplitContractError("Split artifacts must redact absolute paths")
    if manifest["assignment_encoding"] != "sorted-hdf5-row-indices":
        raise SplitContractError("Manifest assignments must contain sorted HDF5 row indices")
    if (
        set(_string_list(manifest["required_bindings"], "manifest.required_bindings"))
        != _REQUIRED_BINDINGS
    ):
        raise SplitContractError("Split manifest source bindings are incomplete")
    if (
        set(_string_list(manifest["required_environment"], "manifest.required_environment"))
        != _REQUIRED_ENVIRONMENT
    ):
        raise SplitContractError("Split manifest environment bindings are incomplete")

    publication = _mapping(contract["publication"], "publication")
    expected_publication = {
        "output_outside_repository": True,
        "mode": "single-process",
        "overwrite": False,
        "temporary_same_directory": True,
        "fsync_before_publish": True,
        "split_manifest_published_before_leakage_audit": True,
        "leakage_audit_is_completion_marker": True,
    }
    if publication != expected_publication:
        raise SplitContractError("Split publication must remain external, atomic, and fail-closed")

    gate = _mapping(contract["generation_gate"], "generation_gate")
    if gate != {
        "split_generation_enabled": True,
        "blocked_by": [],
    }:
        raise SplitContractError("Split generation gate does not match the approved protocol")


def _validate_contract(
    contract: dict[str, Any],
    dataset_spec: dict[str, Any],
    conversion: dict[str, Any],
    dataset_spec_path: Path,
    conversion_contract_path: Path,
) -> None:
    _exact_keys(
        contract,
        {
            "schema_version",
            "contract_id",
            "dataset_id",
            "source",
            "stratification",
            "assignment",
            "leakage",
            "test_isolation",
            "manifest",
            "publication",
            "generation_gate",
        },
        "contract",
    )
    if contract["schema_version"] != 1:
        raise SplitContractError("Unsupported split contract schema version")
    if contract["contract_id"] != "radioml_2016_10a_split_v1":
        raise SplitContractError("Unexpected split contract identifier")
    _validate_source(
        contract, dataset_spec, conversion, dataset_spec_path, conversion_contract_path
    )
    _validate_stratification(contract, dataset_spec)
    _validate_assignment(contract)
    _validate_leakage_and_isolation(contract)
    _validate_manifest_and_gate(contract)


def load_split_contract(
    path: Path, dataset_spec_path: Path, conversion_contract_path: Path
) -> dict[str, Any]:
    """Load and strictly cross-check the repository split design contract."""

    if path.is_symlink():
        raise SplitContractError("Split contract must not be a symlink")
    if dataset_spec_path.is_symlink():
        raise SplitContractError("Dataset specification must not be a symlink")
    if conversion_contract_path.is_symlink():
        raise SplitContractError("Conversion contract must not be a symlink")
    if path.stat().st_size > MAX_SPLIT_CONTRACT_BYTES:
        raise SplitContractError(f"Split contract exceeds {MAX_SPLIT_CONTRACT_BYTES} bytes")
    with path.open(encoding="utf-8") as stream:
        contract = _mapping(yaml.safe_load(stream), "contract")
    dataset_spec = load_dataset_spec(dataset_spec_path)
    conversion = load_conversion_contract(conversion_contract_path, dataset_spec_path)
    _validate_contract(
        contract, dataset_spec, conversion, dataset_spec_path, conversion_contract_path
    )
    return contract


def split_contract_sha256(path: Path) -> str:
    """Hash the exact contract bytes for later split-manifest binding."""

    return _file_sha256(path)
