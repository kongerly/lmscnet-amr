"""Validated near-duplicate audit design and bounded reference similarity helpers."""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import yaml

from na_lmscnet.data.conversion_contract import load_conversion_contract
from na_lmscnet.data.provenance import load_dataset_spec

MAX_NEAR_DUPLICATE_CONTRACT_BYTES = 64 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_SOURCE_KEYS = {
    "dataset_spec_sha256",
    "conversion_contract_sha256",
    "source_archive_sha256",
    "source_dataset_content_sha256",
    "hdf5",
}
_REPRESENTATION_KEYS = {"exact_bytes", "transformed_similarity"}
_SPLITS = ("same_source_transform", "unrelated", "ambiguous")
_REQUIRED_REPORT_BINDINGS = {
    "near_duplicate_contract_sha256",
    "dataset_spec_sha256",
    "conversion_contract_sha256",
    "source_archive_sha256",
    "source_dataset_content_sha256",
    "hdf5_logical_content_sha256",
    "calibration_report_sha256",
    "candidate_assignment_sha256",
    "pair_decision_sha256",
}
_EXPECTED_HDF5_LOGICAL_SHA256 = "0713dd71751ff18fa0f0de26e570afb0f18a8e00191748a3c4a10f9a3271bce4"


class NearDuplicateContractError(ValueError):
    """Raised when the near-duplicate audit contract is inconsistent or unsafe."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NearDuplicateContractError(f"{field} must be a string-keyed mapping")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    keys = set(mapping)
    if keys != expected:
        raise NearDuplicateContractError(
            f"{field} fields differ: missing={sorted(expected - keys)}, "
            f"unexpected={sorted(keys - expected)}"
        )


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise NearDuplicateContractError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise NearDuplicateContractError(f"{field} must be at least {minimum}")
    return result


def _real(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NearDuplicateContractError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise NearDuplicateContractError(f"{field} must be finite and meet the configured bound")
    return result


def _float(value: object, field: str, *, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise NearDuplicateContractError(f"{field} must be a YAML floating-point value")
    return _real(value, field, minimum=minimum)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NearDuplicateContractError(f"{field} must be a non-empty trimmed string")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise NearDuplicateContractError(f"{field} must be a list")
    result = [_string(item, f"{field} item") for item in value]
    if len(result) != len(set(result)):
        raise NearDuplicateContractError(f"{field} must not contain duplicates")
    return result


def _integer_list(value: object, field: str) -> list[int]:
    if not isinstance(value, list):
        raise NearDuplicateContractError(f"{field} must be a list")
    return [_integer(item, f"{field} item") for item in value]


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise NearDuplicateContractError(f"{field} must be a boolean")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise NearDuplicateContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _basename(value: object, field: str, suffix: str) -> str:
    filename = _string(value, field)
    if (
        PurePosixPath(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or filename in {".", ".."}
        or not filename.endswith(suffix)
    ):
        raise NearDuplicateContractError(f"{field} must be a path-free {suffix} filename")
    return filename


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise NearDuplicateContractError("Near-duplicate contract does not bind dataset spec bytes")
    if _sha256(
        source["conversion_contract_sha256"], "source.conversion_contract_sha256"
    ) != _file_sha256(conversion_contract_path):
        raise NearDuplicateContractError("Near-duplicate contract does not bind conversion bytes")
    conversion_source = conversion["source"]
    if source["source_archive_sha256"] != conversion_source["archive_sha256"]:
        raise NearDuplicateContractError("Source archive digest differs from conversion contract")
    if source["source_dataset_content_sha256"] != conversion_source["dataset_content_sha256"]:
        raise NearDuplicateContractError("Source content digest differs from conversion contract")
    _sha256(source["source_archive_sha256"], "source.source_archive_sha256")
    _sha256(source["source_dataset_content_sha256"], "source.source_dataset_content_sha256")
    hdf5 = _mapping(source["hdf5"], "source.hdf5")
    _exact_keys(hdf5, {"filename", "logical_content_sha256"}, "source.hdf5")
    if (
        _basename(hdf5["filename"], "source.hdf5.filename", ".h5")
        != conversion["format"]["output_filename"]
    ):
        raise NearDuplicateContractError("HDF5 filename differs from conversion contract")
    logical_digest = _sha256(hdf5["logical_content_sha256"], "source.hdf5.logical_content_sha256")
    if logical_digest != _EXPECTED_HDF5_LOGICAL_SHA256:
        raise NearDuplicateContractError("HDF5 logical digest differs from verified artifact")
    if contract["dataset_id"] != dataset_spec["dataset_id"]:
        raise NearDuplicateContractError("Dataset ID differs from dataset specification")


def _validate_sample_domain(domain: dict[str, Any], dataset_spec: dict[str, Any]) -> None:
    _exact_keys(
        domain,
        {
            "total_samples",
            "sample_shape",
            "modulation_count",
            "snr_count",
            "cell_count",
            "samples_per_cell",
        },
        "sample_domain",
    )
    expected = dataset_spec["expected"]
    expected_values = {
        "total_samples": expected["total_samples"],
        "sample_shape": expected["sample_shape"],
        "modulation_count": len(expected["modulations"]),
        "snr_count": len(expected["snr_db"]),
        "cell_count": len(expected["modulations"]) * len(expected["snr_db"]),
        "samples_per_cell": expected["samples_per_cell"],
    }
    for field in (
        "total_samples",
        "modulation_count",
        "snr_count",
        "cell_count",
        "samples_per_cell",
    ):
        _integer(domain[field], f"sample_domain.{field}", minimum=1)
    _integer_list(domain["sample_shape"], "sample_domain.sample_shape")
    if domain != expected_values:
        raise NearDuplicateContractError("sample_domain does not match dataset specification")


def _validate_representations(
    representations: dict[str, Any], sample_domain: dict[str, Any]
) -> None:
    _exact_keys(representations, _REPRESENTATION_KEYS, "representations")
    exact = _mapping(representations["exact_bytes"], "representations.exact_bytes")
    if exact != {
        "id": "canonical-little-endian-float32-iq-bytes-v1",
        "input": "/iq",
        "operation": "sha256_raw_c_order_bytes",
        "purpose": "exact_duplicate_reference",
    }:
        raise NearDuplicateContractError("Exact-byte representation is not canonical")
    transformed = _mapping(
        representations["transformed_similarity"], "representations.transformed_similarity"
    )
    _exact_keys(
        transformed,
        {
            "id",
            "input",
            "complex_mapping",
            "finite_input_required",
            "mean_removal",
            "power_normalization",
            "allowed_transform_family",
            "forbidden_transform_family",
            "similarity",
        },
        "representations.transformed_similarity",
    )
    if (
        transformed["id"] != "power-normalized-complex-circular-correlation-v1"
        or transformed["input"] != "/iq"
    ):
        raise NearDuplicateContractError(
            "Transformed representation identifier or input is invalid"
        )
    if transformed["complex_mapping"] != "z[t] = float32(i[t]) + j * float32(q[t])":
        raise NearDuplicateContractError("Complex mapping is not canonical")
    if not _boolean(
        transformed["finite_input_required"], "transformed_similarity.finite_input_required"
    ):
        raise NearDuplicateContractError("Non-finite input must not enter similarity scoring")
    if (
        transformed["mean_removal"] is not False
        or transformed["power_normalization"] != "rms_complex_amplitude"
    ):
        raise NearDuplicateContractError("Transformed normalization policy is not canonical")
    if _string_list(transformed["allowed_transform_family"], "allowed_transform_family") != [
        "global_nonzero_complex_gain",
        "integer_circular_time_shift",
    ]:
        raise NearDuplicateContractError("Allowed transform family changed")
    if _string_list(transformed["forbidden_transform_family"], "forbidden_transform_family") != [
        "resampling",
        "time_scaling",
        "conjugation",
        "clipping",
        "additive_noise",
    ]:
        raise NearDuplicateContractError("Forbidden transform family changed")
    similarity = _mapping(transformed["similarity"], "transformed_similarity.similarity")
    _float(similarity.get("exact_transform_score"), "similarity.exact_transform_score")
    score_range = similarity.get("score_range")
    if not isinstance(score_range, list) or len(score_range) != 2:
        raise NearDuplicateContractError("Similarity score range must contain two floats")
    for index, value in enumerate(score_range):
        _float(value, f"similarity.score_range[{index}]", minimum=0.0)
    _integer_list(similarity.get("lag_domain"), "similarity.lag_domain")
    _boolean(similarity.get("phase_invariant"), "similarity.phase_invariant")
    if similarity != {
        "id": "max_abs_normalized_circular_cross_correlation-v1",
        "formula": "max_lag(abs(sum(conj(z_a[t]) * z_b[(t+lag) mod length]))) / (norm(z_a) * norm(z_b))",
        "score_range": [0.0, 1.0],
        "exact_transform_score": 1.0,
        "numerical_dtype": "float64",
        "lag_domain": [0, 127],
        "phase_invariant": True,
    }:
        raise NearDuplicateContractError("Similarity definition is not canonical")
    if sample_domain["sample_shape"][1] != similarity["lag_domain"][1] + 1:
        raise NearDuplicateContractError("Lag domain does not cover the sample length")


def _validate_candidate_generation(candidate: dict[str, Any]) -> None:
    _exact_keys(
        candidate,
        {
            "status",
            "algorithm",
            "production_enabled",
            "fixture_enabled",
            "fixture_max_samples",
            "candidate_recall_requirement",
            "candidate_recall_evidence",
            "blocking_keys",
            "false_negative_policy",
        },
        "candidate_generation",
    )
    _boolean(candidate["production_enabled"], "candidate_generation.production_enabled")
    _boolean(candidate["fixture_enabled"], "candidate_generation.fixture_enabled")
    _integer(
        candidate["fixture_max_samples"], "candidate_generation.fixture_max_samples", minimum=1
    )
    _float(
        candidate["candidate_recall_requirement"],
        "candidate_generation.candidate_recall_requirement",
        minimum=0.0,
    )
    if candidate != {
        "status": "reference_only",
        "algorithm": "exhaustive_pairwise_reference_v1",
        "production_enabled": False,
        "fixture_enabled": True,
        "fixture_max_samples": 64,
        "candidate_recall_requirement": 1.0,
        "candidate_recall_evidence": "required_before_production",
        "blocking_keys": "none",
        "false_negative_policy": "any_unproven_recall_is_blocking",
    }:
        raise NearDuplicateContractError(
            "Candidate generation must remain reference-only and fail-closed"
        )


def _validate_threshold_calibration(calibration: dict[str, Any]) -> None:
    _exact_keys(
        calibration,
        {"status", "threshold", "positive_fixture", "negative_fixture", "acceptance"},
        "threshold_calibration",
    )
    if calibration["status"] != "pending" or calibration["threshold"] is not None:
        raise NearDuplicateContractError("Threshold calibration must remain pending")
    positive = _mapping(calibration["positive_fixture"], "positive_fixture")
    _integer(positive.get("minimum_cases"), "positive_fixture.minimum_cases", minimum=1)
    if positive != {
        "required_transforms": ["global_nonzero_complex_gain", "integer_circular_time_shift"],
        "quantization": "source_float32_round_trip",
        "minimum_cases": 128,
    }:
        raise NearDuplicateContractError("Positive calibration fixture is incomplete")
    negative = _mapping(calibration["negative_fixture"], "negative_fixture")
    _integer(negative.get("seed"), "negative_fixture.seed", minimum=0)
    _integer(negative.get("minimum_cases"), "negative_fixture.minimum_cases", minimum=1)
    if negative != {
        "construction": "independently_seeded_nonmatching_pairs",
        "seed": 2026,
        "minimum_cases": 1024,
    }:
        raise NearDuplicateContractError("Negative calibration fixture is incomplete")
    acceptance = _mapping(calibration["acceptance"], "threshold_calibration.acceptance")
    _float(acceptance.get("positive_recall"), "acceptance.positive_recall", minimum=0.0)
    _float(
        acceptance.get("negative_false_positive_rate_max"),
        "acceptance.negative_false_positive_rate_max",
        minimum=0.0,
    )
    _boolean(
        acceptance.get("calibration_report_required"),
        "acceptance.calibration_report_required",
    )
    _boolean(
        acceptance.get("unresolved_threshold_blocks_split_generation"),
        "acceptance.unresolved_threshold_blocks_split_generation",
    )
    if acceptance != {
        "positive_recall": 1.0,
        "negative_false_positive_rate_max": 0.001,
        "calibration_report_required": True,
        "unresolved_threshold_blocks_split_generation": True,
    }:
        raise NearDuplicateContractError("Threshold acceptance policy is incomplete")


def _validate_review(review: dict[str, Any]) -> None:
    _exact_keys(
        review,
        {
            "status",
            "pair_labels",
            "ambiguous_policy",
            "reviewer_record_required",
            "reviewer_record_fields",
            "reviewer_record_absolute_paths",
        },
        "review",
    )
    _boolean(review.get("reviewer_record_required"), "review.reviewer_record_required")
    _boolean(
        review.get("reviewer_record_absolute_paths"),
        "review.reviewer_record_absolute_paths",
    )
    if review != {
        "status": "pending",
        "pair_labels": list(_SPLITS),
        "ambiguous_policy": "manual_review_and_block",
        "reviewer_record_required": True,
        "reviewer_record_fields": [
            "pair_digest",
            "decision",
            "reviewer",
            "timestamp_utc",
            "reason",
        ],
        "reviewer_record_absolute_paths": False,
    }:
        raise NearDuplicateContractError("Manual review policy is incomplete or unsafe")


def _validate_report(report: dict[str, Any]) -> None:
    _exact_keys(
        report,
        {
            "schema_version",
            "filename",
            "hash_algorithm",
            "absolute_paths",
            "required_bindings",
            "required_environment",
        },
        "report",
    )
    if _integer(report["schema_version"], "report.schema_version", minimum=1) != 1:
        raise NearDuplicateContractError("Audit report must use schema version 1")
    if report["hash_algorithm"] != "sha256":
        raise NearDuplicateContractError("Audit report must use schema version 1 and SHA-256")
    _basename(report["filename"], "report.filename", ".near-duplicate-audit.json")
    if report["absolute_paths"] is not False:
        raise NearDuplicateContractError("Audit report must redact absolute paths")
    if (
        set(_string_list(report["required_bindings"], "report.required_bindings"))
        != _REQUIRED_REPORT_BINDINGS
    ):
        raise NearDuplicateContractError("Audit report bindings are incomplete")
    if set(_string_list(report["required_environment"], "report.required_environment")) != {
        "project_commit",
        "python",
        "numpy",
        "h5py",
        "hdf5",
    }:
        raise NearDuplicateContractError("Audit report environment bindings are incomplete")


def _validate_publication_and_gate(contract: dict[str, Any]) -> None:
    publication = _mapping(contract["publication"], "publication")
    for field in (
        "output_outside_repository",
        "overwrite",
        "temporary_same_directory",
        "fsync_before_publish",
        "audit_report_is_completion_marker",
    ):
        _boolean(publication.get(field), f"publication.{field}")
    if publication != {
        "output_outside_repository": True,
        "mode": "single-process",
        "overwrite": False,
        "temporary_same_directory": True,
        "fsync_before_publish": True,
        "audit_report_is_completion_marker": True,
    }:
        raise NearDuplicateContractError("Audit publication must remain external and fail-closed")
    gate = _mapping(contract["generation_gate"], "generation_gate")
    _boolean(gate.get("audit_generation_enabled"), "generation_gate.audit_generation_enabled")
    _boolean(gate.get("split_generation_enabled"), "generation_gate.split_generation_enabled")
    if gate != {
        "audit_generation_enabled": False,
        "split_generation_enabled": True,
        "audit_blocked_by": [
            "candidate_recall_evidence",
            "threshold_calibration",
            "manual_review_protocol",
        ],
    }:
        raise NearDuplicateContractError(
            "Near-duplicate audit gate must remain disabled without blocking the approved split"
        )


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
            "sample_domain",
            "representations",
            "candidate_generation",
            "threshold_calibration",
            "review",
            "report",
            "publication",
            "generation_gate",
        },
        "contract",
    )
    if (
        contract["schema_version"] != 1
        or contract["contract_id"] != "radioml_2016_10a_near_duplicate_v1"
    ):
        raise NearDuplicateContractError("Unexpected near-duplicate contract identity")
    _validate_source(
        contract, dataset_spec, conversion, dataset_spec_path, conversion_contract_path
    )
    _validate_sample_domain(_mapping(contract["sample_domain"], "sample_domain"), dataset_spec)
    _validate_representations(
        _mapping(contract["representations"], "representations"), contract["sample_domain"]
    )
    _validate_candidate_generation(
        _mapping(contract["candidate_generation"], "candidate_generation")
    )
    _validate_threshold_calibration(
        _mapping(contract["threshold_calibration"], "threshold_calibration")
    )
    _validate_review(_mapping(contract["review"], "review"))
    _validate_report(_mapping(contract["report"], "report"))
    _validate_publication_and_gate(contract)


def load_near_duplicate_contract(
    path: Path, dataset_spec_path: Path, conversion_contract_path: Path
) -> dict[str, Any]:
    """Load and strictly cross-check the repository near-duplicate design contract."""

    if path.is_symlink() or dataset_spec_path.is_symlink() or conversion_contract_path.is_symlink():
        raise NearDuplicateContractError("Near-duplicate contract inputs must not be symlinks")
    if path.stat().st_size > MAX_NEAR_DUPLICATE_CONTRACT_BYTES:
        raise NearDuplicateContractError(
            f"Near-duplicate contract exceeds {MAX_NEAR_DUPLICATE_CONTRACT_BYTES} bytes"
        )
    with path.open(encoding="utf-8") as stream:
        contract = _mapping(yaml.safe_load(stream), "contract")
    dataset_spec = load_dataset_spec(dataset_spec_path)
    conversion = load_conversion_contract(conversion_contract_path, dataset_spec_path)
    _validate_contract(
        contract, dataset_spec, conversion, dataset_spec_path, conversion_contract_path
    )
    return contract


def near_duplicate_contract_sha256(path: Path) -> str:
    """Hash exact contract bytes for future audit-report binding."""

    return _file_sha256(path)


def _as_complex_iq(iq: np.ndarray) -> np.ndarray:
    array = np.asarray(iq)
    if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] < 1:
        raise ValueError("iq must have shape [2, length] with length >= 1")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("iq must use a floating-point dtype")
    if not bool(np.isfinite(array).all()):
        raise ValueError("iq must contain only finite values")
    return array[0].astype(np.float64) + 1j * array[1].astype(np.float64)


def power_normalized_complex(iq: np.ndarray) -> np.ndarray:
    """Return a read-only, RMS-normalized complex view for bounded fixtures."""

    complex_iq = _as_complex_iq(iq)
    rms = float(np.sqrt(np.mean(np.abs(complex_iq) ** 2, dtype=np.float64)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("iq must have positive finite RMS complex amplitude")
    result = np.asarray(complex_iq / rms, dtype=np.complex128)
    result.setflags(write=False)
    return result


def max_abs_normalized_circular_correlation(iq_a: np.ndarray, iq_b: np.ndarray) -> float:
    """Reference score for phase-invariant integer circular shifts."""

    a = power_normalized_complex(iq_a)
    b = power_normalized_complex(iq_b)
    if a.shape != b.shape:
        raise ValueError("iq inputs must have the same shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("iq inputs must have positive finite norm")
    best = 0.0
    for lag in range(a.size):
        score = abs(np.vdot(a, np.roll(b, lag))) / denominator
        best = max(best, float(score))
    return min(1.0, best)
