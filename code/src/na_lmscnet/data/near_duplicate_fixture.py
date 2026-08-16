"""Deterministic fixture evidence for the near-duplicate audit design."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from na_lmscnet.data.near_duplicate_contract import (
    load_near_duplicate_contract,
    max_abs_normalized_circular_correlation,
    near_duplicate_contract_sha256,
)

MAX_NEAR_DUPLICATE_FIXTURE_CONTRACT_BYTES = 32 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_SAMPLE_SHAPE = (2, 128)
_THRESHOLD_DECIMAL_PLACES = 12


class NearDuplicateFixtureError(ValueError):
    """Raised when fixture evidence is inconsistent or unsafe."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NearDuplicateFixtureError(f"{field} must be a string-keyed mapping")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    keys = set(mapping)
    if keys != expected:
        raise NearDuplicateFixtureError(
            f"{field} fields differ: missing={sorted(expected - keys)}, "
            f"unexpected={sorted(keys - expected)}"
        )


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NearDuplicateFixtureError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise NearDuplicateFixtureError(f"{field} must be at least {minimum}")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise NearDuplicateFixtureError(f"{field} must be a boolean")
    return value


def _float(value: object, field: str, *, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise NearDuplicateFixtureError(f"{field} must be a YAML floating-point value")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise NearDuplicateFixtureError(f"{field} must be finite and meet the configured bound")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NearDuplicateFixtureError(f"{field} must be a non-empty trimmed string")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise NearDuplicateFixtureError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_score(score: float) -> str:
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise NearDuplicateFixtureError("score must be finite and inside [0, 1]")
    return f"{score:.17g}"


def _load_yaml(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink():
        raise NearDuplicateFixtureError(f"{field} must not be a symlink")
    if path.stat().st_size > MAX_NEAR_DUPLICATE_FIXTURE_CONTRACT_BYTES:
        raise NearDuplicateFixtureError(
            f"{field} exceeds {MAX_NEAR_DUPLICATE_FIXTURE_CONTRACT_BYTES} bytes"
        )
    with path.open(encoding="utf-8") as stream:
        return _mapping(yaml.safe_load(stream), field)


def _validate_gain(value: object, field: str) -> complex:
    gain = _mapping(value, field)
    _exact_keys(gain, {"real", "imag"}, field)
    result = complex(_float(gain["real"], f"{field}.real"), _float(gain["imag"], f"{field}.imag"))
    if abs(result) <= 0.0:
        raise NearDuplicateFixtureError(f"{field} must be nonzero")
    return result


def _validate_fixture_contract(
    contract: dict[str, Any],
    near_duplicate_contract_path: Path,
    dataset_spec_path: Path,
    conversion_contract_path: Path,
) -> None:
    _exact_keys(
        contract,
        {
            "schema_version",
            "fixture_contract_id",
            "near_duplicate_contract",
            "sample_domain",
            "fixture_source",
            "calibration_fixture",
            "threshold_selection",
            "bounded_reference_audit",
            "publication",
            "generation_gate",
        },
        "fixture_contract",
    )
    if (
        contract["schema_version"] != 1
        or contract["fixture_contract_id"] != "radioml_2016_10a_near_duplicate_fixture_v1"
    ):
        raise NearDuplicateFixtureError("Unexpected near-duplicate fixture contract identity")

    near_contract = load_near_duplicate_contract(
        near_duplicate_contract_path, dataset_spec_path, conversion_contract_path
    )
    near_binding = _mapping(contract["near_duplicate_contract"], "near_duplicate_contract")
    if near_binding != {
        "contract_id": near_contract["contract_id"],
        "contract_sha256": near_duplicate_contract_sha256(near_duplicate_contract_path),
    }:
        raise NearDuplicateFixtureError("Fixture contract does not bind near-duplicate contract")
    _sha256(near_binding["contract_sha256"], "near_duplicate_contract.contract_sha256")

    domain = _mapping(contract["sample_domain"], "sample_domain")
    if domain != {"sample_shape": [2, 128]}:
        raise NearDuplicateFixtureError("Fixture sample shape must remain [2, 128]")

    source = _mapping(contract["fixture_source"], "fixture_source")
    for field in ("finite_input_required", "positive_rms_required"):
        _boolean(source.get(field), f"fixture_source.{field}")
    if source != {
        "generator": "sha256-counter-float32-v1",
        "finite_input_required": True,
        "positive_rms_required": True,
    }:
        raise NearDuplicateFixtureError("Fixture source generator is not canonical")

    calibration = _mapping(contract["calibration_fixture"], "calibration_fixture")
    gains = calibration.get("global_complex_gain_cycle")
    if not isinstance(gains, list) or len(gains) != 4:
        raise NearDuplicateFixtureError(
            "calibration_fixture.global_complex_gain_cycle must contain four gains"
        )
    [
        _validate_gain(gain, f"global_complex_gain_cycle[{index}]")
        for index, gain in enumerate(gains)
    ]
    if calibration != {
        "positive_cases": 128,
        "positive_quantization": "source_float32_round_trip",
        "global_complex_gain_cycle": gains,
        "shift_rule": "((case_index * 37) + 11) mod 128",
        "negative_cases": 1024,
        "negative_seed": 2026,
    }:
        raise NearDuplicateFixtureError("Calibration fixture policy is incomplete")

    threshold = _mapping(contract["threshold_selection"], "threshold_selection")
    _float(threshold.get("positive_recall_required"), "positive_recall_required", minimum=0.0)
    _float(
        threshold.get("negative_false_positive_rate_max"),
        "negative_false_positive_rate_max",
        minimum=0.0,
    )
    if threshold != {
        "rule": "floor_min_positive_score_to_12_decimal_places",
        "positive_recall_required": 1.0,
        "negative_false_positive_rate_max": 0.001,
    }:
        raise NearDuplicateFixtureError("Threshold selection rule is not canonical")

    audit = _mapping(contract["bounded_reference_audit"], "bounded_reference_audit")
    for field in (
        "sample_count",
        "base_transform_pairs",
        "unrelated_samples",
        "expected_pair_count",
    ):
        _integer(audit.get(field), f"bounded_reference_audit.{field}", minimum=1)
    _float(audit.get("candidate_recall_required"), "candidate_recall_required", minimum=0.0)
    _boolean(audit.get("production_claim"), "bounded_reference_audit.production_claim")
    if audit != {
        "sample_count": 64,
        "base_transform_pairs": 16,
        "unrelated_samples": 32,
        "algorithm": "exhaustive_pairwise_reference_v1",
        "expected_pair_count": 2016,
        "candidate_recall_required": 1.0,
        "production_claim": False,
    }:
        raise NearDuplicateFixtureError("Bounded reference audit policy is incomplete")

    publication = _mapping(contract["publication"], "publication")
    _boolean(publication.get("writes_artifacts"), "publication.writes_artifacts")
    _boolean(publication.get("absolute_paths"), "publication.absolute_paths")
    if publication != {
        "writes_artifacts": False,
        "output": "stdout_summary_only",
        "absolute_paths": False,
    }:
        raise NearDuplicateFixtureError("Fixture publication must remain stdout-only")

    gate = _mapping(contract["generation_gate"], "generation_gate")
    for field in (
        "production_candidate_generation_enabled",
        "near_duplicate_audit_generation_enabled",
        "split_generation_enabled",
    ):
        _boolean(gate.get(field), f"generation_gate.{field}")
    if gate != {
        "production_candidate_generation_enabled": False,
        "near_duplicate_audit_generation_enabled": False,
        "split_generation_enabled": True,
    }:
        raise NearDuplicateFixtureError(
            "Fixture generation gates differ from the approved protocol"
        )


def load_near_duplicate_fixture_contract(
    path: Path,
    near_duplicate_contract_path: Path,
    dataset_spec_path: Path,
    conversion_contract_path: Path,
) -> dict[str, Any]:
    """Load the deterministic calibration fixture contract and its bindings."""

    contract = _load_yaml(path, "fixture_contract")
    _validate_fixture_contract(
        contract, near_duplicate_contract_path, dataset_spec_path, conversion_contract_path
    )
    return contract


def near_duplicate_fixture_contract_sha256(path: Path) -> str:
    """Hash exact fixture contract bytes for report binding."""

    return _file_sha256(path)


def _counter_bytes(label: str, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < byte_count:
        chunks.append(hashlib.sha256(f"{label}:{counter}".encode()).digest())
        counter += 1
    return b"".join(chunks)[:byte_count]


def deterministic_fixture_sample(label: str) -> np.ndarray:
    """Generate one finite nonzero float32 I/Q fixture sample from a stable label."""

    raw = _counter_bytes(label, 256 * 4)
    integers = np.frombuffer(raw, dtype="<u4").astype(np.float64)
    values = (integers / float(2**32 - 1)) * 2.0 - 1.0
    sample = values.reshape(_SAMPLE_SHAPE).astype(np.float32)
    if not bool(np.isfinite(sample).all()):
        raise NearDuplicateFixtureError("Generated fixture sample is not finite")
    if float(np.sqrt(np.mean(np.abs(sample[0] + 1j * sample[1]) ** 2))) <= 0.0:
        raise NearDuplicateFixtureError("Generated fixture sample has zero RMS")
    sample.setflags(write=False)
    return sample


def _transform_iq(iq: np.ndarray, *, case_index: int, gain: complex, shift: int) -> np.ndarray:
    complex_signal = iq[0].astype(np.float32) + 1j * iq[1].astype(np.float32)
    transformed = np.roll(complex_signal * gain, shift).astype(np.complex64)
    result = np.vstack([transformed.real, transformed.imag]).astype(np.float32)
    result.setflags(write=False)
    return result


def _gain_cycle(contract: dict[str, Any]) -> list[complex]:
    gains = contract["calibration_fixture"]["global_complex_gain_cycle"]
    return [complex(float(gain["real"]), float(gain["imag"])) for gain in gains]


def _shift(case_index: int) -> int:
    return ((case_index * 37) + 11) % _SAMPLE_SHAPE[1]


def _floor_score(score: float) -> float:
    scale = float(10**_THRESHOLD_DECIMAL_PLACES)
    return math.floor(score * scale) / scale


def _score_summary(scores: list[float]) -> dict[str, object]:
    if not scores:
        raise NearDuplicateFixtureError("score list must not be empty")
    return {
        "count": len(scores),
        "min": _format_score(min(scores)),
        "max": _format_score(max(scores)),
        "mean": _format_score(float(np.mean(np.asarray(scores, dtype=np.float64)))),
        "sha256": _json_digest([_format_score(score) for score in scores]),
    }


def _calibration_scores(contract: dict[str, Any]) -> tuple[list[float], list[float]]:
    calibration = contract["calibration_fixture"]
    gains = _gain_cycle(contract)
    positives: list[float] = []
    for case_index in range(calibration["positive_cases"]):
        base = deterministic_fixture_sample(f"positive:{case_index:04d}:base")
        transformed = _transform_iq(
            base,
            case_index=case_index,
            gain=gains[case_index % len(gains)],
            shift=_shift(case_index),
        )
        positives.append(max_abs_normalized_circular_correlation(base, transformed))

    seed = calibration["negative_seed"]
    negatives: list[float] = []
    for case_index in range(calibration["negative_cases"]):
        left = deterministic_fixture_sample(f"negative:{seed}:left:{case_index:04d}")
        right = deterministic_fixture_sample(f"negative:{seed}:right:{case_index:04d}")
        negatives.append(max_abs_normalized_circular_correlation(left, right))
    return positives, negatives


def _calibration_report(contract: dict[str, Any]) -> dict[str, object]:
    positives, negatives = _calibration_scores(contract)
    threshold = _floor_score(min(positives))
    positive_hits = sum(score >= threshold for score in positives)
    negative_hits = sum(score >= threshold for score in negatives)
    positive_recall = positive_hits / len(positives)
    negative_fpr = negative_hits / len(negatives)

    requirements = contract["threshold_selection"]
    passed = (
        positive_recall >= requirements["positive_recall_required"]
        and negative_fpr <= requirements["negative_false_positive_rate_max"]
    )
    return {
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "threshold_rule": requirements["rule"],
        "selected_threshold": f"{threshold:.12f}",
        "positive_recall": _format_score(positive_recall),
        "negative_false_positive_rate": _format_score(negative_fpr),
        "positive_scores": _score_summary(positives),
        "negative_scores": _score_summary(negatives),
        "passed": passed,
    }


def _bounded_samples(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    audit = contract["bounded_reference_audit"]
    gains = _gain_cycle(contract)
    samples: dict[str, np.ndarray] = {}
    for index in range(audit["base_transform_pairs"]):
        base = deterministic_fixture_sample(f"audit:base:{index:04d}")
        samples[f"base:{index:04d}"] = base
        samples[f"transform:{index:04d}"] = _transform_iq(
            base,
            case_index=index,
            gain=gains[index % len(gains)],
            shift=_shift(index),
        )
    for index in range(audit["unrelated_samples"]):
        samples[f"unrelated:{index:04d}"] = deterministic_fixture_sample(
            f"audit:unrelated:{index:04d}"
        )
    if len(samples) != audit["sample_count"]:
        raise NearDuplicateFixtureError("Bounded audit sample count mismatch")
    return samples


def _pair_digest(left_id: str, right_id: str, score: float) -> str:
    return _json_digest({"left": left_id, "right": right_id, "score": _format_score(score)})


def _bounded_reference_report(contract: dict[str, Any], threshold: float) -> dict[str, object]:
    samples = _bounded_samples(contract)
    sample_ids = sorted(samples)
    expected_positive_pairs = {
        (f"base:{index:04d}", f"transform:{index:04d}")
        for index in range(contract["bounded_reference_audit"]["base_transform_pairs"])
    }
    pair_scores: list[dict[str, object]] = []
    discovered: set[tuple[str, str]] = set()
    for left_offset, left_id in enumerate(sample_ids):
        for right_id in sample_ids[left_offset + 1 :]:
            score = max_abs_normalized_circular_correlation(samples[left_id], samples[right_id])
            pair = (left_id, right_id)
            if score >= threshold:
                discovered.add(pair)
            pair_scores.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "score": _format_score(score),
                    "pair_digest": _pair_digest(left_id, right_id, score),
                }
            )
    false_negatives = sorted(expected_positive_pairs - discovered)
    false_positives = sorted(discovered - expected_positive_pairs)
    recall = (len(expected_positive_pairs) - len(false_negatives)) / len(expected_positive_pairs)
    audit_contract = contract["bounded_reference_audit"]
    passed = (
        len(pair_scores) == audit_contract["expected_pair_count"]
        and recall >= audit_contract["candidate_recall_required"]
        and not false_negatives
        and not audit_contract["production_claim"]
    )
    return {
        "algorithm": audit_contract["algorithm"],
        "sample_count": len(samples),
        "pair_count": len(pair_scores),
        "threshold": f"{threshold:.12f}",
        "expected_positive_pairs": len(expected_positive_pairs),
        "discovered_pairs_at_threshold": len(discovered),
        "candidate_recall": _format_score(recall),
        "false_negative_pairs": [[left, right] for left, right in false_negatives],
        "false_positive_pairs": [[left, right] for left, right in false_positives],
        "pair_scores_sha256": _json_digest(pair_scores),
        "production_claim": audit_contract["production_claim"],
        "passed": passed,
    }


def build_near_duplicate_fixture_evidence(
    fixture_contract_path: Path,
    near_duplicate_contract_path: Path,
    dataset_spec_path: Path,
    conversion_contract_path: Path,
) -> dict[str, object]:
    """Build deterministic calibration and bounded exhaustive-reference evidence."""

    contract = load_near_duplicate_fixture_contract(
        fixture_contract_path,
        near_duplicate_contract_path,
        dataset_spec_path,
        conversion_contract_path,
    )
    calibration = _calibration_report(contract)
    threshold = float(calibration["selected_threshold"])
    bounded_audit = _bounded_reference_report(contract, threshold)
    evidence = {
        "schema_version": 1,
        "fixture_contract_id": contract["fixture_contract_id"],
        "fixture_contract_sha256": near_duplicate_fixture_contract_sha256(fixture_contract_path),
        "near_duplicate_contract_id": contract["near_duplicate_contract"]["contract_id"],
        "near_duplicate_contract_sha256": contract["near_duplicate_contract"]["contract_sha256"],
        "calibration": calibration,
        "bounded_reference_audit": bounded_audit,
        "generation_gate": contract["generation_gate"],
        "writes_artifacts": contract["publication"]["writes_artifacts"],
    }
    evidence["evidence_sha256"] = _json_digest(evidence)
    if not calibration["passed"] or not bounded_audit["passed"]:
        raise NearDuplicateFixtureError("Near-duplicate fixture evidence did not satisfy its gates")
    return evidence
