"""Generate and validate deterministic RadioML split artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from na_lmscnet.data.conversion_contract import (
    conversion_contract_sha256,
    conversion_row_index,
    conversion_sample_id,
    load_conversion_contract,
)
from na_lmscnet.data.split_contract import (
    assign_stratum_sample_ids,
    load_split_contract,
    split_contract_sha256,
)

SPLIT_NAMES = ("train", "validation", "test")
MAX_SPLIT_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_LEAKAGE_AUDIT_BYTES = 1024 * 1024


class SplitManifestError(ValueError):
    """Raised when split artifacts are incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_json(path: Path, *, limit: int, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SplitManifestError(f"{field} must be a regular file")
    if path.stat().st_size > limit:
        raise SplitManifestError(f"{field} exceeds {limit} bytes")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SplitManifestError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise SplitManifestError(f"{field} must contain a JSON object")
    return value


def validate_split_assignments(
    assignments: object, expected_counts: dict[str, int], total_samples: int
) -> dict[str, list[int]]:
    """Validate sorted, disjoint row lists that cover the HDF5 artifact exactly once."""

    if not isinstance(assignments, dict) or set(assignments) != set(SPLIT_NAMES):
        raise SplitManifestError("assignments must contain train, validation, and test")
    validated: dict[str, list[int]] = {}
    all_rows: list[int] = []
    for split in SPLIT_NAMES:
        rows = assignments[split]
        if not isinstance(rows, list) or any(type(row) is not int for row in rows):
            raise SplitManifestError(f"assignments.{split} must be an integer list")
        if len(rows) != expected_counts[split]:
            raise SplitManifestError(f"assignments.{split} has an unexpected count")
        if rows != sorted(rows) or len(rows) != len(set(rows)):
            raise SplitManifestError(f"assignments.{split} must be strictly sorted and unique")
        if rows and (rows[0] < 0 or rows[-1] >= total_samples):
            raise SplitManifestError(f"assignments.{split} contains an out-of-range row")
        validated[split] = rows
        all_rows.extend(rows)
    if len(all_rows) != total_samples or set(all_rows) != set(range(total_samples)):
        raise SplitManifestError("Split assignments must cover every HDF5 row exactly once")
    return validated


def _build_assignments(
    split_contract: dict[str, Any], conversion_contract: dict[str, Any]
) -> dict[str, list[int]]:
    assignments = {name: [] for name in SPLIT_NAMES}
    for modulation in split_contract["stratification"]["modulation_order"]:
        for snr_db in split_contract["stratification"]["snr_db_order"]:
            sample_ids = [
                conversion_sample_id(conversion_contract, modulation, snr_db, source_index)
                for source_index in range(split_contract["stratification"]["samples_per_stratum"])
            ]
            stratum = assign_stratum_sample_ids(split_contract, sample_ids)
            for split, identifiers in stratum.items():
                rows = [
                    conversion_row_index(
                        conversion_contract,
                        modulation,
                        snr_db,
                        int(identifier.rsplit(":", maxsplit=1)[1]),
                    )
                    for identifier in identifiers
                ]
                assignments[split].extend(rows)
    for rows in assignments.values():
        rows.sort()
    return assignments


def _audit_exact_duplicates(
    hdf5_path: Path, assignments: dict[str, list[int]]
) -> dict[str, object]:
    total_samples = sum(len(rows) for rows in assignments.values())
    split_codes = np.empty(total_samples, dtype=np.uint8)
    for split_code, split in enumerate(SPLIT_NAMES):
        split_codes[np.asarray(assignments[split], dtype=np.int64)] = split_code

    seen: dict[bytes, tuple[int, int]] = {}
    duplicate_groups: dict[bytes, list[tuple[int, int]]] = {}
    with h5py.File(hdf5_path, "r", libver="earliest", swmr=False) as file:
        iq = file["/iq"]
        for start in range(0, total_samples, 2048):
            block = np.asarray(iq[start : start + 2048], dtype=np.dtype("<f4"))
            for offset, sample in enumerate(block):
                row = start + offset
                digest = hashlib.sha256(sample.tobytes(order="C")).digest()
                split_code = int(split_codes[row])
                first = seen.get(digest)
                if first is None:
                    seen[digest] = (row, split_code)
                else:
                    duplicate_groups.setdefault(digest, [first]).append((row, split_code))

    groups = []
    cross_split = 0
    within_split = 0
    for digest, members in sorted(duplicate_groups.items()):
        splits = {split_code for _, split_code in members}
        if len(splits) > 1:
            cross_split += 1
        else:
            within_split += 1
        groups.append(
            {
                "sha256": digest.hex(),
                "rows": [row for row, _ in members],
                "splits": [SPLIT_NAMES[split_code] for _, split_code in members],
            }
        )
    return {
        "representation": "canonical-little-endian-float32-iq-bytes",
        "samples_scanned": total_samples,
        "duplicate_groups": groups,
        "cross_split_group_count": cross_split,
        "within_split_group_count": within_split,
        "passed": cross_split == 0,
    }


def _validate_source_bindings(
    hdf5_path: Path,
    conversion_manifest_path: Path,
    split_contract: dict[str, Any],
) -> dict[str, Any]:
    source = split_contract["source"]
    hdf5_binding = source["hdf5"]
    conversion_binding = source["conversion_manifest"]
    if hdf5_path.name != hdf5_binding["filename"] or not hdf5_path.is_file():
        raise SplitManifestError("HDF5 artifact filename or type differs from the split contract")
    if hdf5_path.stat().st_size != hdf5_binding["size_bytes"]:
        raise SplitManifestError("HDF5 artifact size differs from the split contract")
    if _sha256_file(hdf5_path) != hdf5_binding["file_sha256"]:
        raise SplitManifestError("HDF5 artifact SHA-256 differs from the split contract")
    if (
        conversion_manifest_path.name != conversion_binding["filename"]
        or _sha256_file(conversion_manifest_path) != conversion_binding["file_sha256"]
    ):
        raise SplitManifestError("Conversion manifest identity differs from the split contract")
    manifest = _load_json(
        conversion_manifest_path,
        limit=MAX_LEAKAGE_AUDIT_BYTES,
        field="conversion manifest",
    )
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("output_file_sha256") != hdf5_binding["file_sha256"]
    ):
        raise SplitManifestError("Conversion manifest does not bind the HDF5 artifact")
    if artifacts.get("output_logical_content_sha256") != hdf5_binding["logical_content_sha256"]:
        raise SplitManifestError(
            "Conversion manifest logical digest differs from the split contract"
        )
    return manifest


def _write_json_atomic(value: object, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise SplitManifestError(f"Refusing to overwrite existing artifact: {destination.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def generate_split_artifacts(
    *,
    hdf5_path: Path,
    conversion_manifest_path: Path,
    output_dir: Path,
    split_contract_path: Path,
    dataset_spec_path: Path,
    conversion_contract_path: Path,
    project_root: Path,
    project_commit: str,
) -> dict[str, object]:
    """Generate deterministic split and leakage-audit artifacts outside the repository."""

    split_contract = load_split_contract(
        split_contract_path, dataset_spec_path, conversion_contract_path
    )
    if not split_contract["generation_gate"]["split_generation_enabled"]:
        raise SplitManifestError("Split generation is disabled by the contract")
    conversion_contract = load_conversion_contract(conversion_contract_path, dataset_spec_path)
    _validate_source_bindings(hdf5_path, conversion_manifest_path, split_contract)
    output_dir = output_dir.resolve(strict=True)
    if not output_dir.is_dir():
        raise SplitManifestError("Output directory must already exist")
    project_root = project_root.resolve(strict=True)
    if output_dir == project_root or project_root in output_dir.parents:
        raise SplitManifestError("Split artifacts must be written outside the repository")
    if len(project_commit) != 40 or any(c not in "0123456789abcdef" for c in project_commit):
        raise SplitManifestError("project_commit must be a full lowercase Git commit")

    assignments = _build_assignments(split_contract, conversion_contract)
    expected_counts = split_contract["stratification"]["expected_totals"]
    total_samples = sum(expected_counts.values())
    validate_split_assignments(assignments, expected_counts, total_samples)
    exact_audit = _audit_exact_duplicates(hdf5_path, assignments)
    if not exact_audit["passed"]:
        raise SplitManifestError("Exact duplicate samples cross split boundaries")

    assignment_sha256 = _json_sha256(assignments)
    digests = {
        "split_contract_sha256": split_contract_sha256(split_contract_path),
        "dataset_spec_sha256": _sha256_file(dataset_spec_path),
        "conversion_contract_sha256": conversion_contract_sha256(conversion_contract_path),
        "conversion_manifest_sha256": _sha256_file(conversion_manifest_path),
        "source_archive_sha256": split_contract["source"]["source_archive_sha256"],
        "source_dataset_content_sha256": split_contract["source"]["source_dataset_content_sha256"],
        "hdf5_file_sha256": split_contract["source"]["hdf5"]["file_sha256"],
        "hdf5_logical_content_sha256": split_contract["source"]["hdf5"]["logical_content_sha256"],
        "assignment_sha256": assignment_sha256,
    }
    manifest = {
        "schema_version": 1,
        "contract_id": split_contract["contract_id"],
        "dataset_id": split_contract["dataset_id"],
        "source": {
            "hdf5_filename": hdf5_path.name,
            "conversion_manifest_filename": conversion_manifest_path.name,
        },
        "counts": expected_counts,
        "assignments": assignments,
        "digests": digests,
        "environment": {
            "project_commit": project_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
        },
        "test_isolation": split_contract["test_isolation"],
    }
    manifest_sha256 = _json_sha256(manifest)
    leakage_audit = {
        "schema_version": 1,
        "dataset_id": split_contract["dataset_id"],
        "split_manifest_sha256": manifest_sha256,
        "assignment_sha256": assignment_sha256,
        "exact_duplicates": exact_audit,
        "near_duplicates": {
            "bounded_fixture_verified": True,
            "global_production_audit_completed": False,
            "limitation": "transformed near-duplicate leakage is not globally excluded",
        },
        "adjacent_windows": {
            "audit_completed": False,
            "limitation": "capture/session/window provenance is unavailable",
        },
        "passed_for_approved_split_protocol": True,
    }
    manifest_path = output_dir / split_contract["manifest"]["filename"]
    audit_path = output_dir / split_contract["manifest"]["leakage_audit_filename"]
    _write_json_atomic(manifest, manifest_path)
    try:
        _write_json_atomic(leakage_audit, audit_path)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    return {
        "manifest_path": manifest_path,
        "manifest_file_sha256": _sha256_file(manifest_path),
        "leakage_audit_path": audit_path,
        "leakage_audit_file_sha256": _sha256_file(audit_path),
        "counts": expected_counts,
        "assignment_sha256": assignment_sha256,
        "exact_duplicate_groups": len(exact_audit["duplicate_groups"]),
    }


def load_split_artifacts(
    *,
    manifest_path: Path,
    leakage_audit_path: Path,
    hdf5_path: Path,
    conversion_manifest_path: Path,
    split_contract_path: Path,
    dataset_spec_path: Path,
    conversion_contract_path: Path,
) -> dict[str, Any]:
    """Validate split artifacts and their source bindings for dataset loading."""

    contract = load_split_contract(split_contract_path, dataset_spec_path, conversion_contract_path)
    _validate_source_bindings(hdf5_path, conversion_manifest_path, contract)
    manifest = _load_json(manifest_path, limit=MAX_SPLIT_MANIFEST_BYTES, field="split manifest")
    audit = _load_json(leakage_audit_path, limit=MAX_LEAKAGE_AUDIT_BYTES, field="leakage audit")
    if set(manifest) != {
        "schema_version",
        "contract_id",
        "dataset_id",
        "source",
        "counts",
        "assignments",
        "digests",
        "environment",
        "test_isolation",
    }:
        raise SplitManifestError("Split manifest fields differ from schema version 1")
    if (
        manifest["schema_version"] != 1
        or manifest["contract_id"] != contract["contract_id"]
        or manifest["dataset_id"] != contract["dataset_id"]
        or manifest["source"]
        != {
            "hdf5_filename": hdf5_path.name,
            "conversion_manifest_filename": conversion_manifest_path.name,
        }
        or manifest["counts"] != contract["stratification"]["expected_totals"]
        or manifest["test_isolation"] != contract["test_isolation"]
    ):
        raise SplitManifestError("Split manifest identity or policy differs from the contract")
    assignments = validate_split_assignments(
        manifest["assignments"], manifest["counts"], sum(manifest["counts"].values())
    )
    expected_digests = {
        "split_contract_sha256": split_contract_sha256(split_contract_path),
        "dataset_spec_sha256": _sha256_file(dataset_spec_path),
        "conversion_contract_sha256": conversion_contract_sha256(conversion_contract_path),
        "conversion_manifest_sha256": _sha256_file(conversion_manifest_path),
        "source_archive_sha256": contract["source"]["source_archive_sha256"],
        "source_dataset_content_sha256": contract["source"]["source_dataset_content_sha256"],
        "hdf5_file_sha256": contract["source"]["hdf5"]["file_sha256"],
        "hdf5_logical_content_sha256": contract["source"]["hdf5"]["logical_content_sha256"],
        "assignment_sha256": _json_sha256(assignments),
    }
    if manifest["digests"] != expected_digests:
        raise SplitManifestError("Split manifest digest bindings do not match current artifacts")
    exact = audit.get("exact_duplicates")
    if (
        set(audit)
        != {
            "schema_version",
            "dataset_id",
            "split_manifest_sha256",
            "assignment_sha256",
            "exact_duplicates",
            "near_duplicates",
            "adjacent_windows",
            "passed_for_approved_split_protocol",
        }
        or audit["schema_version"] != 1
        or audit["dataset_id"] != contract["dataset_id"]
        or audit["split_manifest_sha256"] != _json_sha256(manifest)
        or audit["assignment_sha256"] != expected_digests["assignment_sha256"]
        or not isinstance(exact, dict)
        or exact.get("samples_scanned") != sum(manifest["counts"].values())
        or exact.get("cross_split_group_count") != 0
        or exact.get("passed") is not True
        or audit["passed_for_approved_split_protocol"] is not True
    ):
        raise SplitManifestError("Leakage audit does not authorize the approved split protocol")
    return {"manifest": manifest, "leakage_audit": audit, "contract": contract}
