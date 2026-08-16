"""Source-bound RadioML 2018.01A audit, split, and dataset adapter."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from na_lmscnet.data.contracts import ModulationSample, make_sample
from na_lmscnet.data.preprocessing import PreprocessingError, preprocess_iq

SPLITS = ("train", "validation", "test")
TRAINING_SPLITS = ("train", "validation")
MAX_CONFIG_BYTES = 128 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


class RadioML2018Error(ValueError):
    """Raised when a RadioML 2018 artifact or access request is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_sha256(path: Path, expected: str) -> str:
    stat = path.stat()
    key = (str(path.resolve(strict=True)), stat.st_size, stat.st_mtime_ns)
    observed = _SHA256_CACHE.get(key)
    if observed is None:
        observed = _sha256_file(path)
        _SHA256_CACHE[key] = observed
    if observed != expected:
        raise RadioML2018Error(f"SHA-256 differs for {path.name}")
    return observed


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_yaml(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
        raise RadioML2018Error(f"{field} must be a bounded regular file")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RadioML2018Error(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise RadioML2018Error(f"{field} must contain a mapping")
    return value


def _load_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise RadioML2018Error(f"{field} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RadioML2018Error(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise RadioML2018Error(f"{field} must contain an object")
    return value


def _write_json_atomic(value: object, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RadioML2018Error(f"Refusing to overwrite {destination.name}")
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


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _require_external_output(output_dir: Path, project_root: Path) -> Path:
    output = output_dir.resolve(strict=True)
    root = project_root.resolve(strict=True)
    if not output.is_dir() or output == root or root in output.parents:
        raise RadioML2018Error("Artifacts must be written to an existing external directory")
    return output


def _validate_dataset_spec(spec: dict[str, Any]) -> None:
    expected = spec.get("expected")
    source = spec.get("source")
    if (
        spec.get("schema_version") != 1
        or spec.get("dataset_id") != "radioml_2018_01a"
        or not isinstance(expected, dict)
        or not isinstance(source, dict)
    ):
        raise RadioML2018Error("Dataset spec identity or structure is invalid")
    required = {
        "archive_filename": "2018.01.OSC.0001_1024x2M.h5.tar.gz",
        "archive_size_bytes": 19_342_413_140,
        "archive_sha256": "90725106c5fb08ad603d55ab22864eb0c80869f861a14e3c7835180957af8fd3",
        "hdf5_filename": "GOLD_XYZ_OSC.0001_1024.hdf5",
        "hdf5_size_bytes": 21_449_148_312,
        "hdf5_sha256": "e3dd0bef66a3426959ee66a1709a8c0a95d4f8395d18aaf6f1214bdbc763bd38",
    }
    if any(source.get(key) != value for key, value in required.items()):
        raise RadioML2018Error("Dataset spec source bindings differ from the audited artifact")
    if (
        expected.get("x_shape") != [2_555_904, 1024, 2]
        or expected.get("y_shape") != [2_555_904, 24]
        or expected.get("z_shape") != [2_555_904, 1]
        or expected.get("samples_per_stratum") != 4096
        or len(expected.get("modulations", [])) != 24
        or expected.get("snr_db") != list(range(-20, 32, 2))
    ):
        raise RadioML2018Error("Dataset spec expected schema is invalid")


def _validate_split_contract(contract: dict[str, Any], spec: dict[str, Any]) -> None:
    stratification = contract.get("stratification")
    assignment = contract.get("assignment")
    test_isolation = contract.get("test_isolation")
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_id") != "radioml_2018_01a_split_v1"
        or contract.get("dataset_id") != spec["dataset_id"]
        or not isinstance(stratification, dict)
        or not isinstance(assignment, dict)
        or not isinstance(test_isolation, dict)
    ):
        raise RadioML2018Error("Split contract identity or structure is invalid")
    expected = spec["expected"]
    per_stratum = {"train": 2867, "validation": 410, "test": 819}
    totals = {split: count * 624 for split, count in per_stratum.items()}
    if (
        stratification.get("fields") != ["class_index", "snr_db"]
        or stratification.get("class_count") != 24
        or stratification.get("snr_db") != expected["snr_db"]
        or stratification.get("strata") != 624
        or stratification.get("samples_per_stratum") != 4096
        or stratification.get("split_order") != list(SPLITS)
        or stratification.get("ratio_weights")
        != {"train": 7, "validation": 1, "test": 2}
        or stratification.get("rounding")
        != {"algorithm": "largest-remainder-v1", "tie_break": "split-order"}
        or stratification.get("expected_per_stratum") != per_stratum
        or stratification.get("expected_totals") != totals
    ):
        raise RadioML2018Error("Split contract stratification differs from the frozen protocol")
    if assignment != {
        "seed": 2026,
        "algorithm": "sha256-rank-v1",
        "independently_rank_each_stratum": True,
        "sample_identity": "source-coordinate-v1",
        "ranking_domain": "na-lmscnet/radioml_2018_01a/split-ranking/v1",
    }:
        raise RadioML2018Error("Split assignment differs from the frozen protocol")
    if test_isolation != {
        "training_allowed_splits": ["train"],
        "tuning_allowed_splits": ["train", "validation"],
        "test_access_requires": "experiment-freeze-manifest-v1",
        "test_metrics_before_freeze": "forbidden",
    }:
        raise RadioML2018Error("Split test-isolation policy differs from the frozen protocol")


def _validate_hdf5_schema(hdf5_path: Path, spec: dict[str, Any]) -> dict[str, object]:
    expected = spec["expected"]
    with h5py.File(hdf5_path, "r", libver="earliest", swmr=False) as file:
        if set(file.keys()) != {"X", "Y", "Z"}:
            raise RadioML2018Error("HDF5 root datasets must be exactly X, Y, and Z")
        definitions = {
            "X": (tuple(expected["x_shape"]), np.dtype("float32")),
            "Y": (tuple(expected["y_shape"]), np.dtype("int64")),
            "Z": (tuple(expected["z_shape"]), np.dtype("int64")),
        }
        observed: dict[str, object] = {}
        for name, (shape, dtype) in definitions.items():
            dataset = file[name]
            if dataset.shape != shape or dataset.dtype != dtype:
                raise RadioML2018Error(f"HDF5 {name} shape or dtype differs from the spec")
            if dataset.compression is not None or dataset.chunks is not None:
                raise RadioML2018Error(f"HDF5 {name} physical layout differs from the source")
            observed[name] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
        return observed


def _validate_source_manifest(
    manifest: dict[str, Any], spec: dict[str, Any], dataset_spec_path: Path
) -> None:
    source = manifest.get("source")
    bindings = manifest.get("bindings")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset_id") != spec["dataset_id"]
        or manifest.get("purpose") != "radioml_2018_01a_source_schema_audit"
        or manifest.get("test_accessed") is not False
        or not isinstance(source, dict)
        or not isinstance(bindings, dict)
    ):
        raise RadioML2018Error("Source manifest identity or structure is invalid")
    expected_source = spec["source"]
    hdf5 = source.get("hdf5")
    archive = source.get("archive")
    if (
        not isinstance(hdf5, dict)
        or not isinstance(archive, dict)
        or archive.get("filename") != expected_source["archive_filename"]
        or archive.get("size_bytes") != expected_source["archive_size_bytes"]
        or archive.get("sha256") != expected_source["archive_sha256"]
        or hdf5.get("filename") != expected_source["hdf5_filename"]
        or hdf5.get("size_bytes") != expected_source["hdf5_size_bytes"]
        or hdf5.get("sha256") != expected_source["hdf5_sha256"]
        or bindings.get("dataset_spec_sha256") != _sha256_file(dataset_spec_path)
    ):
        raise RadioML2018Error("Source manifest bindings differ from the dataset spec")


def _validate_split_manifest(
    manifest: dict[str, Any],
    *,
    spec: dict[str, Any],
    source_manifest_path: Path,
    dataset_spec_path: Path,
) -> None:
    counts = manifest.get("counts")
    assignment = manifest.get("assignment")
    artifact = manifest.get("artifact")
    bindings = manifest.get("bindings")
    isolation = manifest.get("test_isolation")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset_id") != spec["dataset_id"]
        or manifest.get("purpose") != "radioml_2018_01a_frozen_split"
        or manifest.get("test_accessed") is not False
        or not all(isinstance(value, dict) for value in (counts, assignment, artifact, bindings))
    ):
        raise RadioML2018Error("Split manifest identity or structure is invalid")
    if (
        set(counts) != set(SPLITS)
        or sum(int(counts[split]) for split in SPLITS) != spec["expected"]["x_shape"][0]
        or assignment.get("seed") != 2026
        or assignment.get("algorithm") != "sha256-rank-v1"
        or not isinstance(assignment.get("sha256"), str)
        or len(assignment["sha256"]) != 64
        or bindings.get("source_manifest_sha256") != _sha256_file(source_manifest_path)
        or bindings.get("dataset_spec_sha256") != _sha256_file(dataset_spec_path)
        or bindings.get("source_hdf5_sha256") != spec["source"]["hdf5_sha256"]
        or isolation
        != {
            "training_allowed_splits": ["train"],
            "tuning_allowed_splits": ["train", "validation"],
            "test_access_requires": "experiment-freeze-manifest-v1",
            "test_adapter_available": False,
        }
    ):
        raise RadioML2018Error("Split manifest bindings or isolation policy are invalid")


def audit_radioml_2018_source(
    *,
    archive_path: Path,
    hdf5_path: Path,
    classes_path: Path,
    license_path: Path,
    dataset_spec_path: Path,
    output_dir: Path,
    project_root: Path,
    project_commit: str,
) -> dict[str, object]:
    """Verify the downloaded source and publish a no-overwrite source manifest."""

    output = _require_external_output(output_dir, project_root)
    spec = _load_yaml(dataset_spec_path, "dataset spec")
    _validate_dataset_spec(spec)
    for path, name in (
        (archive_path, "archive"),
        (hdf5_path, "HDF5"),
        (classes_path, "classes"),
        (license_path, "license"),
    ):
        if path.is_symlink() or not path.is_file():
            raise RadioML2018Error(f"{name} must be a regular file")
    source = spec["source"]
    if (
        archive_path.name != source["archive_filename"]
        or archive_path.stat().st_size != source["archive_size_bytes"]
        or _verified_sha256(archive_path, source["archive_sha256"]) != source["archive_sha256"]
        or hdf5_path.name != source["hdf5_filename"]
        or hdf5_path.stat().st_size != source["hdf5_size_bytes"]
        or _verified_sha256(hdf5_path, source["hdf5_sha256"]) != source["hdf5_sha256"]
    ):
        raise RadioML2018Error("Source archive or HDF5 identity differs from the dataset spec")
    observed = _validate_hdf5_schema(hdf5_path, spec)
    classes_sha256 = _sha256_file(classes_path)
    license_sha256 = _sha256_file(license_path)

    expected = spec["expected"]
    try:
        classes_text = classes_path.read_text(encoding="utf-8")
        assignment = classes_text.split("=", maxsplit=1)
        classes = ast.literal_eval(assignment[1]) if len(assignment) == 2 else None
        license_text = license_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, SyntaxError, ValueError) as error:
        raise RadioML2018Error(f"Could not validate source metadata: {error}") from error
    if classes != expected["modulations"]:
        raise RadioML2018Error("classes.txt order differs from the dataset spec")
    if "Attribution-NonCommercial-ShareAlike 4.0 International" not in license_text:
        raise RadioML2018Error("License text does not identify CC BY-NC-SA 4.0")
    class_counts = np.zeros(24, dtype=np.int64)
    pair_counts: dict[tuple[int, int], int] = {}
    finite_x = True
    with h5py.File(hdf5_path, "r", libver="earliest", swmr=False) as file:
        sample_count = file["X"].shape[0]
        for start in range(0, sample_count, 8192):
            stop = min(start + 8192, sample_count)
            x = np.asarray(file["X"][start:stop])
            y = np.asarray(file["Y"][start:stop])
            z = np.asarray(file["Z"][start:stop]).reshape(-1)
            finite_x = finite_x and bool(np.isfinite(x).all())
            if not np.array_equal(y.sum(axis=1), np.ones(len(y), dtype=np.int64)):
                raise RadioML2018Error("Y is not strict one-hot metadata")
            labels = y.argmax(axis=1)
            row_numbers = np.arange(start, stop, dtype=np.int64)
            expected_labels = row_numbers // (len(expected["snr_db"]) * 4096)
            expected_snr_indices = (row_numbers // 4096) % len(expected["snr_db"])
            expected_snr = np.asarray(expected["snr_db"], dtype=np.int64)[expected_snr_indices]
            if not np.array_equal(labels, expected_labels) or not np.array_equal(z, expected_snr):
                raise RadioML2018Error("HDF5 row ordering differs from the audited source")
            class_counts += np.bincount(labels, minlength=24)
            for pair, count in zip(
                *np.unique(np.stack((labels, z), axis=1), axis=0, return_counts=True),
                strict=True,
            ):
                key = (int(pair[0]), int(pair[1]))
                pair_counts[key] = pair_counts.get(key, 0) + int(count)
    if not finite_x:
        raise RadioML2018Error("X contains non-finite values")
    if class_counts.tolist() != [106_496] * 24:
        raise RadioML2018Error("Class counts differ from the audited source")
    expected_pairs = {
        (class_index, snr): expected["samples_per_stratum"]
        for class_index in range(24)
        for snr in expected["snr_db"]
    }
    if pair_counts != expected_pairs:
        raise RadioML2018Error("Class/SNR strata differ from the audited source")

    manifest = {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "purpose": "radioml_2018_01a_source_schema_audit",
        "source": {
            "archive": {
                "filename": archive_path.name,
                "size_bytes": archive_path.stat().st_size,
                "sha256": source["archive_sha256"],
            },
            "hdf5": {
                "filename": hdf5_path.name,
                "size_bytes": hdf5_path.stat().st_size,
                "sha256": source["hdf5_sha256"],
            },
            "classes": {"filename": classes_path.name, "sha256": classes_sha256},
            "license": {"filename": license_path.name, "sha256": license_sha256},
        },
        "schema": observed,
        "ordering": {
            "fields": ["class_index", "snr_db", "source_index"],
            "modulations": expected["modulations"],
            "snr_db": expected["snr_db"],
            "samples_per_stratum": expected["samples_per_stratum"],
        },
        "audit": {
            "all_x_values_finite": True,
            "strict_one_hot_labels": True,
            "strata": len(pair_counts),
            "samples": int(class_counts.sum()),
        },
        "bindings": {
            "dataset_spec_sha256": _sha256_file(dataset_spec_path),
            "project_commit": project_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
        },
        "test_accessed": False,
    }
    _write_json_atomic(manifest, output / "RML2018.01A.source-manifest.json")
    return manifest


def sample_id(class_index: int, snr_db: int, source_index: int) -> str:
    """Return the stable source-coordinate identity for one 2018.01A row."""

    return f"radioml_2018_01a:{class_index:02d}:{snr_db:+03d}:{source_index:04d}"


def _rank_digest(seed: int, identifier: str) -> bytes:
    fields = (
        "na-lmscnet/radioml_2018_01a/split-ranking/v1",
        str(seed),
        identifier,
    )
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _build_split_rows(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    stratification = contract["stratification"]
    seed = int(contract["assignment"]["seed"])
    snr_values = stratification["snr_db"]
    per_split = stratification["expected_per_stratum"]
    samples_per_stratum = int(stratification["samples_per_stratum"])
    rows: dict[str, list[int]] = {split: [] for split in SPLITS}
    for class_index in range(int(stratification["class_count"])):
        for snr_index, snr_db in enumerate(snr_values):
            ranked = sorted(
                range(samples_per_stratum),
                key=lambda source_index: (
                    _rank_digest(seed, sample_id(class_index, int(snr_db), source_index)),
                    source_index,
                ),
            )
            boundaries = (
                int(per_split["train"]),
                int(per_split["train"]) + int(per_split["validation"]),
            )
            assigned = {
                "train": ranked[: boundaries[0]],
                "validation": ranked[boundaries[0] : boundaries[1]],
                "test": ranked[boundaries[1] :],
            }
            base = (class_index * len(snr_values) + snr_index) * samples_per_stratum
            for split in SPLITS:
                rows[split].extend(base + source_index for source_index in assigned[split])
    return {split: np.asarray(sorted(values), dtype="<i8") for split, values in rows.items()}


def _assignment_sha256(rows: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for split in SPLITS:
        name = split.encode("ascii")
        values = np.asarray(rows[split], dtype="<i8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(values).to_bytes(8, "big"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _split_codes(rows: dict[str, np.ndarray], sample_count: int) -> np.ndarray:
    codes = np.full(sample_count, 255, dtype=np.uint8)
    for code, split in enumerate(SPLITS):
        if np.any(codes[rows[split]] != 255):
            raise RadioML2018Error("Split rows overlap")
        codes[rows[split]] = code
    if bool(np.any(codes == 255)):
        raise RadioML2018Error("Split rows do not cover the source")
    return codes


def _audit_exact_duplicates(hdf5_path: Path, codes: np.ndarray) -> dict[str, object]:
    seen: dict[bytes, tuple[int, int]] = {}
    groups: dict[bytes, list[tuple[int, int]]] = {}
    with h5py.File(hdf5_path, "r", libver="earliest", swmr=False) as file:
        x = file["X"]
        for start in range(0, len(codes), 2048):
            block = np.asarray(x[start : start + 2048], dtype="<f4")
            for offset, value in enumerate(block):
                row = start + offset
                digest = hashlib.sha256(value.tobytes(order="C")).digest()
                member = (row, int(codes[row]))
                first = seen.get(digest)
                if first is None:
                    seen[digest] = member
                else:
                    groups.setdefault(digest, [first]).append(member)
    cross_split = sum(1 for members in groups.values() if len({code for _, code in members}) > 1)
    return {
        "representation": "source-float32-1024x2-bytes",
        "samples_scanned": len(codes),
        "duplicate_group_count": len(groups),
        "cross_split_group_count": cross_split,
        "within_split_group_count": len(groups) - cross_split,
        "passed": cross_split == 0,
    }


def generate_radioml_2018_split(
    *,
    hdf5_path: Path,
    source_manifest_path: Path,
    dataset_spec_path: Path,
    split_contract_path: Path,
    output_dir: Path,
    project_root: Path,
    project_commit: str,
    audit_exact_duplicates: bool = True,
) -> dict[str, object]:
    """Publish compact deterministic split rows and their manifest outside the repo."""

    output = _require_external_output(output_dir, project_root)
    spec = _load_yaml(dataset_spec_path, "dataset spec")
    contract = _load_yaml(split_contract_path, "split contract")
    source_manifest = _load_json(source_manifest_path, "source manifest")
    _validate_dataset_spec(spec)
    _validate_split_contract(contract, spec)
    _validate_source_manifest(source_manifest, spec, dataset_spec_path)
    source_binding = source_manifest.get("source", {}).get("hdf5", {})
    if (
        hdf5_path.name != source_binding.get("filename")
        or hdf5_path.stat().st_size != source_binding.get("size_bytes")
        or source_binding.get("sha256") != spec["source"]["hdf5_sha256"]
    ):
        raise RadioML2018Error("HDF5 differs from the audited source manifest")
    _verified_sha256(hdf5_path, spec["source"]["hdf5_sha256"])
    _validate_hdf5_schema(hdf5_path, spec)
    rows = _build_split_rows(contract)
    expected_totals = contract["stratification"]["expected_totals"]
    if any(len(rows[split]) != expected_totals[split] for split in SPLITS):
        raise RadioML2018Error("Generated split counts differ from the contract")
    sample_count = int(spec["expected"]["x_shape"][0])
    codes = _split_codes(rows, sample_count)
    assignment_sha256 = _assignment_sha256(rows)

    duplicate_audit = (
        _audit_exact_duplicates(hdf5_path, codes)
        if audit_exact_duplicates
        else {"completed": False, "passed": None}
    )
    if duplicate_audit.get("passed") is False:
        raise RadioML2018Error("Exact duplicate samples cross split boundaries")

    split_path = output / "RML2018.01A.split.h5"
    if split_path.exists() or split_path.is_symlink():
        raise RadioML2018Error("Refusing to overwrite the split artifact")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".RML2018.01A.split.", suffix=".h5.tmp", dir=output
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w", libver="earliest") as file:
            file.attrs["dataset_id"] = spec["dataset_id"]
            file.attrs["assignment_sha256"] = assignment_sha256
            for split in SPLITS:
                file.create_dataset(
                    split,
                    data=rows[split],
                    dtype="<i8",
                    chunks=(min(65_536, len(rows[split])),),
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                    fletcher32=True,
                )
            file.flush()
        _fsync_file(temporary)
        os.link(temporary, split_path)
    except Exception:
        split_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "purpose": "radioml_2018_01a_frozen_split",
        "counts": {split: len(rows[split]) for split in SPLITS},
        "assignment": {
            "seed": contract["assignment"]["seed"],
            "algorithm": contract["assignment"]["algorithm"],
            "sha256": assignment_sha256,
        },
        "artifact": {
            "filename": split_path.name,
            "size_bytes": split_path.stat().st_size,
            "sha256": _sha256_file(split_path),
        },
        "bindings": {
            "source_manifest_sha256": _sha256_file(source_manifest_path),
            "dataset_spec_sha256": _sha256_file(dataset_spec_path),
            "split_contract_sha256": _sha256_file(split_contract_path),
            "source_hdf5_sha256": spec["source"]["hdf5_sha256"],
            "project_commit": project_commit,
        },
        "leakage_audit": {
            "exact_duplicates": duplicate_audit,
            "near_duplicates": {
                "global_audit_completed": False,
                "limitation": "transformed near-duplicate leakage is not globally excluded",
            },
            "adjacent_windows": {
                "audit_completed": False,
                "limitation": "capture/session/window provenance is unavailable",
            },
        },
        "test_isolation": {
            "training_allowed_splits": ["train"],
            "tuning_allowed_splits": ["train", "validation"],
            "test_access_requires": "experiment-freeze-manifest-v1",
            "test_adapter_available": False,
        },
        "test_accessed": False,
    }
    try:
        _write_json_atomic(manifest, output / "RML2018.01A.split-manifest.json")
    except Exception:
        split_path.unlink(missing_ok=True)
        raise
    return manifest


class RadioML2018HDF5Dataset(Dataset[ModulationSample]):
    """Read only frozen RadioML 2018 train/validation rows with worker-local handles."""

    def __init__(
        self,
        *,
        split: Literal["train", "validation"],
        hdf5_path: Path,
        source_manifest_path: Path,
        split_artifact_path: Path,
        split_manifest_path: Path,
        dataset_spec_path: Path,
    ) -> None:
        if split not in TRAINING_SPLITS:
            raise RadioML2018Error(
                "Only train and validation are available before an experiment freeze manifest"
            )
        spec = _load_yaml(dataset_spec_path, "dataset spec")
        source_manifest = _load_json(source_manifest_path, "source manifest")
        split_manifest = _load_json(split_manifest_path, "split manifest")
        _validate_dataset_spec(spec)
        _validate_source_manifest(source_manifest, spec, dataset_spec_path)
        _validate_split_manifest(
            split_manifest,
            spec=spec,
            source_manifest_path=source_manifest_path,
            dataset_spec_path=dataset_spec_path,
        )
        bindings = split_manifest.get("bindings", {})
        artifact = split_manifest.get("artifact", {})
        if (
            bindings.get("source_manifest_sha256") != _sha256_file(source_manifest_path)
            or bindings.get("dataset_spec_sha256") != _sha256_file(dataset_spec_path)
            or hdf5_path.name != spec["source"]["hdf5_filename"]
            or hdf5_path.stat().st_size != spec["source"]["hdf5_size_bytes"]
            or split_artifact_path.name != artifact.get("filename")
            or split_artifact_path.stat().st_size != artifact.get("size_bytes")
            or _sha256_file(split_artifact_path) != artifact.get("sha256")
        ):
            raise RadioML2018Error("Dataset artifacts differ from their manifest bindings")
        _verified_sha256(hdf5_path, spec["source"]["hdf5_sha256"])
        _validate_hdf5_schema(hdf5_path, spec)
        with h5py.File(split_artifact_path, "r", libver="earliest", swmr=False) as file:
            if set(file.keys()) != set(SPLITS):
                raise RadioML2018Error("Split artifact datasets are incomplete")
            if file.attrs.get("assignment_sha256") != split_manifest["assignment"]["sha256"]:
                raise RadioML2018Error("Split artifact assignment digest differs")
            all_rows = {name: np.asarray(file[name], dtype=np.int64) for name in SPLITS}
        for name, values in all_rows.items():
            if (
                len(values) != split_manifest["counts"][name]
                or bool(np.any(values < 0))
                or bool(np.any(values >= spec["expected"]["x_shape"][0]))
                or bool(np.any(values[1:] <= values[:-1]))
            ):
                raise RadioML2018Error("Split artifact rows are invalid")
        _split_codes(all_rows, int(spec["expected"]["x_shape"][0]))
        if _assignment_sha256(all_rows) != split_manifest["assignment"]["sha256"]:
            raise RadioML2018Error("Split artifact assignment digest differs")
        self.rows = all_rows[split]
        if (
            len(self.rows) != split_manifest["counts"][split]
            or bool(np.any(self.rows[1:] <= self.rows[:-1]))
        ):
            raise RadioML2018Error("Selected split rows are not strictly sorted and unique")
        self.split = split
        self.hdf5_path = hdf5_path.resolve(strict=True)
        self.modulations = tuple(spec["expected"]["modulations"])
        self.snr_values = tuple(int(value) for value in spec["expected"]["snr_db"])
        self.samples_per_stratum = int(spec["expected"]["samples_per_stratum"])
        self.assignment_sha256 = str(split_manifest["assignment"]["sha256"])
        self.split_manifest_sha256 = _sha256_file(split_manifest_path)
        self.preprocessing = "per_sample_max_abs"
        self._file: h5py.File | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _hdf5(self) -> h5py.File:
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            self.close()
            self._file = h5py.File(self.hdf5_path, "r", libver="earliest", swmr=False)
            self._pid = pid
        return self._file

    def _normalize_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        normalized = index + len(self.rows) if index < 0 else index
        if not 0 <= normalized < len(self.rows):
            raise IndexError("dataset index out of range")
        return normalized

    def __getitem__(self, index: int) -> ModulationSample:
        normalized = self._normalize_index(index)
        row = int(self.rows[normalized])
        source = np.asarray(self._hdf5()["X"][row], dtype=np.float32).T.copy()
        return self._make_sample(row, torch.from_numpy(source))

    def __getitems__(self, indices: list[int]) -> list[ModulationSample]:
        if not isinstance(indices, list):
            raise TypeError("dataset indices must be a list of integers")
        if not indices:
            return []
        normalized = [self._normalize_index(index) for index in indices]
        rows = self.rows[np.asarray(normalized, dtype=np.int64)]
        if len(np.unique(rows)) != len(rows):
            return [self[index] for index in normalized]
        order = np.argsort(rows)
        sorted_rows = rows[order]
        source = np.asarray(self._hdf5()["X"][sorted_rows], dtype=np.float32)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        batch = source[inverse].transpose(0, 2, 1).copy()
        return [
            self._make_sample(int(row), torch.from_numpy(iq))
            for row, iq in zip(rows, batch, strict=True)
        ]

    def _make_sample(self, row: int, iq: torch.Tensor) -> ModulationSample:
        try:
            normalized = preprocess_iq(iq, mode="per_sample_max_abs")
        except PreprocessingError as error:
            raise RadioML2018Error(str(error)) from error
        stratum, source_index = divmod(row, self.samples_per_stratum)
        class_index, snr_index = divmod(stratum, len(self.snr_values))
        return make_sample(
            iq=normalized,
            modulation=class_index,
            snr=float(self.snr_values[snr_index]),
            sample_id=sample_id(class_index, self.snr_values[snr_index], source_index),
        )

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self) -> dict[str, object]:
        self.close()
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __enter__(self) -> RadioML2018HDF5Dataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = [
    "RadioML2018Error",
    "RadioML2018HDF5Dataset",
    "audit_radioml_2018_source",
    "generate_radioml_2018_split",
    "sample_id",
]
