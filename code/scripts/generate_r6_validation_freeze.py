"""Freeze the complete Phase R6 validation evidence without authorizing test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = "b6c56ced7b6893a135554b4c8a5fb3c089f58744"
EXPECTED_DIRECTORIES = (
    "validation/r6-fixed-epoch-b6c56ce",
    "validation/r6-fixed-epoch-replay-b6c56ce",
    "validation/r6-fixed-epoch-intervention-replay-b6c56ce",
    "reports/r6-fixed-epoch-summary-b6c56ce",
    "reports/r6-fixed-epoch-contrasts-b6c56ce",
    "audits/r6-fixed-epoch-queue-b6c56ce",
    "audits/r6-intervention-validity-b6c56ce",
)
RUNTIME_FILES = (
    "code/src/na_lmscnet/training/engine.py",
    "code/src/na_lmscnet/data/hdf5_conversion.py",
    "code/scripts/run_multi_seed.py",
    "code/configs/experiments/revision_r6_s2_fixed_epoch_radioml_2016_10a.yml",
    "code/configs/experiments/revision_r6_s1_static_fixed_epoch_radioml_2016_10a.yml",
    "code/configs/experiments/revision_r6_s1_wide_static_fixed_epoch_radioml_2016_10a.yml",
    "code/configs/experiments/revision_r6_sknet_1d_fixed_epoch_radioml_2016_10a.yml",
    "code/configs/experiments/revision_r6_afnet_fixed_epoch_radioml_2016_10a.yml",
)


class FreezeError(ValueError):
    """Raised when the R6 validation evidence cannot be frozen safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"Could not read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeError(f"JSON artifact must contain an object: {path}")
    return value


def _artifact_rows(namespace: Path, relative_directories: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_directory in relative_directories:
        directory = (namespace / relative_directory).resolve(strict=True)
        if namespace != directory and namespace not in directory.parents:
            raise FreezeError(f"Artifact directory escapes namespace: {directory}")
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(namespace).as_posix()
            if any(part.lower() in {"test", "test-only-results", "confirmatory-test"} for part in path.parts):
                raise FreezeError(f"Forbidden test path in validation freeze: {path}")
            if path.suffix.lower() == ".json":
                value = _json(path)
                if "test_accessed" in value and value["test_accessed"] is not False:
                    raise FreezeError(f"JSON artifact reports test access: {path}")
            rows.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size})
    if len({row["path"] for row in rows}) != len(rows):
        raise FreezeError("Duplicate artifact path in freeze")
    return rows


def _git_blob(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _working_blob(path: str) -> str:
    result = subprocess.run(
        ["git", "hash-object", "--", path],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_bindings(commit: str) -> list[dict[str, Any]]:
    rows = []
    for relative in RUNTIME_FILES:
        expected = _git_blob(commit, relative)
        actual = _working_blob(relative)
        if expected != actual:
            raise FreezeError(f"Runtime source differs from training commit: {relative}")
        rows.append({"path": relative, "git_blob": expected, "working_blob": actual, "match": True})
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-commit", default=EXPECTED_COMMIT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    namespace = args.namespace.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite freeze directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise FreezeError("Freeze output must remain outside the repository")
    if output_dir.parent.name != "manifests" or output_dir.parent.parent != namespace:
        raise FreezeError("Freeze directory must be under the R6 namespace manifests directory")
    if args.training_commit != EXPECTED_COMMIT:
        raise FreezeError("Training commit differs from the authorized R6 commit")

    artifacts = _artifact_rows(namespace, EXPECTED_DIRECTORIES)
    source_bindings = _source_bindings(args.training_commit)
    manifest = {
        "schema_version": 1,
        "purpose": "phase_r6_validation_evidence_freeze",
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "namespace": str(namespace),
        "training_commit": args.training_commit,
        "selection_metric": "fixed_epoch",
        "checkpoint_epoch": 100,
        "models": [
            "lmscnet_s2",
            "lmscnet_s1_static",
            "lmscnet_s1_wide_static",
            "sknet_1d_adaptation",
            "afnet_adaptation",
        ],
        "seeds": [13, 37, 73, 101, 137],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "source_bindings": source_bindings,
        "test_accessed": False,
        "locked_test_accessed": False,
        "confirmatory_test_authorized": False,
        "confirmatory_test_construction_allowed": False,
        "does_not_replace_phase_r2_gatekeeping": True,
    }
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "r6-validation-freeze-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha = _sha256(manifest_path)
    (output_dir / "r6-validation-freeze-manifest.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifact_count": len(artifacts),
                "manifest_sha256": manifest_sha,
                "test_accessed": False,
                "confirmatory_test_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
