"""Audit the single frozen test result without permitting any second test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation import (  # noqa: E402
    audit_freeze_manifest,
    sha256_file,
)
from na_lmscnet.evaluation.experiment_freeze import load_json  # noqa: E402


class FrozenTestAuditError(ValueError):
    """Raised when the frozen test result is incomplete or inconsistent."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    return parser.parse_args(argv)


def audit_results(manifest_path: Path) -> dict[str, object]:
    manifest_path = manifest_path.resolve(strict=True)
    audit_freeze_manifest(manifest_path, project_root=PROJECT_ROOT, require_unconsumed=False)
    manifest = load_json(manifest_path, "experiment freeze manifest")
    marker_path = Path(manifest["test_authorization"]["consumption_marker_path"])
    output_dir = Path(manifest["test_authorization"]["output_dir"])
    marker = load_json(marker_path, "test consumption marker")
    report_path = output_dir / "test-report.json"
    result_manifest_path = output_dir / "result-manifest.json"
    if marker.get("status") != "complete" or marker.get("retry_allowed") is not False:
        raise FrozenTestAuditError("Test consumption marker is not complete and non-retryable")
    if (
        marker.get("manifest_sha256") != sha256_file(manifest_path)
        or marker.get("result_report_sha256") != sha256_file(report_path)
        or marker.get("result_manifest_sha256") != sha256_file(result_manifest_path)
    ):
        raise FrozenTestAuditError("Test marker hashes differ from current artifacts")
    result_manifest = load_json(result_manifest_path, "test result manifest")
    for item in result_manifest.get("files", []):
        path = output_dir / str(item["path"])
        if sha256_file(path) != item.get("sha256") or path.stat().st_size != item.get("size_bytes"):
            raise FrozenTestAuditError(f"Test result artifact changed: {item.get('path')}")
    report = load_json(report_path, "test report")
    if (
        report.get("test_accessed") is not True
        or report.get("test_access_count") != 1
        or report.get("freeze_manifest_sha256") != sha256_file(manifest_path)
        or report.get("implementation_commit") != manifest.get("implementation_commit")
        or report.get("protocol") != manifest.get("test_protocol")
        or len(report.get("runs", [])) != 10
    ):
        raise FrozenTestAuditError("Test report identity or protocol differs from freeze")
    identities = {(row.get("model"), row.get("seed")) for row in report["runs"]}
    expected = {
        (model, seed)
        for model in manifest["test_protocol"]["models"]
        for seed in manifest["test_protocol"]["seeds"]
    }
    if identities != expected:
        raise FrozenTestAuditError("Test report run matrix is incomplete")
    baseline = None
    for item in report.get("prediction_manifest", []):
        path = output_dir / "predictions" / str(item["filename"])
        if sha256_file(path) != item.get("sha256"):
            raise FrozenTestAuditError("Prediction artifact hash differs")
        with np.load(path, allow_pickle=False) as data:
            alignment = tuple(
                np.asarray(data[field]) for field in ("sample_ids", "targets", "snr_db")
            )
        if baseline is None:
            baseline = alignment
        elif any(
            not np.array_equal(left, right) for left, right in zip(baseline, alignment, strict=True)
        ):
            raise FrozenTestAuditError("Test predictions are not sample-aligned")
    return {
        "schema_version": 1,
        "status": "pass",
        "test_access_count": 1,
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "test_report_sha256": sha256_file(report_path),
        "result_manifest_sha256": sha256_file(result_manifest_path),
        "run_count": 10,
        "prediction_count": len(report["prediction_manifest"]),
        "hashes_consistent": True,
        "sample_alignment": True,
        "retry_allowed": False,
        "post_test_design_changes_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    result = audit_results(parse_args(argv).manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
