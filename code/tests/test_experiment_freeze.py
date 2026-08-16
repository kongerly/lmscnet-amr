from __future__ import annotations

import json
from pathlib import Path

import pytest

from na_lmscnet.data import RadioML2016HDF5Dataset, RadioMLDatasetError
from na_lmscnet.evaluation.experiment_freeze import (
    ExperimentFreezeError,
    _validate_report,
    authorize_frozen_test_dataset,
    canonical_json_sha256,
    consume_test_authorization,
    sha256_file,
    update_consumption_marker,
    write_manifest_atomic,
)


def _manifest(tmp_path: Path) -> Path:
    marker = tmp_path / "test-consumed.json"
    output = tmp_path / "test-output"
    manifest = {
        "schema_version": 1,
        "purpose": "experiment_freeze_manifest_v1",
        "implementation_commit": "1" * 40,
        "dataset": {
            "assignment_sha256": "2" * 64,
            "preprocessing_mode": "per_sample_max_abs",
        },
        "test_authorization": {
            "consumption_marker_path": str(marker),
            "output_dir": str(output),
        },
    }
    path = tmp_path / "experiment-freeze-manifest.json"
    write_manifest_atomic(manifest, path)
    return path


def test_freeze_manifest_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    with pytest.raises(ExperimentFreezeError, match="overwrite"):
        write_manifest_atomic({}, path)


def test_test_authorization_can_be_consumed_only_once(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    marker = consume_test_authorization(manifest_path)

    assert marker["status"] == "consumed"
    assert marker["test_dataset_constructed"] is False
    assert marker["retry_allowed"] is False
    assert marker["manifest_sha256"] == sha256_file(manifest_path)
    with pytest.raises(ExperimentFreezeError, match="already been consumed"):
        consume_test_authorization(manifest_path)


def test_failed_test_marker_remains_non_retryable(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    consume_test_authorization(manifest_path)
    update_consumption_marker(
        manifest_path,
        status="failed",
        test_dataset_constructed=True,
        failure_type="RuntimeError",
    )

    marker_path = Path(
        json.loads(manifest_path.read_text(encoding="utf-8"))["test_authorization"][
            "consumption_marker_path"
        ]
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "failed"
    assert marker["retry_allowed"] is False
    assert marker["test_dataset_constructed"] is True
    with pytest.raises(ExperimentFreezeError, match="already been consumed"):
        consume_test_authorization(manifest_path)


def test_frozen_dataset_authorization_requires_consumed_marker(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    consume_test_authorization(manifest_path)
    authorization = authorize_frozen_test_dataset(manifest_path)

    assert authorization["manifest_sha256"] == sha256_file(manifest_path)
    assert authorization["assignment_sha256"] == "2" * 64
    assert authorization["preprocessing_mode"] == "per_sample_max_abs"


def test_regular_dataset_still_rejects_test_before_freeze() -> None:
    with pytest.raises(RadioMLDatasetError, match="Only train and validation"):
        RadioML2016HDF5Dataset(  # type: ignore[arg-type]
            split="test",
            hdf5_path=Path("missing.h5"),
            conversion_manifest_path=Path("missing.json"),
            split_manifest_path=Path("missing.json"),
            leakage_audit_path=Path("missing.json"),
            split_contract_path=Path("missing.yml"),
            dataset_spec_path=Path("missing.yml"),
            conversion_contract_path=Path("missing.yml"),
        )


def test_complete_2018_recovery_manifest_is_accepted_as_report_binding(tmp_path: Path) -> None:
    path = tmp_path / "download-manifest.json"
    path.write_text(
        json.dumps(
            {
                "purpose": "radioml_2018_validation_replay_recovery",
                "all_sha256_match": True,
                "complete_evidence_bundle": True,
                "files": {"summary.json": {"sha256": "3" * 64}},
            }
        ),
        encoding="utf-8",
    )

    binding = _validate_report(path, "2018 download manifest")

    assert binding["sha256"] == sha256_file(path)


def test_canonical_json_digest_is_independent_of_file_formatting(tmp_path: Path) -> None:
    value = {"b": [2, 1], "a": "test"}
    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"
    compact.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    pretty.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert sha256_file(compact) != sha256_file(pretty)
    assert canonical_json_sha256(value) == canonical_json_sha256(
        json.loads(pretty.read_text(encoding="utf-8"))
    )
