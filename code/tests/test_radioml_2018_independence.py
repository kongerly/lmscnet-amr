from __future__ import annotations

import json
from pathlib import Path

import pytest

from na_lmscnet.evaluation.radioml_2018_independence import (
    EVIDENCE_FILENAME,
    HASH_FILENAME,
    HUMAN_REPORT_FILENAME,
    REPORT_FILENAME,
    RadioML2018IndependenceAuditError,
    audit_radioml_2018_test_independence,
)


def _write_repository(root: Path) -> Path:
    project = root / "repo"
    source = project / "code/src/na_lmscnet/data"
    config = project / "code/configs/data"
    source.mkdir(parents=True)
    config.mkdir(parents=True)
    (source / "radioml_2018.py").write_text(
        "\n".join(
            [
                'assigned = {"test": ranked[boundaries[1] :]}',
                "_audit_exact_duplicates(hdf5_path, codes)",
                "all_rows = {name: np.asarray(file[name], dtype=np.int64) for name in SPLITS}",
            ]
        ),
        encoding="utf-8",
    )
    (config / "radioml_2018_01a_split.yml").write_text(
        "stratification:\n  expected_totals: {train: 7, validation: 1, test: 2}\n",
        encoding="utf-8",
    )
    return project


def _write_ineligible_artifacts(root: Path) -> Path:
    artifacts = root / "artifacts"
    split = artifacts / "radioml-2018.01a-split"
    split.mkdir(parents=True)
    (split / "RML2018.01A.split.h5").write_bytes(b"binary-not-opened")
    manifest = {
        "dataset_id": "radioml_2018_01a",
        "purpose": "radioml_2018_01a_frozen_split",
        "counts": {"train": 7, "validation": 1, "test": 2},
        "assignment": {"sha256": "a" * 64},
        "artifact": {"filename": "RML2018.01A.split.h5", "sha256": "b" * 64},
        "leakage_audit": {"exact_duplicates": {"samples_scanned": 10}},
        "test_accessed": False,
    }
    (split / "RML2018.01A.split-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (artifacts / "queue-protocol.json").write_text(
        json.dumps(
            {
                "dataset_id": "radioml_2018_01a",
                "purpose": "radioml_2018_01a_validation_replication",
                "run_count": 3,
                "split_artifact_sha256": "b" * 64,
                "test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return artifacts


def test_positive_evidence_makes_test_ineligible_and_writes_hashed_outputs(tmp_path: Path) -> None:
    project = _write_repository(tmp_path)
    artifacts = _write_ineligible_artifacts(tmp_path)
    output = tmp_path / "audit-output"

    result = audit_radioml_2018_test_independence(
        project_root=project,
        artifact_roots=[artifacts],
        output_dir=output,
        audit_date="2026-08-14",
    )

    assert result["conclusion"] == "ineligible"
    assert result["decisive_evidence_count"] == 3
    assert result["test_sample_content_opened_by_this_audit"] is False
    assert {REPORT_FILENAME, EVIDENCE_FILENAME, HUMAN_REPORT_FILENAME, HASH_FILENAME} == {
        path.name for path in output.iterdir()
    }
    evidence = [
        json.loads(line)
        for line in (output / EVIDENCE_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert {item["criterion"] for item in evidence if item["decisive"]} == {
        "constructed",
        "read",
        "statistics",
    }
    assert (output / HASH_FILENAME).read_text(encoding="ascii").count("\n") == 3


def test_missing_artifact_root_is_indeterminate_without_positive_evidence(tmp_path: Path) -> None:
    project = _write_repository(tmp_path)
    output = tmp_path / "audit-output"

    result = audit_radioml_2018_test_independence(
        project_root=project,
        artifact_roots=[tmp_path / "missing"],
        output_dir=output,
        audit_date="2026-08-14",
    )

    assert result["conclusion"] == "indeterminate"
    assert result["scan"]["errors"]


def test_clean_complete_scan_is_eligible(tmp_path: Path) -> None:
    project = _write_repository(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "radioml-2018.01a-audit.log").write_text(
        "No partition was generated.\n", encoding="utf-8"
    )

    result = audit_radioml_2018_test_independence(
        project_root=project,
        artifact_roots=[artifacts],
        output_dir=tmp_path / "audit-output",
        audit_date="2026-08-14",
    )

    assert result["conclusion"] == "eligible"
    assert result["decisive_evidence_count"] == 0


def test_output_inside_repository_or_existing_output_is_rejected(tmp_path: Path) -> None:
    project = _write_repository(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with pytest.raises(RadioML2018IndependenceAuditError, match="outside"):
        audit_radioml_2018_test_independence(
            project_root=project,
            artifact_roots=[artifacts],
            output_dir=project / "audit-output",
            audit_date="2026-08-14",
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RadioML2018IndependenceAuditError, match="must not already exist"):
        audit_radioml_2018_test_independence(
            project_root=project,
            artifact_roots=[artifacts],
            output_dir=existing,
            audit_date="2026-08-14",
        )
