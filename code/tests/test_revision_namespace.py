from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from na_lmscnet.evaluation.revision_namespace import (
    HASH_FILENAME,
    INVENTORY_FILENAME,
    MANIFEST_FILENAME,
    README_FILENAME,
    RevisionNamespaceError,
    initialize_revision_namespace,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "repo"
    (project / ".agents").mkdir(parents=True)
    (project / "paper").mkdir()
    (project / "code/configs/revision").mkdir(parents=True)
    (project / ".agents/PLAN.md").write_text("plan\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("agents\n", encoding="utf-8")
    decision = project / "paper/reassessment.md"
    decision.write_text("decision\n", encoding="utf-8")

    external = tmp_path / "external"
    external.mkdir()
    marker = external / "test-consumed.json"
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "retry_allowed": False,
                "test_dataset_constructed": True,
            }
        ),
        encoding="utf-8",
    )
    independence = external / "independence.json"
    independence.write_text(
        json.dumps(
            {
                "conclusion": "ineligible",
                "test_sample_content_opened_by_this_audit": False,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "revision_id": "revision-v1",
        "phase": "R0",
        "authority": ".agents/PLAN.md",
        "historical_cutoff": {
            "event": "radioml_2016_10a_test_consumed",
            "date": date(2026, 8, 13),
            "marker_path": str(marker),
            "marker_sha256": _sha256(marker),
            "retry_allowed": False,
        },
        "decision_source": {
            "path": "paper/reassessment.md",
            "date": "2026-08-14",
            "sha256": _sha256(decision),
        },
        "post_test_hypotheses": [
            {"id": f"H{index}", "formed_on": date(2026, 8, 14)}
            for index in range(1, 6)
        ],
        "post_test_rules": {
            "classification": "post_test_hypothesis",
            "forbidden_splits": ["radioml_2016_10a_test", "radioml_2018_01a_test"],
            "no_relabeling_as_pre_test": True,
            "old_test_may_not_support_component_claims": True,
        },
        "confirmatory_test": {
            "status": "blocked",
            "construction_allowed": False,
            "selected_candidate": None,
            "candidates": {
                "radioml_2018_01a": {
                    "status": "ineligible",
                    "audit_report": str(independence),
                    "audit_report_sha256": _sha256(independence),
                },
                "alternative": {"status": "unselected"},
            },
        },
        "artifact_namespace": {
            "directory_name": "namespace",
            "allowed_top_level_directories": [
                "audits",
                "logs",
                "manifests",
                "reports",
                "smoke",
                "validation",
            ],
            "forbidden_top_level_directories": [
                "test",
                "test-only-results",
                "confirmatory-test",
            ],
            "overwrite": False,
            "formal_runs_authorized": False,
        },
    }
    config_path = project / "code/configs/revision/phase_r0.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "project": project,
        "config": config_path,
        "marker": marker,
        "independence": independence,
        "output": external / "namespace",
    }


def _initialize(paths: dict[str, Path]) -> dict[str, object]:
    return initialize_revision_namespace(
        project_root=paths["project"],
        output_dir=paths["output"],
        config_path=paths["config"],
        independence_report_path=paths["independence"],
        test_consumed_marker_path=paths["marker"],
        initialization_date="2026-08-14",
        project_commit="a" * 40,
        worktree_status=" M .agents/PLAN.md\n",
    )


def test_initializer_creates_only_allowed_namespace_and_fail_closed_manifest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = _initialize(paths)
    output = paths["output"]

    assert result["status"] == "initialized"
    assert result["worktree_clean"] is False
    assert result["guards"] == {
        "formal_runs_authorized": False,
        "radioml_2016_10a_test": "permanently_locked",
        "radioml_2018_01a_test": "ineligible",
        "confirmatory_candidate_selected": False,
        "confirmatory_test_construction_allowed": False,
        "test_dataset_constructed_by_initializer": False,
        "test_sample_content_read_by_initializer": False,
    }
    assert {path.name for path in output.iterdir()} == {
        "audits",
        "logs",
        "manifests",
        "reports",
        "smoke",
        "validation",
        MANIFEST_FILENAME,
        INVENTORY_FILENAME,
        README_FILENAME,
        HASH_FILENAME,
    }
    assert not any(path.name == "test" for path in output.iterdir())
    for line in (output / HASH_FILENAME).read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", maxsplit=1)
        assert _sha256(output / name) == expected


def test_initializer_rejects_unblocked_confirmatory_test(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    config["confirmatory_test"]["construction_allowed"] = True
    paths["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(RevisionNamespaceError, match="blocked and unselected"):
        _initialize(paths)


def test_initializer_rejects_tampered_historical_consumption_marker(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["marker"].write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    with pytest.raises(RevisionNamespaceError, match="consumption marker"):
        _initialize(paths)


def test_initializer_rejects_existing_or_repository_internal_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].mkdir()
    with pytest.raises(RevisionNamespaceError, match="must not already exist"):
        _initialize(paths)

    paths = _fixture(tmp_path / "second")
    paths["output"] = paths["project"] / "namespace"
    with pytest.raises(RevisionNamespaceError, match="outside"):
        _initialize(paths)


def test_initializer_rejects_directory_name_different_from_frozen_config(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"] = paths["output"].with_name("different-name")

    with pytest.raises(RevisionNamespaceError, match="differs from the frozen"):
        _initialize(paths)
