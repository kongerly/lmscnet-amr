"""Initialize a fail-closed external artifact namespace for the Major Revision."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

MANIFEST_FILENAME = "revision-namespace-manifest.json"
INVENTORY_FILENAME = "post-test-hypothesis-inventory.json"
README_FILENAME = "README.md"
HASH_FILENAME = "SHA256SUMS"


class RevisionNamespaceError(ValueError):
    """Raised when a revision namespace request violates the frozen R0 policy."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RevisionNamespaceError(f"Expected YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RevisionNamespaceError(f"Expected JSON object: {path}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_sha256(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RevisionNamespaceError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("phase") != "R0"
        or config.get("authority") != ".agents/PLAN.md"
    ):
        raise RevisionNamespaceError("Revision config identity is invalid")
    cutoff = config.get("historical_cutoff", {})
    if (
        cutoff.get("event") != "radioml_2016_10a_test_consumed"
        or cutoff.get("retry_allowed") is not False
    ):
        raise RevisionNamespaceError("Historical test cutoff is invalid")
    cutoff_date = date.fromisoformat(str(cutoff.get("date")))
    _require_sha256(cutoff.get("marker_sha256"), "historical_cutoff.marker_sha256")

    hypotheses = config.get("post_test_hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 5:
        raise RevisionNamespaceError("Exactly five post-test hypothesis groups are required")
    identifiers = [item.get("id") for item in hypotheses if isinstance(item, dict)]
    if identifiers != ["H1", "H2", "H3", "H4", "H5"]:
        raise RevisionNamespaceError("Post-test hypotheses must be ordered H1 through H5")
    for item in hypotheses:
        formed_on = date.fromisoformat(str(item.get("formed_on")))
        if formed_on <= cutoff_date:
            raise RevisionNamespaceError("Post-test hypotheses must postdate test consumption")

    rules = config.get("post_test_rules", {})
    forbidden = set(rules.get("forbidden_splits", []))
    if (
        rules.get("classification") != "post_test_hypothesis"
        or rules.get("no_relabeling_as_pre_test") is not True
        or rules.get("old_test_may_not_support_component_claims") is not True
        or forbidden != {"radioml_2016_10a_test", "radioml_2018_01a_test"}
    ):
        raise RevisionNamespaceError("Post-test isolation rules are invalid")

    confirmatory = config.get("confirmatory_test", {})
    candidates = confirmatory.get("candidates", {})
    if (
        confirmatory.get("status") != "blocked"
        or confirmatory.get("construction_allowed") is not False
        or confirmatory.get("selected_candidate") is not None
        or candidates.get("radioml_2018_01a", {}).get("status") != "ineligible"
        or candidates.get("alternative", {}).get("status") != "unselected"
    ):
        raise RevisionNamespaceError("Confirmatory test must remain blocked and unselected")
    _require_sha256(
        candidates.get("radioml_2018_01a", {}).get("audit_report_sha256"),
        "confirmatory_test.candidates.radioml_2018_01a.audit_report_sha256",
    )

    namespace = config.get("artifact_namespace", {})
    allowed = namespace.get("allowed_top_level_directories")
    forbidden_directories = set(namespace.get("forbidden_top_level_directories", []))
    if (
        not isinstance(namespace.get("directory_name"), str)
        or allowed != ["audits", "logs", "manifests", "reports", "smoke", "validation"]
        or forbidden_directories != {"test", "test-only-results", "confirmatory-test"}
        or namespace.get("overwrite") is not False
        or namespace.get("formal_runs_authorized") is not False
    ):
        raise RevisionNamespaceError("Artifact namespace policy is invalid")


def initialize_revision_namespace(
    *,
    project_root: Path,
    output_dir: Path,
    config_path: Path,
    independence_report_path: Path,
    test_consumed_marker_path: Path,
    initialization_date: str,
    project_commit: str,
    worktree_status: str,
) -> dict[str, Any]:
    """Create an external R0 namespace without constructing or reading any test dataset."""

    project = project_root.resolve(strict=True)
    output = output_dir.resolve()
    if output.exists():
        raise RevisionNamespaceError("Revision namespace must not already exist")
    if output == project or _is_relative_to(output, project):
        raise RevisionNamespaceError("Revision namespace must remain outside the repository")
    if len(project_commit) != 40 or any(
        character not in "0123456789abcdef" for character in project_commit
    ):
        raise RevisionNamespaceError("project_commit must be a lowercase Git commit")
    date.fromisoformat(initialization_date)

    config_file = config_path.resolve(strict=True)
    independence_file = independence_report_path.resolve(strict=True)
    consumed_file = test_consumed_marker_path.resolve(strict=True)
    config = _load_yaml(config_file)
    _validate_config(config)
    if output.name != config["artifact_namespace"]["directory_name"]:
        raise RevisionNamespaceError("Output directory name differs from the frozen namespace config")
    independence = _load_json(independence_file)
    consumed = _load_json(consumed_file)

    configured_cutoff = config["historical_cutoff"]
    configured_candidate = config["confirmatory_test"]["candidates"]["radioml_2018_01a"]
    if (
        str(consumed_file) != str(Path(configured_cutoff["marker_path"]).resolve())
        or _sha256_file(consumed_file) != configured_cutoff["marker_sha256"]
        or consumed.get("status") != "complete"
        or consumed.get("retry_allowed") is not False
        or consumed.get("test_dataset_constructed") is not True
    ):
        raise RevisionNamespaceError("Historical 2016 test consumption marker is invalid")
    if (
        str(independence_file) != str(Path(configured_candidate["audit_report"]).resolve())
        or _sha256_file(independence_file) != configured_candidate["audit_report_sha256"]
        or independence.get("conclusion") != "ineligible"
        or independence.get("test_sample_content_opened_by_this_audit") is not False
    ):
        raise RevisionNamespaceError("RadioML 2018 independence audit binding is invalid")

    plan_path = project / ".agents" / "PLAN.md"
    agents_path = project / "AGENTS.md"
    decision_path = project / config["decision_source"]["path"]
    if _sha256_file(decision_path) != config["decision_source"]["sha256"]:
        raise RevisionNamespaceError("Major Revision decision source binding is invalid")

    allowed_directories = config["artifact_namespace"]["allowed_top_level_directories"]
    inventory_text = (
        json.dumps(_json_ready(config), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "purpose": "major_revision_phase_r0_namespace",
        "revision_id": config["revision_id"],
        "phase": "R0",
        "status": "initialized",
        "initialization_date": initialization_date,
        "project_commit": project_commit,
        "worktree_clean": not bool(worktree_status.strip()),
        "worktree_status_sha256": _sha256_bytes(worktree_status.encode("utf-8")),
        "directories": allowed_directories,
        "bindings": {
            "agents_sha256": _sha256_file(agents_path),
            "config_sha256": _sha256_file(config_file),
            "decision_source_sha256": _sha256_file(decision_path),
            "independence_report_sha256": _sha256_file(independence_file),
            "plan_sha256": _sha256_file(plan_path),
            "test_consumed_marker_sha256": _sha256_file(consumed_file),
        },
        "guards": {
            "formal_runs_authorized": False,
            "radioml_2016_10a_test": "permanently_locked",
            "radioml_2018_01a_test": "ineligible",
            "confirmatory_candidate_selected": False,
            "confirmatory_test_construction_allowed": False,
            "test_dataset_constructed_by_initializer": False,
            "test_sample_content_read_by_initializer": False,
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    readme_text = (
        "# Major Revision Phase R0 artifact namespace\n\n"
        "This directory establishes only the revision-artifact namespace and governance bindings; "
        "it does not authorize formal training or test access.\n\n"
        "- RadioML 2016.10A test: permanently locked.\n"
        "- RadioML 2018.01A test: `ineligible`.\n"
        "- New confirmatory candidate: not selected.\n"
        "- Formal experiments: awaiting a clean commit and a subsequent phase manifest.\n"
        "- This initialization did not construct or read any test dataset.\n"
    )

    temporary = output.with_name(f".{output.name}.initializing")
    if temporary.exists():
        raise RevisionNamespaceError("Temporary revision namespace already exists")
    temporary.mkdir(parents=True, exist_ok=False)
    for name in allowed_directories:
        (temporary / name).mkdir()
    inventory_path = temporary / INVENTORY_FILENAME
    manifest_path = temporary / MANIFEST_FILENAME
    readme_path = temporary / README_FILENAME
    inventory_path.write_text(inventory_text, encoding="utf-8")
    manifest_path.write_text(manifest_text, encoding="utf-8")
    readme_path.write_text(readme_text, encoding="utf-8")
    hashes = {
        path.name: _sha256_file(path) for path in (inventory_path, manifest_path, readme_path)
    }
    sums_path = temporary / HASH_FILENAME
    sums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="ascii",
    )
    temporary.rename(output)
    return {
        **manifest,
        "output_dir": str(output),
        "output_sha256": {**hashes, HASH_FILENAME: _sha256_file(output / HASH_FILENAME)},
    }


__all__ = [
    "HASH_FILENAME",
    "INVENTORY_FILENAME",
    "MANIFEST_FILENAME",
    "README_FILENAME",
    "RevisionNamespaceError",
    "initialize_revision_namespace",
]
