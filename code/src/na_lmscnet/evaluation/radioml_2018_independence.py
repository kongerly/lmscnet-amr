"""Fail-closed independence audit for the RadioML 2018.01A test partition."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
BINARY_SUFFIXES = {".h5", ".hdf5", ".npz", ".npy", ".pt", ".pth"}
SKIP_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
REPORT_FILENAME = "radioml-2018.01a-test-independence-report.json"
EVIDENCE_FILENAME = "radioml-2018.01a-test-independence-evidence.jsonl"
HUMAN_REPORT_FILENAME = "radioml-2018.01a-test-independence-report.md"
HASH_FILENAME = "SHA256SUMS"


class RadioML2018IndependenceAuditError(ValueError):
    """Raised when the audit request or evidence is invalid."""


@dataclass(frozen=True)
class ScannedText:
    path: Path
    scope: str
    text: str
    sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _iter_files(root: Path) -> Sequence[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(root):
        directories[:] = [name for name in directories if name not in SKIP_DIRECTORIES]
        base = Path(current)
        files.extend(base / name for name in names)
    return files


def _read_text(path: Path, *, max_text_bytes: int) -> tuple[str, str] | None:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > max_text_bytes:
        return None
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace"), _sha256_bytes(raw)


def _line_number(text: str, needle: str) -> int | None:
    offset = text.find(needle)
    return None if offset < 0 else text.count("\n", 0, offset) + 1


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _evidence(
    *,
    evidence_id: str,
    category: str,
    path: Path,
    scope: str,
    fact: str,
    criterion: str,
    decisive: bool,
    sha256: str | None = None,
    line: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "category": category,
        "criterion": criterion,
        "decisive": decisive,
        "details": details or {},
        "fact": fact,
        "line": line,
        "path": str(path.resolve()),
        "scope": scope,
        "sha256": sha256,
    }


def _inspect_repository_file(item: ScannedText) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    text = item.text
    if item.path.name == "radioml_2018_01a_split.yml":
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError:
            value = None
        totals = (
            value.get("stratification", {}).get("expected_totals", {})
            if isinstance(value, dict)
            else {}
        )
        if isinstance(totals, dict) and int(totals.get("test", 0)) > 0:
            evidence.append(
                _evidence(
                    evidence_id="repo-split-contract-test-count",
                    category="test_partition_declared",
                    path=item.path,
                    scope=item.scope,
                    fact="The frozen split contract declares a non-empty test partition.",
                    criterion="constructed",
                    decisive=False,
                    sha256=item.sha256,
                    line=_line_number(text, "expected_totals"),
                    details={"expected_test_count": int(totals["test"])},
                )
            )
    if item.path.name == "radioml_2018.py":
        patterns = (
            (
                '"test": ranked[boundaries[1] :]',
                "repo-split-builder",
                "The implementation constructs test-row assignments.",
                "constructed",
            ),
            (
                "_audit_exact_duplicates(hdf5_path, codes)",
                "repo-cross-split-statistics",
                "The split generator scans source samples using split-membership codes.",
                "statistics",
            ),
            (
                "all_rows = {name: np.asarray(file[name], dtype=np.int64) for name in SPLITS}",
                "repo-adapter-reads-all-split-rows",
                "Train/validation adapter initialization reads row indexes for every split, including test.",
                "read",
            ),
        )
        for needle, evidence_id, fact, criterion in patterns:
            if needle in text:
                evidence.append(
                    _evidence(
                        evidence_id=evidence_id,
                        category="repository_behavior",
                        path=item.path,
                        scope=item.scope,
                        fact=fact,
                        criterion=criterion,
                        decisive=False,
                        sha256=item.sha256,
                        line=_line_number(text, needle),
                    )
                )
    return evidence


def _inspect_external_json(item: ScannedText) -> list[dict[str, Any]]:
    value = _json_object(item.text)
    if value is None:
        return []
    evidence: list[dict[str, Any]] = []
    dataset_id = value.get("dataset_id")
    purpose = value.get("purpose")
    if purpose == "radioml_2018_01a_frozen_split" and dataset_id == "radioml_2018_01a":
        counts = value.get("counts", {})
        test_count = int(counts.get("test", 0)) if isinstance(counts, dict) else 0
        artifact = value.get("artifact", {})
        artifact_path = item.path.parent / str(artifact.get("filename", ""))
        artifact_exists = artifact_path.is_file()
        if test_count > 0:
            evidence.append(
                _evidence(
                    evidence_id=f"external-split-manifest-{item.sha256[:12]}",
                    category="test_partition_constructed",
                    path=item.path,
                    scope=item.scope,
                    fact="The external frozen split manifest records a constructed non-empty test partition.",
                    criterion="constructed",
                    decisive=True,
                    sha256=item.sha256,
                    details={
                        "artifact_exists": artifact_exists,
                        "artifact_path": str(artifact_path.resolve()),
                        "artifact_recorded_sha256": artifact.get("sha256"),
                        "assignment_sha256": value.get("assignment", {}).get("sha256"),
                        "test_count": test_count,
                    },
                )
            )
        duplicate = value.get("leakage_audit", {}).get("exact_duplicates", {})
        total = sum(int(count) for count in counts.values()) if isinstance(counts, dict) else 0
        if (
            test_count > 0
            and isinstance(duplicate, dict)
            and int(duplicate.get("samples_scanned", 0)) == total
            and total > 0
        ):
            evidence.append(
                _evidence(
                    evidence_id=f"external-cross-split-statistics-{item.sha256[:12]}",
                    category="test_partition_statistics",
                    path=item.path,
                    scope=item.scope,
                    fact="The exact-duplicate audit scanned all assigned samples, including test members.",
                    criterion="statistics",
                    decisive=True,
                    sha256=item.sha256,
                    details={"samples_scanned": total, "test_count": test_count},
                )
            )
    if purpose == "radioml_2018_01a_source_schema_audit" and dataset_id == "radioml_2018_01a":
        samples = value.get("audit", {}).get("samples")
        evidence.append(
            _evidence(
                evidence_id=f"external-source-global-read-{item.sha256[:12]}",
                category="source_global_statistics",
                path=item.path,
                scope=item.scope,
                fact="The source audit read and summarized all X/Y/Z arrays before confirmatory isolation.",
                criterion="read",
                decisive=True,
                sha256=item.sha256,
                details={"samples_scanned": samples},
            )
        )
    if dataset_id == "radioml_2018_01a" and purpose == "radioml_2018_01a_validation_replication":
        run_count = int(value.get("run_count", 0))
        if run_count > 0 and value.get("split_artifact_sha256"):
            evidence.append(
                _evidence(
                    evidence_id=f"external-validation-bound-split-{item.sha256[:12]}",
                    category="test_partition_index_read",
                    path=item.path,
                    scope=item.scope,
                    fact="Completed validation runs were bound to an artifact whose indexes for every split are read by the adapter.",
                    criterion="read",
                    decisive=True,
                    sha256=item.sha256,
                    details={
                        "run_count": run_count,
                        "split_artifact_sha256": value.get("split_artifact_sha256"),
                        "test_accessed_flag": value.get("test_accessed"),
                    },
                )
            )
    return evidence


def _relevant_text(item: ScannedText) -> bool:
    lowered = item.text.lower()
    path_text = str(item.path).lower()
    return any(
        marker in lowered or marker in path_text
        for marker in ("radioml_2018", "radioml-2018", "rml2018", "2018.01a")
    )


def _write_outputs(
    *, output_dir: Path, report: dict[str, Any], evidence: Sequence[dict[str, Any]]
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / REPORT_FILENAME
    evidence_path = output_dir / EVIDENCE_FILENAME
    human_path = output_dir / HUMAN_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    evidence_path.write_text(
        "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in evidence),
        encoding="utf-8",
    )
    decisive = [item for item in evidence if item["decisive"]]
    lines = [
        "# RadioML 2018.01A Test-Independence Audit",
        "",
        f"- Audit date: `{report['audit_date']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Decisive evidence items: `{len(decisive)}`",
        f"- Text files scanned: `{report['scan']['text_files_scanned']}`",
        f"- Binary files handled as metadata only: `{report['scan']['binary_files_metadata_only']}`",
        "",
        "## Conclusion",
        "",
        report["human_conclusion"],
        "",
        "## Decisive Evidence",
        "",
    ]
    lines.extend(
        f"- `{item['criterion']}`: {item['fact']} (`{item['path']}`)" for item in decisive
    )
    lines.extend(
        [
            "",
            "## Access Boundary",
            "",
            "This audit did not open the contents of any `.h5`, `.hdf5`, `.pt`, `.pth`, `.npy`, or `.npz` file.",
            "It scanned only source code, configuration, JSON/YAML, logs, and documentation, while recording binary-file metadata.",
            "Historical `test_accessed=false` records mean only that no test-performance evaluation was declared; they do not override evidence of construction, index reads, or statistics.",
        ]
    )
    human_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {
        path.name: _sha256_file(path) for path in (report_path, evidence_path, human_path)
    }
    (output_dir / HASH_FILENAME).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="ascii",
    )
    return {**hashes, HASH_FILENAME: _sha256_file(output_dir / HASH_FILENAME)}


def audit_radioml_2018_test_independence(
    *,
    project_root: Path,
    artifact_roots: Sequence[Path],
    output_dir: Path,
    audit_date: str,
    max_text_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Scan non-dataset evidence and publish an external independence audit."""

    project = project_root.resolve(strict=True)
    output = output_dir.resolve()
    if output.exists():
        raise RadioML2018IndependenceAuditError("Audit output directory must not already exist")
    if output == project or _is_relative_to(output, project):
        raise RadioML2018IndependenceAuditError("Audit output must remain outside the repository")
    if not artifact_roots:
        raise RadioML2018IndependenceAuditError("At least one external artifact root is required")

    roots = [("repository", project)]
    scan_errors: list[dict[str, str]] = []
    for root in artifact_roots:
        try:
            resolved = root.resolve(strict=True)
        except OSError as error:
            scan_errors.append({"path": str(root), "error": str(error)})
            continue
        roots.append(("external", resolved))

    scanned: list[ScannedText] = []
    binary_metadata: list[dict[str, Any]] = []
    relevant_files: list[dict[str, Any]] = []
    for scope, root in roots:
        try:
            paths = _iter_files(root)
        except OSError as error:
            scan_errors.append({"path": str(root), "error": str(error)})
            continue
        for path in paths:
            try:
                suffix = path.suffix.lower()
                path_text = str(path).lower()
                if suffix in BINARY_SUFFIXES and any(
                    marker in path_text for marker in ("2018.01a", "rml2018", "radioml-2018")
                ):
                    binary_metadata.append(
                        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "scope": scope}
                    )
                    continue
                result = _read_text(path, max_text_bytes=max_text_bytes)
            except OSError as error:
                scan_errors.append({"path": str(path), "error": str(error)})
                continue
            if result is None:
                continue
            text, digest = result
            item = ScannedText(path=path, scope=scope, text=text, sha256=digest)
            scanned.append(item)
            if _relevant_text(item):
                relevant_files.append(
                    {
                        "path": str(path.resolve()),
                        "scope": scope,
                        "sha256": digest,
                        "size_bytes": path.stat().st_size,
                    }
                )

    evidence: list[dict[str, Any]] = []
    for item in scanned:
        if item.scope == "repository":
            evidence.extend(_inspect_repository_file(item))
        else:
            evidence.extend(_inspect_external_json(item))
    evidence.sort(
        key=lambda item: (not item["decisive"], item["criterion"], item["path"], item["id"])
    )

    decisive = [item for item in evidence if item["decisive"]]
    if decisive:
        conclusion = "ineligible"
        human_conclusion = (
            "The RadioML 2018.01A test is ineligible as a new independent confirmatory test: "
            "the test partition was constructed, its members were included in all-sample statistics, "
            "and the validation execution path read indexes for every split."
        )
    elif scan_errors:
        conclusion = "indeterminate"
        human_conclusion = (
            "The audit contains uncovered or unreadable paths and therefore cannot establish "
            "that the test partition remained independent."
        )
    else:
        conclusion = "eligible"
        human_conclusion = (
            "Within the declared and completely scanned evidence scope, no construction, read, "
            "statistical use, or visualization of the test partition was found."
        )

    report = {
        "schema_version": 1,
        "purpose": "radioml_2018_01a_test_independence_audit",
        "audit_date": audit_date,
        "conclusion": conclusion,
        "human_conclusion": human_conclusion,
        "policy": {
            "eligible_requires_no_evidence_of": [
                "test_partition_constructed",
                "test_partition_read",
                "test_partition_statistics",
                "test_partition_visualization",
                "test_partition_model_selection",
            ],
            "precedence": ["ineligible", "indeterminate", "eligible"],
        },
        "scan": {
            "artifact_roots_requested": [str(path) for path in artifact_roots],
            "binary_files": binary_metadata,
            "binary_files_metadata_only": len(binary_metadata),
            "errors": scan_errors,
            "relevant_text_files": relevant_files,
            "text_files_scanned": len(scanned),
        },
        "evidence_count": len(evidence),
        "decisive_evidence_count": len(decisive),
        "test_sample_content_opened_by_this_audit": False,
    }
    hashes = _write_outputs(output_dir=output, report=report, evidence=evidence)
    return {**report, "output_dir": str(output), "output_sha256": hashes}


__all__ = [
    "HASH_FILENAME",
    "REPORT_FILENAME",
    "EVIDENCE_FILENAME",
    "HUMAN_REPORT_FILENAME",
    "RadioML2018IndependenceAuditError",
    "audit_radioml_2018_test_independence",
]
