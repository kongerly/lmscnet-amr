"""Create a no-new-test revision freeze for the current manuscript snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NAMESPACE = Path(r"D:\Datasets\RadioML\revision-controlled-fusion-r0-20260814-v3")
DEFAULT_R6_NAMESPACE = Path(
    r"D:\Datasets\RadioML\revision-selection-bias-correction-r6-20260815-v1"
)


class RevisionFreezeError(ValueError):
    """Raised when the revision freeze cannot be generated safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _verified_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    binding = _binding(path)
    if binding["sha256"] != expected_sha256:
        raise RevisionFreezeError(
            f"SHA-256 mismatch for {path}: {binding['sha256']} != {expected_sha256}"
        )
    return binding


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", type=Path, default=DEFAULT_NAMESPACE)
    parser.add_argument("--r6-namespace", type=Path, default=DEFAULT_R6_NAMESPACE)
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=PROJECT_ROOT / "paper/manuscript_major_revision_2026-08-15.md",
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=PROJECT_ROOT / "literature/bibliography/references.bib",
    )
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--aeu-package-dir", type=Path, required=True)
    parser.add_argument("--aeu-package-audit-dir", type=Path, required=True)
    parser.add_argument("--process-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    namespace = args.namespace.resolve()
    r6_namespace = args.r6_namespace.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if PROJECT_ROOT.resolve() == output_dir or PROJECT_ROOT.resolve() in output_dir.parents:
        raise RevisionFreezeError("Revision freeze must remain outside the repository")

    audit_report_path = args.audit_dir.resolve() / "submission-package-audit.json"
    figure_manifest_path = args.figure_dir.resolve() / "figure-assets-manifest.json"
    aeu_package_dir = args.aeu_package_dir.resolve()
    aeu_manifest_path = aeu_package_dir / "submission-final-manifest.json"
    aeu_audit_path = (
        args.aeu_package_audit_dir.resolve() / "aeu-submission-package-audit.json"
    )
    process_audit_path = args.process_audit.resolve()
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    aeu_manifest = json.loads(aeu_manifest_path.read_text(encoding="utf-8"))
    aeu_audit = json.loads(aeu_audit_path.read_text(encoding="utf-8"))
    process_audit = json.loads(process_audit_path.read_text(encoding="utf-8"))
    r6_validation_freeze_path = (
        r6_namespace
        / "manifests/r6-validation-freeze-b6c56ce/r6-validation-freeze-manifest.json"
    )
    r6_contrast_path = (
        r6_namespace / "reports/r6-fixed-epoch-contrasts-b6c56ce/r6-fixed-epoch-contrasts.json"
    )
    r6_summary_path = (
        r6_namespace
        / "reports/r6-fixed-epoch-summary-b6c56ce/r6-fixed-epoch-five-seed-summary.json"
    )
    r6_queue_audit_path = (
        r6_namespace
        / "audits/r6-fixed-epoch-queue-b6c56ce/r6-fixed-epoch-queue-audit.json"
    )
    r6_intervention_audit_path = (
        r6_namespace
        / "audits/r6-intervention-validity-b6c56ce/r25-intervention-validity-report.json"
    )
    r6_validation_freeze = json.loads(r6_validation_freeze_path.read_text(encoding="utf-8"))
    if not audit_report.get("passed") or audit_report.get("test_accessed") is not False:
        raise RevisionFreezeError("Submission audit must pass with test_accessed=false")
    if figure_manifest.get("test_accessed") is not False:
        raise RevisionFreezeError("Figure manifest is not test-isolated")
    if not aeu_manifest.get("passed") or aeu_manifest.get("test_accessed") is not False:
        raise RevisionFreezeError("AEU package manifest must pass with test_accessed=false")
    if not aeu_audit.get("passed") or aeu_audit.get("test_accessed") is not False:
        raise RevisionFreezeError("AEU package audit must pass with test_accessed=false")
    if not process_audit.get("passed") or process_audit.get("test_accessed") is not False:
        raise RevisionFreezeError("Process audit must pass with test_accessed=false")
    if (
        r6_validation_freeze.get("test_accessed") is not False
        or r6_validation_freeze.get("locked_test_accessed") is not False
        or r6_validation_freeze.get("confirmatory_test_authorized") is not False
    ):
        raise RevisionFreezeError("R6 validation freeze must remain test-blind and unauthorized")

    validation_summary_path = namespace / "validation/r2-multiseed-b0310ec/multi-seed-summary.json"
    validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    if validation_summary.get("test_accessed") is not False:
        raise RevisionFreezeError("Validation summary is not test-isolated")
    validation_dir = validation_summary_path.parent
    checkpoint_bindings = []
    for row in validation_summary["runs"]:
        if row.get("test_accessed") is not False:
            raise RevisionFreezeError(f"Run is not test-isolated: {row['run_id']}")
        checkpoint = _verified_binding(
            validation_dir / row["run_id"] / "best.pt", row["checkpoint_sha256"]
        )
        config = _verified_binding(
            validation_dir / "configs" / row["config_filename"], row["config_sha256"]
        )
        checkpoint_bindings.append(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "checkpoint": checkpoint,
                "config": config,
                "test_accessed": False,
            }
        )

    figure_outputs = [
        _verified_binding(Path(row["path"]), row["sha256"])
        for row in figure_manifest["outputs"]
    ]
    aeu_upload_files = [
        _verified_binding(Path(row["path"]), row["sha256"])
        for row in aeu_manifest["upload_files"]
    ]

    report_paths = [
        namespace / "reports/r2-primary-contrasts-b0310ec/r2-primary-contrasts.json",
        namespace / "reports/r2-five-seed-summary-b0310ec/r2-five-seed-summary.json",
        namespace / "reports/r2-efficiency-b0310ec/r2-efficiency.json",
        namespace / "reports/r2-gate-mechanism-b0310ec/r2-gate-mechanism.json",
        namespace / "audits/r2-queue-audit-b0310ec/r2-queue-audit.json",
        namespace
        / "audits/r25-intervention-validity-8fa0562/r25-intervention-validity-report.json",
        r6_validation_freeze_path,
        r6_contrast_path,
        r6_summary_path,
        r6_queue_audit_path,
        r6_intervention_audit_path,
    ]
    for path in report_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("test_accessed") is not False:
            raise RevisionFreezeError(f"Freeze input is not test-isolated: {path}")

    repository_files = [
        args.manuscript.resolve(),
        args.bibliography.resolve(),
        PROJECT_ROOT / ".agents/PLAN.md",
        PROJECT_ROOT / ".agents/HANDOFF.md",
        PROJECT_ROOT / "paper/results_boundary_table_2026-08-15.md",
        PROJECT_ROOT / "paper/citation_support_audit_2026-08-15.md",
        PROJECT_ROOT / "paper/simulated_review_and_revision_audit_2026-08-15.md",
        PROJECT_ROOT / "paper/aeu_submission_engineering_2026-08-15.md",
        PROJECT_ROOT / "code/src/na_lmscnet/models/final_lmscnet.py",
        PROJECT_ROOT / "code/scripts/run_r2_primary_contrasts.py",
        PROJECT_ROOT / "code/scripts/audit_r2_intervention_validity.py",
        PROJECT_ROOT / "code/scripts/generate_revision_figures.py",
        PROJECT_ROOT / "code/scripts/build_aeu_submission.py",
        PROJECT_ROOT / "code/scripts/audit_aeu_submission_package.py",
        PROJECT_ROOT / "code/scripts/audit_submission_package.py",
        PROJECT_ROOT / "code/scripts/audit_revision_processes.py",
        PROJECT_ROOT / "code/scripts/generate_no_new_test_revision_freeze.py",
        PROJECT_ROOT / "code/scripts/prune_bibliography_to_manuscript.py",
        PROJECT_ROOT / "code/scripts/summarize_r6_fixed_epoch_validation.py",
        PROJECT_ROOT / "code/scripts/run_r6_fixed_epoch_contrasts.py",
        PROJECT_ROOT / "code/scripts/generate_r6_validation_freeze.py",
    ]
    config_dir = PROJECT_ROOT / "code/configs/experiments"
    config_names = [
        "lmscnet_s2_radioml_2016_10a_selected.yml",
        "revision_r2_s1_static_radioml_2016_10a_selected.yml",
        "revision_r2_s1_wide_static_radioml_2016_10a_selected.yml",
        "revision_r2_sknet_1d_adaptation_radioml_2016_10a_selected.yml",
        "revision_r2_afnet_adaptation_radioml_2016_10a_selected.yml",
    ]
    repository_files.extend(config_dir / name for name in config_names)
    repository_files.extend(
        config_dir / name
        for name in (
            "revision_r6_s2_fixed_epoch_radioml_2016_10a.yml",
            "revision_r6_s1_static_fixed_epoch_radioml_2016_10a.yml",
            "revision_r6_s1_wide_static_fixed_epoch_radioml_2016_10a.yml",
            "revision_r6_sknet_1d_fixed_epoch_radioml_2016_10a.yml",
            "revision_r6_afnet_fixed_epoch_radioml_2016_10a.yml",
        )
    )

    head = str(_git("rev-parse", "HEAD")).strip()
    status = str(_git("status", "--porcelain=v1"))
    diff = _git("diff", "--binary", "HEAD", text=False)
    assert isinstance(diff, bytes)
    diff_sha256 = hashlib.sha256(diff).hexdigest()

    freeze = {
        "schema_version": 1,
        "purpose": "no_new_test_major_revision_freeze",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "frozen",
        "test_accessed": False,
        "test_authorization": {
            "authorized": False,
            "new_test_allowed": False,
            "historical_test_reopened": False,
            "forbidden_operations": [
                "construct test dataset",
                "read locked RadioML 2016.10A test",
                "replay or slice test predictions",
                "re-bootstrap test results",
                "evaluate a new model or metric on test",
            ],
        },
        "repository": {
            "head_commit": head,
            "dirty": bool(status.strip()),
            "status_porcelain": status.splitlines(),
            "working_tree_diff_sha256": diff_sha256,
            "snapshot_note": "Exact bound files are hashed below; no commit was created by the freeze tool.",
        },
        "manuscript_and_references": [_binding(path) for path in repository_files],
        "validation_training_commit": validation_summary["bindings"]["project_commit"],
        "r6_validation_training_commit": r6_validation_freeze["training_commit"],
        "assignment_sha256": validation_summary["bindings"]["assignment_sha256"],
        "split_manifest_sha256": validation_summary["bindings"]["split_manifest_sha256"],
        "validation_summary": _binding(validation_summary_path),
        "checkpoint_and_config_bindings": checkpoint_bindings,
        "statistical_and_audit_reports": [_binding(path) for path in report_paths],
        "figure_manifest": _binding(figure_manifest_path),
        "figure_assets": figure_outputs,
        "submission_audit": _binding(audit_report_path),
        "aeu_submission_package": {
            "package_dir": str(aeu_package_dir),
            "manifest": _binding(aeu_manifest_path),
            "upload_files": aeu_upload_files,
            "package_audit": _binding(aeu_audit_path),
        },
        "process_audit": _binding(process_audit_path),
        "historical_test_record": {
            "role": "earlier frozen whole-model comparison only",
            "recorded_report_sha256": "c4ba5a3e0fe5209c8df71595a5ff1d8e472727299b9329ca23f5ba3dbb0fddf7",
            "source_of_record": str(PROJECT_ROOT / ".agents/HANDOFF.md"),
            "artifact_read_during_freeze": False,
        },
        "claims_frozen": {
            "allowed": [
                "capacity alone is insufficient to explain the S2 result",
                "frozen S2 checkpoints depend on input-conditioned gate assignment",
                "C2b is bounded by 95.8% same-modulation pairing",
                "C1 and C4 remain unresolved under the validation protocol",
                "R6 fixed-epoch sensitivity reproduces the same evidence boundary",
            ],
            "forbidden": [
                "independent benefit from sample-specific content matching",
                "superiority over learned-static fusion",
                "superiority over SKNet-1D or AFNet",
                "confirmatory-test language for revision component hypotheses",
                "over-the-air or operational generalization",
            ],
        },
        "stop_rules": [
            "No new training seeds or model variants after this freeze.",
            "No new validation replay, bootstrap, metric, or gate analysis after this freeze.",
            "No access to the permanently locked RadioML 2016.10A test.",
            "Any material methods, metric, contrast, caption, or claim change invalidates this freeze.",
            "A confirmatory test requires a separate independence audit, manifest, and one-shot authorization.",
        ],
    }
    output_dir.mkdir(parents=True)
    freeze_path = output_dir / "no-new-test-revision-freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_sha = _sha256(freeze_path)
    (output_dir / "SHA256SUMS").write_text(
        f"{freeze_sha}  {freeze_path.name}\n", encoding="ascii"
    )
    summary = [
        "# No-New-Test Revision Freeze",
        "",
        "- Status: **FROZEN**",
        "- New test authorized: **NO**",
        "- Historical locked test reopened: **NO**",
        f"- Repository HEAD: `{head}`",
        f"- Dirty worktree snapshot: `{bool(status.strip())}`",
        f"- Freeze SHA-256: `{freeze_sha}`",
        "- Any substantive post-freeze change invalidates this manifest.",
    ]
    (output_dir / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "sha256": freeze_sha, "test_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
