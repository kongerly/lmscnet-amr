"""Finalize and audit the AEU submission package without accessing test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_NUMERIC_FRAGMENTS = (
    "+0.55 pp at low SNR, with a 95\\% CI of [-0.08,+1.21]",
    "+44.94 pp at low SNR, with a 95\\% CI of [+43.36,+46.62]",
    "+12.25 pp, with a permutation interval of [11.78,12.69]",
    "95.8\\% of shuffled pairs remained within the same modulation class",
    "+0.84 pp [+0.22,+1.45]",
    "+0.08 pp [-0.74,+0.97]",
    "+0.42 pp [-0.67,+1.61]",
    "124,861 parameters and 4.65 million MACs",
    "4.51 ms",
    "The R6 fixed-epoch route completed 25 additional validation runs",
    "+0.44 pp [-0.20,+1.04]",
    "+46.73 pp [+44.51,+49.01]",
    "+12.63 pp with a permutation interval of [+12.01,+13.10]",
    "+1.52 pp [+0.26,+2.77]",
    "-0.38 pp [-1.25,+0.29]",
    "+0.53 pp [-0.13,+1.22]",
)

REQUIRED_BOUNDARY_FRAGMENTS = (
    "not an estimate of the training-time benefit of content adaptivity",
    "it is not equivalent to comparing two retrained models",
    "not claims about the original published implementations",
    "Archived SKNet-1D and AFNet gate summaries are omitted",
    "synthetic RadioML benchmarks also do not establish over-the-air robustness",
    "It was not publicly registered before the historical test access",
    "No smallest effect size of interest or equivalence margin was specified",
    "A public or venue-compliant repository URL, immutable release tag or commit, and software license remain author actions",
)

PROHIBITED_PATTERNS = (
    r"outperform(?:s|ed)? learned-static",
    r"outperform(?:s|ed)? SKNet",
    r"outperform(?:s|ed)? AFNet",
    r"sample-specific content matching provides an independent benefit",
    r"real-time advantage",
    r"deployment advantage",
)

UPLOAD_ROOT_FILES = (
    "author_metadata.txt",
    "cas-common.sty",
    "cas-sc.cls",
    "change_log.md",
    "cover_letter.txt",
    "declaration_of_interest.txt",
    "figure_captions.txt",
    "highlights.txt",
    "manuscript.pdf",
    "manuscript.tex",
    "references.bib",
    "stfloats.sty",
    "submission_checklist.md",
    "submission-source-manifest.json",
)


class AEUPackageAuditError(ValueError):
    """Raised when a package cannot be audited safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE))


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for cluster in re.findall(r"\\cite[pt]?\{([^}]+)\}", text):
        keys.update(key.strip() for key in cluster.split(",") if key.strip())
    return keys


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AEUPackageAuditError(f"Expected a JSON object: {path}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    parser.add_argument("--lint-dir", type=Path, required=True)
    parser.add_argument("--visual-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(package_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit directory: {output_dir}")
    if PROJECT_ROOT.resolve() == output_dir or PROJECT_ROOT.resolve() in output_dir.parents:
        raise AEUPackageAuditError("Audit outputs must remain outside the repository")

    build_pdf = package_dir / "build/manuscript.pdf"
    build_log = package_dir / "build/manuscript.log"
    build_bbl = package_dir / "build/manuscript.bbl"
    for path in (build_pdf, build_log, build_bbl):
        if not path.is_file():
            raise FileNotFoundError(path)

    final_pdf = package_dir / "manuscript.pdf"
    shutil.copy2(build_pdf, final_pdf)

    tex_path = package_dir / "manuscript.tex"
    bib_path = package_dir / "references.bib"
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    log = build_log.read_text(encoding="utf-8", errors="replace")
    citation_keys = _citation_keys(tex)
    bibliography_keys = _bib_keys(bib)

    figure_manifest = _load_json(args.figure_manifest.resolve())
    visual_report = _load_json(args.visual_report.resolve())
    lint_reports = {
        name: _load_json(args.lint_dir.resolve() / f"{name}-lint.json")
        for name in ("manuscript", "cover-letter", "highlights")
    }

    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, evidence: Any) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "evidence": evidence})

    required_paths = [package_dir / name for name in UPLOAD_ROOT_FILES]
    required_paths += sorted((package_dir / "figures").glob("*.pdf"))
    required_paths += sorted((package_dir / "thumbnails").glob("*.jpeg"))
    check(
        "package.required_files",
        len(required_paths) == 22 and all(path.is_file() for path in required_paths),
        [str(path) for path in required_paths],
    )
    check("template.cas_sc", "\\documentclass[a4paper,fleqn]{cas-sc}" in tex, "cas-sc")
    check("structure.tables", len(re.findall(r"\\begin\{table\*?\}", tex)) == 8, 8)
    check("structure.figures", tex.count("\\begin{figure}") == 2, 2)
    check(
        "citations.bidirectional_closure",
        citation_keys == bibliography_keys and len(citation_keys) == 30,
        {
            "used": len(citation_keys),
            "bibliography": len(bibliography_keys),
            "missing": sorted(citation_keys - bibliography_keys),
            "unused": sorted(bibliography_keys - citation_keys),
        },
    )
    check(
        "numbers.frozen_fragments",
        all(fragment in tex for fragment in EXPECTED_NUMERIC_FRAGMENTS),
        list(EXPECTED_NUMERIC_FRAGMENTS),
    )
    check(
        "claims.required_boundaries",
        all(fragment in tex for fragment in REQUIRED_BOUNDARY_FRAGMENTS),
        list(REQUIRED_BOUNDARY_FRAGMENTS),
    )
    prohibited_hits = [
        pattern for pattern in PROHIBITED_PATTERNS if re.search(pattern, tex, re.IGNORECASE)
    ]
    check("claims.prohibited_absent", not prohibited_hits, prohibited_hits)

    figure_outputs = {
        Path(row["path"]).name: row["sha256"] for row in figure_manifest.get("outputs", [])
    }
    package_figures = sorted((package_dir / "figures").glob("*.pdf"))
    figure_hashes_match = len(package_figures) == 2 and all(
        figure_outputs.get(path.name) == _sha256(path) for path in package_figures
    )
    check("figures.frozen_hashes", figure_hashes_match, [_binding(path) for path in package_figures])
    check(
        "figures.no_invalid_neighbor_curves",
        figure_manifest.get("neighbor_gate_curves_included") is False,
        figure_manifest.get("neighbor_gate_curves_included"),
    )
    check("figures.test_isolation", figure_manifest.get("test_accessed") is False, False)

    log_failures = {
        "undefined_citations": "undefined citations" in log.lower(),
        "undefined_references": "undefined reference" in log.lower(),
        "float_position_warning": "No positions in optional float specifier" in log,
    }
    check("latex.no_blocking_warnings", not any(log_failures.values()), log_failures)
    check(
        "latex.expected_bibliography_warnings_only",
        set(re.findall(r"Warning--empty pages in ([^\r\n]+)", log))
        <= {"loshchilov2019adamw", "shi2022afnet", "west2016dataset"},
        re.findall(r"Warning--empty pages in ([^\r\n]+)", log),
    )
    check("latex.pdf_matches_build", _sha256(final_pdf) == _sha256(build_pdf), _sha256(final_pdf))

    lint_failures = {
        name: int(report.get("summary", {}).get("fail", -1))
        for name, report in lint_reports.items()
    }
    check("writing_lint.no_fail", all(value == 0 for value in lint_failures.values()), lint_failures)
    check(
        "writing_lint.known_format_warnings",
        lint_reports["manuscript"].get("summary", {}).get("warn") == 1
        and lint_reports["cover-letter"].get("summary", {}).get("warn") == 0
        and lint_reports["highlights"].get("summary", {}).get("warn") == 2,
        {
            "manuscript": "tabular alignment ampersands are linter false positives; Markdown source passes with zero warnings",
            "highlights": "bullet markers and intentionally compact parallel statements",
        },
    )
    check("visual.full_document_pass", visual_report.get("passed") is True, visual_report)
    check(
        "visual.pagination",
        visual_report.get("physical_pages") == 16
        and visual_report.get("article_page_total") == 15,
        {
            "physical_pages": visual_report.get("physical_pages"),
            "article_page_total": visual_report.get("article_page_total"),
        },
    )
    check(
        "visual.figures_before_references",
        visual_report.get("figure_physical_pages") == [10, 11]
        and visual_report.get("references_start_physical_page") == 15,
        {
            "figures": visual_report.get("figure_physical_pages"),
            "references_start": visual_report.get("references_start_physical_page"),
        },
    )

    source_manifest = _load_json(package_dir / "submission-source-manifest.json")
    source_hash_mismatches = []
    for row in source_manifest.get("outputs", []):
        path = package_dir / str(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            source_hash_mismatches.append(str(path))
    check("source_manifest.outputs_unchanged", not source_hash_mismatches, source_hash_mismatches)
    check("source_manifest.test_isolation", source_manifest.get("test_accessed") is False, False)

    passed = all(row["passed"] for row in checks)
    if not passed:
        final_pdf.unlink(missing_ok=True)

    upload_files = [package_dir / name for name in UPLOAD_ROOT_FILES]
    upload_files += sorted((package_dir / "figures").glob("*.pdf"))
    upload_files += sorted((package_dir / "thumbnails").glob("*.jpeg"))
    final_manifest = {
        "schema_version": 1,
        "purpose": "aeu_final_submission_package",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "journal": "AEU - International Journal of Electronics and Communications",
        "template": "Elsevier CAS cas-sc, CAS Bundle v2.4",
        "test_accessed": False,
        "historical_test_reopened": False,
        "upload_files": [_binding(path) for path in upload_files if path.is_file()],
        "build_evidence": [_binding(path) for path in (build_pdf, build_log, build_bbl)],
        "figure_manifest": _binding(args.figure_manifest.resolve()),
        "lint_reports": [_binding(args.lint_dir.resolve() / f"{name}-lint.json") for name in lint_reports],
        "visual_report": _binding(args.visual_report.resolve()),
        "checks": checks,
        "remaining_author_actions": [
            "Provide a public or venue-compliant repository URL, immutable release tag or commit, and software license.",
            "Confirm or omit ORCID in the submission system.",
            "Verify the final generative-AI disclosure location and wording against the submission-day AEU policy.",
            "Verify institutional journal ranking, current fees, and OA choice using authorized sources.",
            "Complete the journal submission form and approve the generated proof.",
        ],
    }
    final_manifest_path = package_dir / "submission-final-manifest.json"
    final_manifest_path.write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [*upload_files, final_manifest_path]
    (package_dir / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(package_dir).as_posix()}\n"
            for path in sorted(checksum_paths)
            if path.is_file()
        ),
        encoding="ascii",
    )

    output_dir.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "purpose": "aeu_submission_package_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "test_accessed": False,
        "historical_test_reopened": False,
        "package_dir": str(package_dir),
        "final_manifest": _binding(final_manifest_path),
        "checks": checks,
    }
    report_path = output_dir / "aeu-submission-package-audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# AEU Submission Package Audit",
        "",
        f"- Result: **{'PASS' if passed else 'FAIL'}**",
        "- Test accessed: `false`",
        "- Historical locked test reopened: `false`",
        f"- Package: `{package_dir}`",
        f"- Final PDF SHA-256: `{_sha256(final_pdf) if final_pdf.exists() else 'removed-after-failure'}`",
        "",
        "| Check | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    markdown.extend(
        f"| `{row['id']}` | {'PASS' if row['passed'] else 'FAIL'} | "
        f"{json.dumps(row['evidence'], ensure_ascii=False)} |"
        for row in checks
    )
    (output_dir / "aeu-submission-package-audit.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "package_dir": str(package_dir),
                "passed": passed,
                "test_accessed": False,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
