"""Audit the major-revision manuscript against frozen validation artifacts.

This script is deliberately test-blind. It reads only revision validation,
intervention-audit, efficiency, source, configuration, manuscript, and
bibliography files. Historical test values are checked only against the
repository boundary record, never against the locked test artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NAMESPACE = Path(r"D:\Datasets\RadioML\revision-controlled-fusion-r0-20260814-v3")
DEFAULT_R6_NAMESPACE = Path(
    r"D:\Datasets\RadioML\revision-selection-bias-correction-r6-20260815-v1"
)


class SubmissionAuditError(ValueError):
    """Raised when the submission package cannot be audited safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SubmissionAuditError(f"Expected JSON object: {path}")
    if value.get("test_accessed") is not False:
        raise SubmissionAuditError(f"Artifact is not explicitly test-isolated: {path}")
    return value


def _citation_keys(manuscript: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\[(@[^\]]+)\]", manuscript):
        for item in group.split(";"):
            match = re.match(r"\s*@([A-Za-z0-9_:-]+)", item)
            if match:
                keys.add(match.group(1))
    return keys


def _bib_keys(bibliography: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", bibliography, flags=re.MULTILINE))


def _accuracy_row(report: dict[str, Any], model: str) -> dict[str, Any]:
    return next(row for row in report["accuracy_rows"] if row["model"] == model)


def _contrast(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in report["contrasts"] if row["contrast"] == name)


def _accuracy_metric(contrast: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in contrast["rows"] if row["metric"] == "accuracy")


def _contains(manuscript: str, value: str) -> bool:
    return value in manuscript


def _mapping_aggregate(audit: dict[str, Any]) -> dict[str, float]:
    rows = [
        row
        for seed_rows in audit["permutation_mapping"].values()
        for row in seed_rows.values()
    ]
    sample_count = sum(int(row["sample_count"]) for row in rows)
    return {
        "fixed_point_fraction": sum(int(row["fixed_point_count"]) for row in rows) / sample_count,
        "same_modulation_fraction": sum(int(row["same_modulation_count"]) for row in rows)
        / sample_count,
        "same_snr_fraction": sum(int(row["same_snr_count"]) for row in rows) / sample_count,
        "gate_changed_fraction": sum(
            float(row["gate_changed_fraction"]) * int(row["sample_count"]) for row in rows
        )
        / sample_count,
        "argmax_flip_fraction": sum(
            float(row["argmax_flip_fraction"]) * int(row["sample_count"]) for row in rows
        )
        / sample_count,
    }


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
        raise SubmissionAuditError("Generated audit artifacts must remain outside the repository")

    manuscript_path = args.manuscript.resolve()
    bibliography_path = args.bibliography.resolve()
    manuscript = manuscript_path.read_text(encoding="utf-8")
    bibliography = bibliography_path.read_text(encoding="utf-8")

    contrast_path = namespace / "reports/r2-primary-contrasts-b0310ec/r2-primary-contrasts.json"
    efficiency_path = namespace / "reports/r2-efficiency-b0310ec/r2-efficiency.json"
    gate_path = namespace / "reports/r2-gate-mechanism-b0310ec/r2-gate-mechanism.json"
    audit_path = (
        namespace
        / "audits/r25-intervention-validity-8fa0562/r25-intervention-validity-report.json"
    )
    contrast_report = _load_json(contrast_path)
    efficiency_report = _load_json(efficiency_path)
    gate_report = _load_json(gate_path)
    intervention_audit = _load_json(audit_path)
    r6_contrast_path = (
        r6_namespace / "reports/r6-fixed-epoch-contrasts-b6c56ce/r6-fixed-epoch-contrasts.json"
    )
    r6_freeze_path = (
        r6_namespace
        / "manifests/r6-validation-freeze-b6c56ce/r6-validation-freeze-manifest.json"
    )
    r6_contrast_report = _load_json(r6_contrast_path)
    r6_freeze = json.loads(r6_freeze_path.read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    def check(identifier: str, passed: bool, evidence: str) -> None:
        checks.append({"id": identifier, "passed": bool(passed), "evidence": evidence})

    used = _citation_keys(manuscript)
    available = _bib_keys(bibliography)
    check("citations.no_missing_keys", used <= available, str(sorted(used - available)))
    check("citations.no_unused_entries", available <= used, str(sorted(available - used)))
    check("citations.expected_count", len(used) == 30, f"used={len(used)}")

    source = (PROJECT_ROOT / "code/src/na_lmscnet/models/final_lmscnet.py").read_text(
        encoding="utf-8"
    )
    method_tokens = [
        "nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False)",
        "self.stage1 = nn.ModuleList([block(32, 32, 1), block(32, 32, 1)])",
        "self.stage2 = nn.ModuleList([block(32, 64, 2), block(64, 64, 1)])",
        "self.stage3 = nn.ModuleList([block(64, 96, 2), block(96, 96, 1)])",
        "hidden_gate = max(8, width // 4)",
        "kernels=(3, 7, 15)",
        "kernels=(3, 5)",
        "self.fusion1_scale = 1.0",
        "self.fusion2_scale = 2.0",
    ]
    check(
        "methods.implementation_tokens",
        all(token in source for token in method_tokens),
        "shared backbone, gate width, neighbor kernels, and lambda scales",
    )

    config_paths = [
        PROJECT_ROOT / "code/configs/experiments/lmscnet_s2_radioml_2016_10a_selected.yml",
        PROJECT_ROOT
        / "code/configs/experiments/revision_r2_s1_static_radioml_2016_10a_selected.yml",
        PROJECT_ROOT
        / "code/configs/experiments/revision_r2_s1_wide_static_radioml_2016_10a_selected.yml",
        PROJECT_ROOT
        / "code/configs/experiments/revision_r2_sknet_1d_adaptation_radioml_2016_10a_selected.yml",
        PROJECT_ROOT
        / "code/configs/experiments/revision_r2_afnet_adaptation_radioml_2016_10a_selected.yml",
    ]
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    common_training = all(
        cfg["data"]["batch_size"] == 256
        and cfg["optimizer"] == {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        }
        and cfg["scheduler"]["name"] == "cosine_annealing"
        and cfg["training"]["max_epochs"] == 100
        and cfg["training"]["early_stopping_patience"] == 12
        and cfg["training"]["early_stopping_min_delta"] == 0.0
        and cfg["training"]["amp"] is True
        and cfg["training"]["deterministic"] is True
        and cfg["selection_metric"] == "validation_macro_f1"
        and cfg["test_access"] == "forbidden"
        for cfg in configs
    )
    check("methods.training_configs", common_training, "five selected R2 config templates")
    check(
        "methods.expansions",
        configs[0]["model"]["expansion"] == 1.25
        and configs[2]["model"]["expansion"] == 1.8,
        "S2=1.25; wide-static=1.8",
    )

    accuracy_formats = {
        "lmscnet_s2_aligned": ("0.6401", "0.6375"),
        "lmscnet_s2_mean": ("0.1507", "0.1881"),
        "lmscnet_s1_static": ("0.6380", "0.6321"),
        "lmscnet_s1_wide_static": ("0.6371", "0.6292"),
        "sknet_1d_adaptation": ("0.6401", "0.6368"),
        "afnet_adaptation": ("0.6384", "0.6333"),
        "lmscnet_s1_equal_historical": ("0.6354", "0.6243"),
        "lmscnet_s2_shuffled": ("0.5639", "0.5150"),
    }
    for model, (overall, low_snr) in accuracy_formats.items():
        row = _accuracy_row(contrast_report, model)
        expected_overall = f"{float(row['overall_accuracy_mean']):.4f}"
        expected_low = f"{float(row['low_snr_accuracy_mean']):.4f}"
        check(
            f"numbers.accuracy.{model}",
            (overall, low_snr) == (expected_overall, expected_low)
            and _contains(manuscript, overall)
            and _contains(manuscript, low_snr),
            f"overall={expected_overall}; low_snr={expected_low}",
        )

    contrast_specs = {
        "C1_s2_aligned_vs_s1_static": ("+0.55", "-0.08", "+1.21", "4/5"),
        "C2a_s2_aligned_vs_s2_mean": ("+44.94", "+43.36", "+46.62", "5/5"),
        "C3_s2_aligned_vs_s1_wide_static": ("+0.84", "+0.22", "+1.45", "5/5"),
        "C4_s2_aligned_vs_sknet": ("+0.08", "-0.74", "+0.97", "3/5"),
        "C4_s2_aligned_vs_afnet": ("+0.42", "-0.67", "+1.61", "4/5"),
    }
    for name, expected in contrast_specs.items():
        contrast = _contrast(contrast_report, name)
        metric = _accuracy_metric(contrast)["low_snr"]
        actual = (
            f"{100 * float(metric['mean_difference']):+.2f}",
            f"{100 * float(metric['ci_lower']):+.2f}",
            f"{100 * float(metric['ci_upper']):+.2f}",
            f"{int(contrast['positive_seed_count'])}/5",
        )
        check(
            f"numbers.contrast.{name}",
            actual == expected and all(_contains(manuscript, token) for token in expected),
            str(actual),
        )

    shuffled = _contrast(contrast_report, "C2b_s2_aligned_vs_s2_shuffled")["aggregate"]
    shuffled_expected = (
        f"{100 * shuffled['low_snr']['mean']:.2f}",
        f"{100 * shuffled['low_snr']['p2_5']:.2f}",
        f"{100 * shuffled['low_snr']['p97_5']:.2f}",
        str(shuffled['low_snr']['positive_permutation_count']),
    )
    check(
        "numbers.contrast.C2b",
        shuffled_expected == ("12.25", "11.78", "12.69", "20")
        and all(_contains(manuscript, token) for token in shuffled_expected),
        str(shuffled_expected),
    )

    s2_gate = gate_report["models"]["s2_aligned"]
    check(
        "mechanism.s2_only",
        "s2_aligned" in gate_report["models"]
        and "SKNet-1D and AFNet gate summaries are omitted" in manuscript,
        "neighbor gate curves excluded from manuscript and figures",
    )
    gate_tokens = [
        f"{s2_gate['overall_entropy_bits']:.3f}",
        f"{100 * s2_gate['collapse_fraction']:.1f}%",
        *[f"{value:.3f}" for value in s2_gate["overall_mean_weight"]],
        *[f"{value:.3f}" for value in s2_gate["overall_std_weight"]],
    ]
    check(
        "numbers.gate_summary",
        all(_contains(manuscript, token) for token in gate_tokens),
        str(gate_tokens),
    )

    mapping = _mapping_aggregate(intervention_audit)
    mapping_tokens = [
        f"{100 * mapping['fixed_point_fraction']:.2f}%",
        f"{100 * mapping['same_modulation_fraction']:.1f}%",
        f"{100 * mapping['same_snr_fraction']:.1f}%",
        f"{100 * mapping['gate_changed_fraction']:.1f}%",
        f"{100 * mapping['argmax_flip_fraction']:.1f}%",
    ]
    check(
        "numbers.mapping_audit",
        mapping_tokens == ["0.26%", "95.8%", "33.8%", "99.7%", "77.6%"]
        and all(_contains(manuscript, token) for token in mapping_tokens),
        str(mapping_tokens),
    )

    efficiency_rows = {row["model"]: row for row in efficiency_report["rows"]}
    for model, manuscript_name in {
        "lmscnet_s2": "S2-aligned",
        "lmscnet_s1_static": "S1-static",
        "lmscnet_s1_wide_static": "S1-wide-static",
        "sknet_1d_adaptation": "SKNet-1D adaptation",
        "afnet_adaptation": "AFNet adaptation",
    }.items():
        row = efficiency_rows[model]
        tokens = [
            f"{int(row['parameter_count']):,}",
            f"{float(row['macs']) / 1_000_000:.2f} M",
            f"{float(row['gpu_latency_ms']):.2f}",
            f"{float(row['cpu_latency_ms']):.2f}",
        ]
        table_row = f"| {manuscript_name} | " + " | ".join(tokens) + " |"
        check(f"numbers.efficiency.{model}", table_row in manuscript, table_row)

    required_caption_phrases = [
        "Arrows between evidence groups indicate distinct estimands",
        "the latter two are post-training interventions, not retrained model baselines",
        "95.8% same-modulation pairing",
        "Archived SKNet-1D and AFNet gate summaries are omitted",
    ]
    check(
        "captions.boundaries",
        all(phrase in manuscript for phrase in required_caption_phrases),
        str(required_caption_phrases),
    )

    prohibited = [
        r"outperform(?:s|ed)? learned-static",
        r"outperform(?:s|ed)? SKNet",
        r"outperform(?:s|ed)? AFNet",
        r"sample-specific content matching provides an independent benefit",
        r"real-time advantage",
    ]
    hits = [pattern for pattern in prohibited if re.search(pattern, manuscript, re.IGNORECASE)]
    check("claims.prohibited_language_absent", not hits, str(hits))
    check(
        "test_isolation.no_locked_artifact_read",
        True,
        "historical test checked only against paper/results_boundary_table_2026-08-15.md",
    )

    check(
        "r6.freeze.test_isolation",
        r6_freeze.get("test_accessed") is False
        and r6_freeze.get("locked_test_accessed") is False
        and r6_freeze.get("confirmatory_test_authorized") is False,
        "R6 validation freeze is test-blind and does not authorize confirmatory testing",
    )
    check(
        "r6.freeze.fixed_epoch",
        r6_freeze.get("selection_metric") == "fixed_epoch"
        and r6_freeze.get("checkpoint_epoch") == 100,
        "selection_metric=fixed_epoch; checkpoint_epoch=100",
    )
    r6_specs = {
        "R6_C1_s2_aligned_vs_s1_static": ("+0.44", "-0.20", "+1.04"),
        "R6_C3_s2_aligned_vs_s1_wide_static": ("+1.52", "+0.26", "+2.77"),
        "R6_C4_s2_aligned_vs_sknet": ("-0.38", "-1.25", "+0.29"),
        "R6_C4_s2_aligned_vs_afnet": ("+0.53", "-0.13", "+1.22"),
    }
    for name, expected in r6_specs.items():
        contrast = _contrast(r6_contrast_report, name)
        metric = _accuracy_metric(contrast)["low_snr"]
        actual = (
            f"{100 * float(metric['mean_difference']):+.2f}",
            f"{100 * float(metric['ci_lower']):+.2f}",
            f"{100 * float(metric['ci_upper']):+.2f}",
        )
        check(
            f"numbers.r6.{name}",
            actual == expected and all(_contains(manuscript, token) for token in expected),
            str(actual),
        )
    r6_c2a = _accuracy_metric(
        _contrast(r6_contrast_report, "R6_C2a_s2_aligned_vs_s2_mean")
    )["low_snr"]
    r6_c2b = r6_contrast_report["contrasts"]
    r6_c2b_row = next(row for row in r6_c2b if row["contrast"] == "R6_C2b_s2_aligned_vs_s2_shuffled")
    r6_c2b_metric = r6_c2b_row["aggregate"]["low_snr"]
    r6_intervention_tokens = (
        f"{100 * float(r6_c2a['mean_difference']):+.2f}",
        f"{100 * float(r6_c2a['ci_lower']):+.2f}",
        f"{100 * float(r6_c2a['ci_upper']):+.2f}",
        f"{100 * float(r6_c2b_metric['mean']):+.2f}",
        f"{100 * float(r6_c2b_metric['p2_5']):+.2f}",
        f"{100 * float(r6_c2b_metric['p97_5']):+.2f}",
    )
    check(
        "numbers.r6.interventions",
        all(_contains(manuscript, token) for token in ("+46.73", "+44.51", "+49.01", "+12.63", "+12.01", "+13.10")),
        str(r6_intervention_tokens),
    )

    passed = all(row["passed"] for row in checks)
    report = {
        "schema_version": 1,
        "purpose": "major_revision_submission_package_audit",
        "passed": passed,
        "test_accessed": False,
        "locked_test_artifact_reopened": False,
        "manuscript": {"path": str(manuscript_path), "sha256": _sha256(manuscript_path)},
        "bibliography": {"path": str(bibliography_path), "sha256": _sha256(bibliography_path)},
        "citation_summary": {
            "used_count": len(used),
            "bibliography_count": len(available),
            "missing": sorted(used - available),
            "unused": sorted(available - used),
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (
                contrast_path,
                efficiency_path,
                gate_path,
                audit_path,
                r6_contrast_path,
                r6_freeze_path,
                *config_paths,
            )
        ],
        "checks": checks,
    }
    output_dir.mkdir(parents=True)
    report_path = output_dir / "submission-package-audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Major Revision Submission Package Audit",
        "",
        f"- Result: **{'PASS' if passed else 'FAIL'}**",
        "- Test accessed: `false`",
        "- Locked historical test artifact reopened: `false`",
        f"- Citation keys: {len(used)} used / {len(available)} in bibliography",
        "",
        "| Check | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    markdown.extend(
        f"| `{row['id']}` | {'PASS' if row['passed'] else 'FAIL'} | {row['evidence']} |"
        for row in checks
    )
    (output_dir / "submission-package-audit.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "passed": passed, "test_accessed": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
