"""Validation-only seed-13 report for multi-scale and dynamic-fusion ablations."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from na_lmscnet.data.contracts import ModulationSample
from na_lmscnet.evaluation.efficiency import count_macs, count_parameters
from na_lmscnet.evaluation.na_lmscnet_report import ALL_SNRS, _line_plot
from na_lmscnet.evaluation.snr_auxiliary_ablation_report import (
    _assert_replay_matches_metrics,
    _load_json,
    _mapping,
    _replay_run,
    _sha256_file,
    _validate_run,
    _write_csv,
)
from na_lmscnet.training.engine import experiment_config_sha256

REFERENCE_MODEL = "na_lmscnet"
WO_MULTI_SCALE_MODEL = "na_lmscnet_wo_multi_scale"
FIXED_AVERAGE_MODEL = "na_lmscnet_fixed_average"
CLEAR_LOW_SNR_DROP = 0.010
CLEAR_OVERALL_DROP = 0.005
BASICALLY_UNCHANGED = 0.005


class CoreAblationReportError(ValueError):
    """Raised when core-ablation report inputs violate the frozen protocol."""


def validate_split_audit_pair(split_manifest: Path, leakage_audit: Path) -> None:
    """Reject mixed-generation split artifacts before opening the HDF5 dataset."""

    split_manifest = split_manifest.resolve(strict=True)
    leakage_audit = leakage_audit.resolve(strict=True)
    if split_manifest.parent != leakage_audit.parent:
        raise CoreAblationReportError(
            "split manifest and leakage audit must come from the same frozen artifact directory; "
            f"got {split_manifest.parent} and {leakage_audit.parent}"
        )


def variant_screening_decision(
    *,
    reference_accuracy: float,
    reference_macro_f1: float,
    reference_low_snr: float,
    ablation_accuracy: float,
    ablation_macro_f1: float,
    ablation_low_snr: float,
) -> dict[str, object]:
    """Apply the pre-registered seed-13 screen without treating it as formal evidence."""

    deltas = {
        "accuracy": reference_accuracy - ablation_accuracy,
        "macro_f1": reference_macro_f1 - ablation_macro_f1,
        "low_snr_accuracy": reference_low_snr - ablation_low_snr,
    }
    clear_drop = deltas["low_snr_accuracy"] >= CLEAR_LOW_SNR_DROP and (
        deltas["accuracy"] >= CLEAR_OVERALL_DROP
        or deltas["macro_f1"] >= CLEAR_OVERALL_DROP
    )
    unchanged = all(abs(value) <= BASICALLY_UNCHANGED for value in deltas.values())
    if clear_drop:
        action = "run_five_seed_formal_validation"
        reason = "clear_seed13_drop"
    elif unchanged:
        action = "do_not_claim_independent_gain_from_seed13"
        reason = "seed13_basically_unchanged"
    elif any(value < -BASICALLY_UNCHANGED for value in deltas.values()):
        action = "do_not_claim_independent_gain_from_seed13"
        reason = "ablation_not_worse"
    else:
        action = "run_five_seed_formal_validation"
        reason = "seed13_borderline"
    return {"action": action, "reason": reason, "reference_minus_ablation": deltas}


def core_screening_decision(
    wo_multi_scale: dict[str, object], fixed_average: dict[str, object]
) -> dict[str, object]:
    """Combine both screens into a provisional next step, never a final contribution claim."""

    decisions = {
        WO_MULTI_SCALE_MODEL: wo_multi_scale,
        FIXED_AVERAGE_MODEL: fixed_average,
    }
    formal_variants = [
        model_name
        for model_name, decision in decisions.items()
        if decision["action"] == "run_five_seed_formal_validation"
    ]
    if formal_variants == [WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL]:
        scope = "lightweight_multiscale_dynamic_fusion_pending_formal_validation"
    elif formal_variants == [WO_MULTI_SCALE_MODEL]:
        scope = "lightweight_multiscale_fusion_pending_formal_validation"
    elif formal_variants == [FIXED_AVERAGE_MODEL]:
        scope = "core_mainline_inconclusive_pending_formal_validation"
    else:
        scope = "lightweight_performance_complexity_tradeoff_only"
    return {
        "action": "run_listed_five_seed_formal_validation"
        if formal_variants
        else "narrow_claim_without_additional_core_ablation_runs",
        "formal_validation_variants": formal_variants,
        "provisional_scope": scope,
        "final_claim_authorized": False,
    }


def _measure_latency(model: nn.Module, device: torch.device, warmup: int, iterations: int) -> float:
    model.eval().to(device)
    inputs = torch.zeros((1, 2, 128), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000 / iterations


def _plot_accuracy(path: Path, rows: list[dict[str, object]]) -> None:
    _line_plot(
        path,
        title="Core ablations seed-13 validation accuracy",
        panels=[
            (
                "Frozen split, seed 13",
                [
                    (
                        "NA-LMSCNet",
                        "#146c94",
                        [
                            (float(row["snr_db"]), float(row["na_lmscnet_accuracy"]))
                            for row in rows
                        ],
                    ),
                    (
                        "w/o multi-scale",
                        "#c5542d",
                        [
                            (float(row["snr_db"]), float(row["wo_multi_scale_accuracy"]))
                            for row in rows
                        ],
                    ),
                    (
                        "fixed-average",
                        "#397a4f",
                        [
                            (float(row["snr_db"]), float(row["fixed_average_accuracy"]))
                            for row in rows
                        ],
                    ),
                ],
            )
        ],
        y_label="Accuracy",
    )


def _efficiency_row(
    *,
    model_name: str,
    label: str,
    replay: dict[str, object],
    checkpoint_path: Path,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    model = replay["model"]
    if not isinstance(model, nn.Module):
        raise CoreAblationReportError("Replay did not return a model")
    parameters = count_parameters(model)
    macs = count_macs(model, (1, 2, 128), torch.device("cpu"))
    gpu_latency = (
        _measure_latency(model, device, warmup, iterations)
        if device.type == "cuda"
        else float("nan")
    )
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        cpu_latency = _measure_latency(model, torch.device("cpu"), warmup, iterations)
    finally:
        torch.set_num_threads(old_threads)
    return {
        "model": model_name,
        "label": label,
        "parameter_count": parameters,
        "macs": macs,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "gpu_latency_ms": gpu_latency,
        "gpu_throughput_samples_per_s": 1000.0 / gpu_latency
        if math.isfinite(gpu_latency)
        else float("nan"),
        "cpu_latency_ms": cpu_latency,
        "gpu_warmup": warmup,
        "gpu_iterations": iterations,
        "cpu_threads": 1,
    }


def generate_core_ablation_report(
    *,
    reference_config_path: Path,
    reference_run_dir: Path,
    reference_training_commit: str,
    wo_multi_scale_config_path: Path,
    wo_multi_scale_run_dir: Path,
    wo_multi_scale_training_commit: str,
    fixed_average_config_path: Path,
    fixed_average_run_dir: Path,
    fixed_average_training_commit: str,
    report_dir: Path,
    hdf5_path: Path,
    split_manifest_path: Path,
    leakage_audit_path: Path,
    validation_dataset: Dataset[ModulationSample],
    project_root: Path,
    report_generation_commit: str,
    device: torch.device,
    warmup: int = 100,
    iterations: int = 1000,
) -> dict[str, object]:
    """Generate the complete module-7 first-priority seed-13 evidence package."""

    project_root = project_root.resolve(strict=True)
    report_dir = report_dir.resolve()
    if report_dir == project_root or project_root in report_dir.parents:
        raise CoreAblationReportError("Report output must be outside repository")
    if report_dir.exists():
        raise CoreAblationReportError(f"Refusing to overwrite report: {report_dir}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CoreAblationReportError("CUDA requested but unavailable")
    if warmup < 1 or iterations < 1:
        raise CoreAblationReportError("Latency warmup and iterations must be positive")
    split_sha256 = _sha256_file(split_manifest_path)
    assignment_sha256 = validation_dataset.assignment_sha256
    audit = _mapping(_load_json(leakage_audit_path, "leakage audit"), "leakage audit")
    if audit.get("split_manifest_sha256") != validation_dataset.split_manifest_sha256:
        raise CoreAblationReportError("Validation split differs from leakage audit")
    inputs = [
        (
            REFERENCE_MODEL,
            "NA-LMSCNet",
            reference_config_path,
            reference_run_dir,
            reference_training_commit,
        ),
        (
            WO_MULTI_SCALE_MODEL,
            "w/o multi-scale",
            wo_multi_scale_config_path,
            wo_multi_scale_run_dir,
            wo_multi_scale_training_commit,
        ),
        (
            FIXED_AVERAGE_MODEL,
            "fixed-average",
            fixed_average_config_path,
            fixed_average_run_dir,
            fixed_average_training_commit,
        ),
    ]
    runs: dict[str, dict[str, object]] = {}
    for model_name, label, config_path, run_dir, training_commit in inputs:
        config, recorded_metrics, checkpoint_path = _validate_run(
            config_path=config_path,
            run_dir=run_dir,
            expected_model=model_name,
            expected_commit=training_commit,
            split_sha256=split_sha256,
            assignment_sha256=assignment_sha256,
        )
        if float(config.model.get("snr_loss_weight", -1.0)) != 0.1:
            raise CoreAblationReportError(f"{model_name} must retain SNR loss weight 0.1")
        replay = _replay_run(
            config=config,
            checkpoint_path=checkpoint_path,
            dataset=validation_dataset,
            device=device,
        )
        _assert_replay_matches_metrics(replay, recorded_metrics)
        if replay["snr_hat_db"] is None:
            raise CoreAblationReportError(f"{model_name} unexpectedly removed the SNR head")
        runs[model_name] = {
            "label": label,
            "config_path": config_path,
            "training_commit": training_commit,
            "checkpoint_path": checkpoint_path,
            "replay": replay,
            "efficiency": _efficiency_row(
                model_name=model_name,
                label=label,
                replay=replay,
                checkpoint_path=checkpoint_path,
                device=device,
                warmup=warmup,
                iterations=iterations,
            ),
        }
    reference = runs[REFERENCE_MODEL]["replay"]
    decisions = {}
    for model_name in (WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
        ablation = runs[model_name]["replay"]
        decisions[model_name] = variant_screening_decision(
            reference_accuracy=reference["metrics"].accuracy,
            reference_macro_f1=reference["metrics"].macro_f1,
            reference_low_snr=reference["low_snr_accuracy"],
            ablation_accuracy=ablation["metrics"].accuracy,
            ablation_macro_f1=ablation["metrics"].macro_f1,
            ablation_low_snr=ablation["low_snr_accuracy"],
        )
    combined_decision = core_screening_decision(
        decisions[WO_MULTI_SCALE_MODEL], decisions[FIXED_AVERAGE_MODEL]
    )
    comparison_rows = [
        {
            "model": model_name,
            "label": runs[model_name]["label"],
            "accuracy": runs[model_name]["replay"]["metrics"].accuracy,
            "macro_f1": runs[model_name]["replay"]["metrics"].macro_f1,
            "low_snr_accuracy": runs[model_name]["replay"]["low_snr_accuracy"],
            "snr_mae_db": runs[model_name]["replay"]["metrics"].snr_mae_db,
        }
        for model_name in (REFERENCE_MODEL, WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL)
    ]
    efficiency_rows = [
        runs[model_name]["efficiency"]
        for model_name in (REFERENCE_MODEL, WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL)
    ]
    per_snr_rows = []
    for snr in ALL_SNRS:
        key = f"{snr:+d}"
        reference_accuracy = reference["metrics"].per_snr_accuracy[key]
        wo_accuracy = runs[WO_MULTI_SCALE_MODEL]["replay"]["metrics"].per_snr_accuracy[key]
        fixed_accuracy = runs[FIXED_AVERAGE_MODEL]["replay"]["metrics"].per_snr_accuracy[key]
        per_snr_rows.append(
            {
                "snr_db": snr,
                "na_lmscnet_accuracy": reference_accuracy,
                "wo_multi_scale_accuracy": wo_accuracy,
                "fixed_average_accuracy": fixed_accuracy,
                "reference_minus_wo_multi_scale": reference_accuracy - wo_accuracy,
                "reference_minus_fixed_average": reference_accuracy - fixed_accuracy,
            }
        )
    bindings = {
        "reference_training_commit": reference_training_commit,
        "wo_multi_scale_training_commit": wo_multi_scale_training_commit,
        "fixed_average_training_commit": fixed_average_training_commit,
        "report_generation_commit": report_generation_commit,
        "reference_config_sha256": experiment_config_sha256(reference_config_path),
        "wo_multi_scale_config_sha256": experiment_config_sha256(wo_multi_scale_config_path),
        "fixed_average_config_sha256": experiment_config_sha256(fixed_average_config_path),
        "reference_checkpoint_sha256": _sha256_file(runs[REFERENCE_MODEL]["checkpoint_path"]),
        "wo_multi_scale_checkpoint_sha256": _sha256_file(
            runs[WO_MULTI_SCALE_MODEL]["checkpoint_path"]
        ),
        "fixed_average_checkpoint_sha256": _sha256_file(
            runs[FIXED_AVERAGE_MODEL]["checkpoint_path"]
        ),
        "split_manifest_sha256": split_sha256,
        "assignment_sha256": assignment_sha256,
        "hdf5_file_sha256": _sha256_file(hdf5_path),
        "leakage_audit_sha256": _sha256_file(leakage_audit_path),
        "seed": 13,
        "validation_sample_count": len(validation_dataset),
    }
    summary = {
        "schema_version": 1,
        "purpose": "core_multiscale_dynamic_fusion_seed13_screen",
        "test_accessed": False,
        "test_dataset_constructed": False,
        "bindings": bindings,
        "screening_rule": {
            "clear_low_snr_drop": CLEAR_LOW_SNR_DROP,
            "clear_accuracy_or_macro_f1_drop": CLEAR_OVERALL_DROP,
            "basically_unchanged_absolute_delta": BASICALLY_UNCHANGED,
        },
        "comparison": comparison_rows,
        "efficiency": efficiency_rows,
        "variant_decisions": decisions,
        "combined_decision": combined_decision,
    }
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=report_dir.parent))
    try:
        (staging / "figures").mkdir()
        _write_csv(staging / "comparison.csv", list(comparison_rows[0]), comparison_rows)
        _write_csv(staging / "efficiency.csv", list(efficiency_rows[0]), efficiency_rows)
        _write_csv(staging / "per_snr_accuracy.csv", list(per_snr_rows[0]), per_snr_rows)
        _plot_accuracy(staging / "figures" / "per_snr_accuracy.png", per_snr_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        lines = [
            "# Module 7 Core Ablations: Seed-13 Validation Screen",
            "",
            "Only the frozen validation split was used; no test dataset was constructed or accessed. The single-seed results are screening evidence only and do not authorize a final contribution claim.",
            "",
            f"- next action: `{combined_decision['action']}`",
            f"- formal variants: `{combined_decision['formal_validation_variants']}`",
            f"- provisional scope: `{combined_decision['provisional_scope']}`",
        ]
        for model_name in (WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
            decision = decisions[model_name]
            lines.extend(
                [
                    "",
                    f"## {model_name}",
                    f"- reason: `{decision['reason']}`",
                    f"- Accuracy delta: `{decision['reference_minus_ablation']['accuracy']:.6f}`",
                    f"- Macro-F1 delta: `{decision['reference_minus_ablation']['macro_f1']:.6f}`",
                    f"- Low-SNR accuracy delta: `{decision['reference_minus_ablation']['low_snr_accuracy']:.6f}`",
                ]
            )
        (staging / "report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest = {**summary, "files": files}
        (staging / "report-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        shutil.move(str(staging), str(report_dir))
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CoreAblationReportError",
    "core_screening_decision",
    "generate_core_ablation_report",
    "validate_split_audit_pair",
    "variant_screening_decision",
]
