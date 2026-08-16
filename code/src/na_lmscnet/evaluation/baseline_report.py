"""Build a validation-only, manifest-bound baseline report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from na_lmscnet.data.contracts import ModulationSample
from na_lmscnet.models import build_model
from na_lmscnet.training.engine import (
    ExperimentConfig,
    experiment_config_sha256,
    load_experiment_config,
)
from na_lmscnet.training.multiseed import (
    MultiSeedError,
    MultiSeedRunSpec,
    _best_epoch_record,
    _load_metrics,
    _sha256_file,
    _validate_completed_run,
    multi_seed_run_specs,
)

REPORT_SCHEMA_VERSION = 1
EXPECTED_MODELS = ("cnn2", "cldnn", "resnet1d")
EXPECTED_SEEDS = (13, 37, 73, 101, 137)
REPRESENTATIVE_SNRS = (-10, -6, 0)
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)


class BaselineReportError(ValueError):
    """Raised when baseline evidence is incomplete or violates a binding."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BaselineReportError(f"{field} must be a string-keyed mapping")
    return value


def _load_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BaselineReportError(f"{field} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BaselineReportError(f"Could not read {field}: {error}") from error
    return _mapping(value, field)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_file(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise BaselineReportError(f"{field} must be a regular file")


def _float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BaselineReportError(f"{field} must be finite")
    return float(value)


def _parse_snr(value: object, field: str) -> int:
    if not isinstance(value, str):
        raise BaselineReportError(f"{field} must be an SNR string")
    try:
        parsed = int(value)
    except ValueError as error:
        raise BaselineReportError(f"{field} is not a signed integer SNR") from error
    if value != f"{parsed:+d}":
        raise BaselineReportError(f"{field} uses a non-canonical SNR representation")
    return parsed


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    rows = []
    for label, values in zip(labels, matrix.tolist(), strict=True):
        rows.append({"true_label": label, **{predicted: value for predicted, value in zip(labels, values, strict=True)}})
    _write_csv(path, ["true_label", *labels], rows)


def _load_baseline_summary(
    output_root: Path,
    *,
    split_manifest_sha256: str,
    assignment_sha256: str,
    project_commit: str,
) -> tuple[dict[str, Any], list[tuple[MultiSeedRunSpec, Path, Path, dict[str, Any]]]]:
    summary = _load_json(output_root / "multi-seed-summary.json", "multi-seed summary")
    if summary.get("schema_version") != 1 or summary.get("purpose") != "baseline_multi_seed_training":
        raise BaselineReportError("Multi-seed summary schema or purpose is invalid")
    if summary.get("test_accessed") is not False or summary.get("run_count") != 15:
        raise BaselineReportError("Baseline summary must contain 15 runs with test_accessed=false")
    bindings = _mapping(summary.get("bindings"), "summary bindings")
    if bindings != {
        "split_manifest_sha256": split_manifest_sha256,
        "assignment_sha256": assignment_sha256,
        "project_commit": project_commit,
        "seeds": list(EXPECTED_SEEDS),
    }:
        raise BaselineReportError("Multi-seed summary bindings differ from requested artifacts")
    run_records = summary.get("runs")
    if not isinstance(run_records, list) or len(run_records) != 15:
        raise BaselineReportError("Multi-seed summary runs are incomplete")

    expected_specs = multi_seed_run_specs(EXPECTED_MODELS)
    records: list[tuple[MultiSeedRunSpec, Path, Path, dict[str, Any]]] = []
    for spec, record_value in zip(expected_specs, run_records, strict=True):
        record = _mapping(record_value, f"summary record {spec.run_id}")
        if record.get("run_id") != spec.run_id or record.get("model") != spec.model or record.get("seed") != spec.seed:
            raise BaselineReportError(f"Unexpected run ordering or identity for {spec.run_id}")
        config_path = output_root / "configs" / spec.config_filename
        run_dir = output_root / spec.output_directory
        _regular_file(config_path, f"config {spec.run_id}")
        _regular_file(run_dir / "metrics.json", f"metrics {spec.run_id}")
        _regular_file(run_dir / "best.pt", f"checkpoint {spec.run_id}")
        if (run_dir / "last.pt").exists():
            raise BaselineReportError(f"Completed run still contains last.pt: {spec.run_id}")
        try:
            metrics = _load_metrics(run_dir / "metrics.json")
            _validate_completed_run(
                metrics,
                spec=spec,
                config_path=config_path,
                split_manifest_sha256=split_manifest_sha256,
                assignment_sha256=assignment_sha256,
                project_commit=project_commit,
            )
        except MultiSeedError as error:
            raise BaselineReportError(f"Invalid completed run {spec.run_id}: {error}") from error
        expected_checkpoint = _sha256_file(run_dir / "best.pt")
        artifacts = _mapping(metrics.get("artifacts"), f"artifacts {spec.run_id}")
        if artifacts.get("checkpoint_sha256") != expected_checkpoint:
            raise BaselineReportError(f"Checkpoint digest mismatch for {spec.run_id}")
        if record.get("checkpoint_sha256") != expected_checkpoint or record.get("config_sha256") != experiment_config_sha256(config_path):
            raise BaselineReportError(f"Summary digest mismatch for {spec.run_id}")
        records.append((spec, config_path, run_dir, metrics))
    return summary, records


def _best_validation(metrics: dict[str, Any], run_id: str) -> dict[str, Any]:
    try:
        best = _best_epoch_record(metrics)
        validation = _mapping(best.get("validation"), f"validation metrics {run_id}")
        per_snr = _mapping(validation.get("per_snr_accuracy"), f"per-SNR metrics {run_id}")
    except MultiSeedError as error:
        raise BaselineReportError(f"Invalid best validation metrics for {run_id}: {error}") from error
    if set(per_snr) != {f"{snr:+d}" for snr in range(-20, 20, 2)}:
        raise BaselineReportError(f"Per-SNR metrics are incomplete for {run_id}")
    return {
        "accuracy": _float(validation.get("accuracy"), f"accuracy {run_id}"),
        "macro_f1": _float(validation.get("macro_f1"), f"macro F1 {run_id}"),
        "loss": _float(best.get("validation_loss"), f"validation loss {run_id}"),
        "per_snr_accuracy": {
            _parse_snr(key, f"SNR {run_id}"): _float(value, f"per-SNR accuracy {run_id}")
            for key, value in per_snr.items()
        },
    }


def _build_validation_loader(dataset: Dataset[ModulationSample], config: ExperimentConfig) -> DataLoader[ModulationSample]:
    return DataLoader(
        dataset,
        batch_size=int(config.data["batch_size"]),
        shuffle=False,
        num_workers=int(config.data["num_workers"]),
        pin_memory=bool(config.data["pin_memory"]),
        persistent_workers=int(config.data["num_workers"]) > 0,
    )


@torch.inference_mode()
def _replay_validation(
    model: nn.Module,
    checkpoint_path: Path,
    dataset: Dataset[ModulationSample],
    config: ExperimentConfig,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise BaselineReportError(f"Could not load checkpoint {checkpoint_path.name}: {error}") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise BaselineReportError(f"Checkpoint schema is invalid: {checkpoint_path.name}")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise BaselineReportError(f"Checkpoint lacks model state: {checkpoint_path.name}")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise BaselineReportError(f"Checkpoint model shape mismatch: {checkpoint_path.name}") from error
    model.eval().to(device)
    loader = _build_validation_loader(dataset, config)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    snr_confusions: dict[int, torch.Tensor] = {}
    correct = 0
    total = 0
    amp_enabled = bool(config.training["amp"]) and device.type == "cuda"
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        targets = batch["modulation"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(iq).argmax(dim=1)
        predictions_cpu = predictions.cpu()
        targets_cpu = targets.cpu()
        flat = targets_cpu * num_classes + predictions_cpu
        confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        for snr in sorted(int(item) for item in batch["snr"].unique().tolist()):
            mask = batch["snr"] == snr
            snr_flat = targets_cpu[mask] * num_classes + predictions_cpu[mask]
            matrix = snr_confusions.setdefault(snr, torch.zeros((num_classes, num_classes), dtype=torch.int64))
            matrix += torch.bincount(snr_flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        correct += int((predictions_cpu == targets_cpu).sum())
        total += len(targets_cpu)
    if total == 0:
        raise BaselineReportError("Validation replay consumed zero samples")
    true_positive = confusion.diag().to(torch.float64)
    false_positive = confusion.sum(dim=0).to(torch.float64) - true_positive
    false_negative = confusion.sum(dim=1).to(torch.float64) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = torch.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    return {
        "accuracy": correct / total,
        "macro_f1": float(f1.mean()),
        "sample_count": total,
        "confusion": confusion.numpy(),
        "snr_confusions": {snr: matrix.numpy() for snr, matrix in snr_confusions.items()},
    }


def _count_macs(model: nn.Module, input_shape: tuple[int, int, int], device: torch.device) -> int:
    counts = 0

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> None:
        nonlocal counts
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if not isinstance(output, torch.Tensor):
                raise BaselineReportError("Convolution output is not a tensor")
            kernel = int(np.prod(module.kernel_size))
            counts += int(output.numel()) * (module.in_channels // module.groups) * kernel
        elif isinstance(module, nn.Linear):
            if not isinstance(output, torch.Tensor):
                raise BaselineReportError("Linear output is not a tensor")
            counts += int(output.numel()) * module.in_features
        elif isinstance(module, nn.LSTM):
            value = inputs[0]
            if value.ndim != 3:
                raise BaselineReportError("LSTM input must be three-dimensional")
            sequence = value.shape[1] if module.batch_first else value.shape[0]
            batch = value.shape[0] if module.batch_first else value.shape[1]
            directions = 2 if module.bidirectional else 1
            for layer in range(module.num_layers):
                input_size = module.input_size if layer == 0 else module.hidden_size * directions
                counts += int(batch * sequence * directions * 4 * module.hidden_size * (input_size + module.hidden_size))

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LSTM))
    ]
    try:
        model.eval().to(device)
        with torch.inference_mode():
            model(torch.zeros(input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
    return counts


def _measure_latency(model: nn.Module, device: torch.device, warmup: int, iterations: int) -> float:
    if warmup < 0 or iterations < 1:
        raise BaselineReportError("Latency warmup must be non-negative and iterations positive")
    model.eval().to(device)
    input_tensor = torch.zeros((1, 2, 128), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def _plot_snr_curves(path: Path, aggregate: dict[str, list[dict[str, object]]]) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.0), dpi=160)
    for model in EXPECTED_MODELS:
        rows = aggregate[model]
        snr = [int(row["snr_db"]) for row in rows]
        mean = np.asarray([float(row["mean_accuracy"]) for row in rows])
        std = np.asarray([float(row["std_accuracy"]) for row in rows])
        axis.plot(snr, mean, marker="o", linewidth=1.8, label=model)
        axis.fill_between(snr, np.maximum(0.0, mean - std), np.minimum(1.0, mean + std), alpha=0.12)
    axis.set(title="RadioML 2016.10A validation accuracy by SNR", xlabel="SNR (dB)", ylabel="Accuracy", ylim=(0.0, 1.0))
    axis.set_xticks(list(range(-20, 20, 2)))
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_confusion(path: Path, matrix: np.ndarray, labels: list[str], title: str) -> None:
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(8.2, 7.0), dpi=160)
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set(xlabel="Predicted modulation", ylabel="True modulation", title=title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = normalized[row, column]
            if value >= 0.05:
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _aggregate_snr(records: list[dict[str, Any]]) -> dict[str, list[dict[str, object]]]:
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        best = record["best"]
        for snr, accuracy in best["per_snr_accuracy"].items():
            values[record["model"]][snr].append(accuracy)
    result: dict[str, list[dict[str, object]]] = {}
    for model in EXPECTED_MODELS:
        rows = []
        for snr in sorted(values[model]):
            data = np.asarray(values[model][snr], dtype=np.float64)
            rows.append({
                "model": model,
                "snr_db": snr,
                "mean_accuracy": float(data.mean()),
                "std_accuracy": float(data.std(ddof=1)),
                "min_accuracy": float(data.min()),
                "max_accuracy": float(data.max()),
                "run_count": len(data),
            })
        result[model] = rows
    return result


def _aggregate_summary(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for model in EXPECTED_MODELS:
        selected = [record for record in records if record["model"] == model]
        f1_data = np.asarray([float(item["best"]["macro_f1"]) for item in selected], dtype=np.float64)
        acc_data = np.asarray([float(item["best"]["accuracy"]) for item in selected], dtype=np.float64)
        f1_mean, f1_std = float(f1_data.mean()), float(f1_data.std(ddof=1))
        acc_mean, acc_std = float(acc_data.mean()), float(acc_data.std(ddof=1))
        epoch_data = np.asarray([item["best_epoch"] for item in selected], dtype=np.float64)
        low_snr = np.asarray([
            np.mean([item["best"]["per_snr_accuracy"][snr] for snr in LOW_SNR_VALUES])
            for item in selected
        ], dtype=np.float64)
        rows.append({
            "model": model,
            "parameter_count": selected[0]["parameter_count"],
            "validation_accuracy_mean": acc_mean,
            "validation_accuracy_std": acc_std,
            "validation_macro_f1_mean": f1_mean,
            "validation_macro_f1_std": f1_std,
            "low_snr_accuracy_mean": float(low_snr.mean()),
            "low_snr_accuracy_std": float(low_snr.std(ddof=1)),
            "best_epoch_mean": float(epoch_data.mean()),
            "best_epoch_std": float(epoch_data.std(ddof=1)),
            "run_count": len(selected),
        })
    return rows


def _aggregate_efficiency(efficiency: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for model in EXPECTED_MODELS:
        selected = [row for row in efficiency if row["model"] == model]
        row = {"model": model, "parameter_count": selected[0]["parameter_count"], "macs": selected[0]["macs"]}
        for field in ("checkpoint_size_bytes", "gpu_latency_ms", "gpu_throughput_samples_per_s", "cpu_latency_ms"):
            value = float(selected[0][field])
            row[f"{field}_mean"] = value
            row[f"{field}_std"] = 0.0
        result.append(row)
    return result


def _markdown_report(
    summary_rows: list[dict[str, object]],
    efficiency_rows: list[dict[str, object]],
    *,
    report_dir: Path,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# RadioML 2016.10A Baseline Report",
        "",
        "This report uses only the frozen train/validation protocol. Report generation did not read test data; the training-code commit, split manifest, assignment, seed, and checkpoint digest bindings were verified for all 15 runs.",
        "",
        f"- training commit: `{summary['bindings']['project_commit']}`",
        f"- split manifest SHA-256: `{summary['bindings']['split_manifest_sha256']}`",
        f"- assignment SHA-256: `{summary['bindings']['assignment_sha256']}`",
        f"- run count: `{summary['run_count']}`",
        "- low-SNR summary: mean per-SNR accuracy over `-10/-8/-6/-4/-2/0 dB`",
        "",
        "## Summary",
        "",
        "| Model | Parameters | Validation accuracy | Macro-F1 | Low-SNR accuracy | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['parameter_count']:,} | {row['validation_accuracy_mean']:.4f} +/- {row['validation_accuracy_std']:.4f} | {row['validation_macro_f1_mean']:.4f} +/- {row['validation_macro_f1_std']:.4f} | {row['low_snr_accuracy_mean']:.4f} +/- {row['low_snr_accuracy_std']:.4f} | {row['best_epoch_mean']:.1f} +/- {row['best_epoch_std']:.1f} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "- [Per-SNR accuracy curves](figures/per_snr_accuracy.png)",
        "- `confusion_matrices/` contains overall, `-10 dB`, `-6 dB`, and `0 dB` aggregate matrices for each model",
        "",
        "## Efficiency",
        "",
        "| Model | MACs | Checkpoint bytes | GPU latency ms | GPU samples/s | CPU single-thread latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in efficiency_rows:
        lines.append(
            f"| {row['model']} | {row['macs']:,} | {row['checkpoint_size_bytes_mean']:.0f} +/- {row['checkpoint_size_bytes_std']:.0f} | {row['gpu_latency_ms_mean']:.4f} +/- {row['gpu_latency_ms_std']:.4f} | {row['gpu_throughput_samples_per_s_mean']:.1f} +/- {row['gpu_throughput_samples_per_s_std']:.1f} | {row['cpu_latency_ms_mean']:.4f} +/- {row['cpu_latency_ms_std']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "These are validation results and efficiency measurements, not test results. Transformed near-duplicates and capture/session/window adjacency were not globally audited for RadioML 2016.10A, so this report is not evidence of capture-disjoint evaluation or real OTA generalization.",
        "",
        f"Report directory: `{report_dir}`",
        "",
    ])
    return "\n".join(lines)


def generate_baseline_report(
    *,
    output_root: Path,
    report_dir: Path,
    hdf5_path: Path,
    conversion_manifest_path: Path,
    split_manifest_path: Path,
    leakage_audit_path: Path,
    split_contract_path: Path,
    dataset_spec_path: Path,
    conversion_contract_path: Path,
    project_root: Path,
    training_project_commit: str,
    report_generation_project_commit: str,
    validation_dataset: Dataset[ModulationSample],
    device: torch.device,
    warmup: int = 100,
    iterations: int = 1000,
) -> dict[str, object]:
    """Validate completed baseline runs and publish a deterministic external report."""

    output_root = output_root.resolve(strict=True)
    report_dir = report_dir.resolve()
    project_root = project_root.resolve(strict=True)
    if project_root in report_dir.parents or report_dir == project_root:
        raise BaselineReportError("Report output must be outside the repository")
    split_manifest_sha256 = _sha256_file(split_manifest_path)
    assignment_sha256 = "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941"
    summary, run_inputs = _load_baseline_summary(
        output_root,
        split_manifest_sha256=split_manifest_sha256,
        assignment_sha256=assignment_sha256,
        project_commit=training_project_commit,
    )
    if validation_dataset.assignment_sha256 != assignment_sha256:
        raise BaselineReportError("Validation dataset assignment differs from baseline runs")
    leakage_audit = _load_json(leakage_audit_path, "leakage audit")
    canonical_split_manifest_sha256 = leakage_audit.get("split_manifest_sha256")
    if validation_dataset.split_manifest_sha256 != canonical_split_manifest_sha256:
        raise BaselineReportError("Validation dataset canonical split manifest digest differs from audit")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BaselineReportError("CUDA was requested but is unavailable")

    staging_parent = report_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=staging_parent))
    (staging / "figures").mkdir()
    (staging / "confusion_matrices").mkdir()
    try:
        replay_records: list[dict[str, Any]] = []
        efficiency_records: list[dict[str, object]] = []
        measured_models: set[str] = set()
        labels = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
        cpu_threads = torch.get_num_threads()
        try:
            for spec, config_path, run_dir, metrics in run_inputs:
                config = load_experiment_config(config_path)
                best = _best_validation(metrics, spec.run_id)
                model = build_model(spec.model, num_classes=int(config.model["num_classes"]), dropout=float(config.model["dropout"]))
                replay = _replay_validation(model, run_dir / "best.pt", validation_dataset, config, device, len(labels))
                if not math.isclose(replay["accuracy"], best["accuracy"], rel_tol=2e-5, abs_tol=2e-5) or not math.isclose(replay["macro_f1"], best["macro_f1"], rel_tol=2e-5, abs_tol=2e-5):
                    raise BaselineReportError(f"Validation replay differs from metrics for {spec.run_id}")
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                if spec.model not in measured_models:
                    macs = _count_macs(model, (1, 2, 128), device)
                    checkpoint_size = (run_dir / "best.pt").stat().st_size
                    gpu_latency = _measure_latency(model, device, warmup, iterations) if device.type == "cuda" else float("nan")
                    gpu_throughput = 1000.0 / gpu_latency if math.isfinite(gpu_latency) and gpu_latency > 0.0 else float("nan")
                    torch.set_num_threads(1)
                    cpu_latency = _measure_latency(model, torch.device("cpu"), warmup, iterations)
                    torch.set_num_threads(cpu_threads)
                    efficiency_records.append({
                        "run_id": spec.run_id,
                        "model": spec.model,
                        "seed": spec.seed,
                        "parameter_count": parameter_count,
                        "macs": macs,
                        "checkpoint_size_bytes": checkpoint_size,
                        "gpu_latency_ms": gpu_latency,
                        "gpu_throughput_samples_per_s": gpu_throughput,
                        "cpu_latency_ms": cpu_latency,
                        "cpu_threads": 1,
                        "warmup_iterations": warmup,
                        "measurement_iterations": iterations,
                    })
                    measured_models.add(spec.model)
                replay_records.append({
                    "run_id": spec.run_id,
                    "model": spec.model,
                    "seed": spec.seed,
                    "best_epoch": metrics["best_epoch"],
                    "metrics_accuracy": best["accuracy"],
                    "metrics_macro_f1": best["macro_f1"],
                    "replay_accuracy": replay["accuracy"],
                    "replay_macro_f1": replay["macro_f1"],
                    "sample_count": replay["sample_count"],
                })
                for suffix, matrix in [("overall", replay["confusion"]), *[(f"snr_{snr:+d}", replay["snr_confusions"][snr]) for snr in REPRESENTATIVE_SNRS]]:
                    _write_matrix_csv(staging / "confusion_matrices" / f"{spec.run_id}_{suffix}.csv", labels, matrix)
            torch.set_num_threads(cpu_threads)
        finally:
            torch.set_num_threads(cpu_threads)

        summary_records = []
        parameter_counts = {row["model"]: row["parameter_count"] for row in efficiency_records}
        for replay in replay_records:
            metrics = next(item[3] for item in run_inputs if item[0].run_id == replay["run_id"])
            summary_records.append({
                "run_id": replay["run_id"],
                "model": replay["model"],
                "seed": replay["seed"],
                "best_epoch": replay["best_epoch"],
                "parameter_count": parameter_counts[replay["model"]],
                "accuracy": replay["metrics_accuracy"],
                "macro_f1": replay["metrics_macro_f1"],
                "validation_loss": _best_validation(metrics, replay["run_id"])["loss"],
                "low_snr_accuracy": float(np.mean([_best_validation(metrics, replay["run_id"])["per_snr_accuracy"][snr] for snr in LOW_SNR_VALUES])),
            })
        summary_rows = _aggregate_summary([{**record, "best": _best_validation(next(item[3] for item in run_inputs if item[0].run_id == record["run_id"]), record["run_id"])} for record in summary_records])
        aggregate_snr = _aggregate_snr([{"model": spec.model, "best": _best_validation(metrics, spec.run_id)} for spec, _, _, metrics in run_inputs])
        efficiency_rows = _aggregate_efficiency(efficiency_records)

        _write_csv(staging / "summary_runs.csv", list(summary_records[0]), summary_records)
        _write_csv(staging / "summary_models.csv", list(summary_rows[0]), summary_rows)
        _write_csv(staging / "efficiency_runs.csv", list(efficiency_records[0]), efficiency_records)
        _write_csv(staging / "efficiency_models.csv", list(efficiency_rows[0]), efficiency_rows)
        snr_rows = [row for model in EXPECTED_MODELS for row in aggregate_snr[model]]
        _write_csv(staging / "per_snr_accuracy.csv", list(snr_rows[0]), snr_rows)
        _write_csv(staging / "replay_validation.csv", list(replay_records[0]), replay_records)
        _plot_snr_curves(staging / "figures" / "per_snr_accuracy.png", aggregate_snr)

        aggregate_matrices: dict[str, dict[str, np.ndarray]] = {model: {} for model in EXPECTED_MODELS}
        for spec, _, _, _ in run_inputs:
            for suffix in ["overall", *[f"snr_{snr:+d}" for snr in REPRESENTATIVE_SNRS]]:
                matrix_path = staging / "confusion_matrices" / f"{spec.run_id}_{suffix}.csv"
                values = np.loadtxt(matrix_path, delimiter=",", skiprows=1, usecols=range(1, len(labels) + 1), dtype=np.int64)
                aggregate_matrices[spec.model][suffix] = aggregate_matrices[spec.model].get(suffix, np.zeros_like(values)) + values
        for model in EXPECTED_MODELS:
            for suffix, matrix in aggregate_matrices[model].items():
                base = staging / "confusion_matrices" / f"{model}_{suffix}"
                _write_matrix_csv(base.with_suffix(".csv"), labels, matrix)
                _plot_confusion(base.with_suffix(".png"), matrix, labels, f"{model} validation confusion matrix ({suffix})")

        markdown = _markdown_report(summary_rows, efficiency_rows, report_dir=report_dir, summary=summary)
        _write_text(staging / "baseline-report.md", markdown)
        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "report-manifest.json":
                files.append({"path": path.relative_to(staging).as_posix(), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size})
        report_manifest = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "purpose": "baseline_report_and_freeze_evidence",
            "test_accessed": False,
            "bindings": {
                "training_project_commit": training_project_commit,
                "report_generation_project_commit": report_generation_project_commit,
                "split_manifest_sha256": split_manifest_sha256,
                "canonical_split_manifest_sha256": canonical_split_manifest_sha256,
                "assignment_sha256": assignment_sha256,
                "run_count": len(run_inputs),
                "seeds": list(EXPECTED_SEEDS),
                "models": list(EXPECTED_MODELS),
                "validation_sample_count": len(validation_dataset),
                "hdf5_file_sha256": _sha256_file(hdf5_path),
                "conversion_manifest_sha256": _sha256_file(conversion_manifest_path),
                "leakage_audit_sha256": _sha256_file(leakage_audit_path),
            },
            "replay": {
                "split": "validation",
                "representative_snrs_db": list(REPRESENTATIVE_SNRS),
                "metrics_reconciled": True,
                "test_dataset_constructed": False,
            },
            "efficiency_protocol": {
                "input_shape": [1, 2, 128],
                "gpu_warmup_iterations": warmup,
                "gpu_measurement_iterations": iterations,
                "cpu_threads": 1,
                "macs_definition": "multiply-accumulate operations for Conv/Linear/LSTM gates",
            },
            "files": files,
        }
        _write_text(staging / "report-manifest.json", json.dumps(report_manifest, indent=2, sort_keys=True) + "\n")
        if report_dir.exists():
            raise BaselineReportError(f"Refusing to overwrite report directory: {report_dir}")
        shutil.move(str(staging), str(report_dir))
        return report_manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
