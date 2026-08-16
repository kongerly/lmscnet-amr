"""Build an external validation-only NA-LMSCNet report."""

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

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

from na_lmscnet.data.contracts import ModulationSample
from na_lmscnet.evaluation.efficiency import count_macs, count_parameters
from na_lmscnet.models import build_model
from na_lmscnet.training.engine import (
    ExperimentConfig,
    experiment_config_sha256,
    load_experiment_config,
)
from na_lmscnet.training.multiseed import _best_epoch_record

REPORT_SCHEMA_VERSION = 1
EXPECTED_SEEDS = (13, 37, 73, 101, 137)
REPRESENTATIVE_SNRS = (-10, -6, 0)
ALL_SNRS = tuple(range(-20, 20, 2))
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)
KERNELS = (3, 7, 15)
LABELS = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]


class NALMSCNetReportError(ValueError):
    """Raised when NA-LMSCNet report inputs violate the experiment contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NALMSCNetReportError(f"{field} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NALMSCNetReportError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise NALMSCNetReportError(f"{field} must be an object")
    return value


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NALMSCNetReportError(f"{field} must be a string-keyed mapping")
    return value


def _float(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise NALMSCNetReportError(f"{field} must be finite")
    return float(value)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    rows = []
    for label, values in zip(LABELS, matrix.tolist(), strict=True):
        rows.append(
            {
                "true_label": label,
                **{predicted: value for predicted, value in zip(LABELS, values, strict=True)},
            }
        )
    _write_csv(path, ["true_label", *LABELS], rows)


def _build_loader(
    dataset: Dataset[ModulationSample], config: ExperimentConfig
) -> DataLoader[ModulationSample]:
    workers = int(config.data["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config.data["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config.data["pin_memory"]),
        persistent_workers=workers > 0,
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _line_plot(
    path: Path,
    *,
    title: str,
    panels: list[tuple[str, list[tuple[str, str, list[tuple[float, float]]]]]],
    y_label: str,
) -> None:
    width, height = 1500, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, label_font = _font(28), _font(18), _font(16)
    draw.text((width / 2, 22), title, font=title_font, fill="#17212b", anchor="ma")
    panel_width = (width - 150) / len(panels)
    for panel_index, (panel_title, series) in enumerate(panels):
        left = 90 + panel_index * panel_width
        right = 60 + (panel_index + 1) * panel_width
        top, bottom = 100, height - 105
        draw.rectangle((left, top, right, bottom), outline="#7a8793", width=2)
        draw.text(
            ((left + right) / 2, 68), panel_title, font=axis_font, fill="#17212b", anchor="ma"
        )
        for step in range(6):
            y = bottom - (bottom - top) * step / 5
            draw.line((left, y, right, y), fill="#d9dee3", width=1)
            draw.text(
                (left - 12, y), f"{step / 5:.1f}", font=label_font, fill="#52606d", anchor="rm"
            )
        for index, value in enumerate(ALL_SNRS):
            x = left + (right - left) * index / (len(ALL_SNRS) - 1)
            draw.line((x, bottom, x, bottom + 6), fill="#7a8793", width=1)
            draw.text((x, bottom + 11), str(value), font=label_font, fill="#52606d", anchor="ma")
        for series_index, (name, color, points) in enumerate(series):
            coords = [
                (
                    left + (right - left) * ALL_SNRS.index(int(x)) / (len(ALL_SNRS) - 1),
                    bottom - (bottom - top) * max(0.0, min(1.0, y)),
                )
                for x, y in points
            ]
            draw.line(coords, fill=color, width=4, joint="curve")
            for x, y in coords:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            legend_x = left + 20 + series_index * 125
            draw.line((legend_x, top + 22, legend_x + 30, top + 22), fill=color, width=4)
            draw.text((legend_x + 37, top + 22), name, font=label_font, fill="#17212b", anchor="lm")
        draw.text(
            ((left + right) / 2, height - 37),
            "SNR (dB)",
            font=axis_font,
            fill="#17212b",
            anchor="ma",
        )
    draw.text((25, height / 2), y_label, font=axis_font, fill="#17212b", anchor="mm")
    image.save(path, format="PNG")


def _plot_snr(path: Path, rows: list[dict[str, object]]) -> None:
    points = [(float(row["snr_db"]), float(row["mean_accuracy"])) for row in rows]
    _line_plot(
        path,
        title="NA-LMSCNet RadioML 2016.10A validation accuracy",
        panels=[("Five-seed mean", [("NA-LMSCNet", "#146c94", points)])],
        y_label="Accuracy",
    )


def _plot_confusion(path: Path, matrix: np.ndarray, title: str) -> None:
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    cell, left, top = 82, 170, 110
    size = len(LABELS) * cell
    image = Image.new("RGB", (left + size + 80, top + size + 115), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, value_font = _font(26), _font(16), _font(14)
    draw.text((image.width / 2, 22), title, font=title_font, fill="#17212b", anchor="ma")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            value = float(normalized[row, column])
            shade = int(247 - 190 * value)
            color = (max(20, shade - 25), max(65, shade), min(255, shade + 8))
            x0, y0 = left + column * cell, top + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline="white")
            if value >= 0.05:
                draw.text(
                    (x0 + cell / 2, y0 + cell / 2),
                    f"{value:.2f}",
                    font=value_font,
                    fill="white" if value > 0.55 else "#17212b",
                    anchor="mm",
                )
    for index, label in enumerate(LABELS):
        draw.text(
            (left - 12, top + index * cell + cell / 2),
            label,
            font=label_font,
            fill="#17212b",
            anchor="rm",
        )
        draw.text(
            (left + index * cell + cell / 2, top + size + 12),
            label,
            font=label_font,
            fill="#17212b",
            anchor="ma",
        )
    draw.text(
        (left + size / 2, image.height - 30),
        "Predicted modulation",
        font=label_font,
        fill="#17212b",
        anchor="ma",
    )
    draw.text((20, top + size / 2), "True modulation", font=label_font, fill="#17212b", anchor="mm")
    image.save(path, format="PNG")


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


def _load_run_inputs(
    output_root: Path,
    summary: dict[str, Any],
    project_commit: str,
    split_sha: str,
    assignment_sha: str,
) -> list[tuple[dict[str, Any], Path, Path, dict[str, Any], ExperimentConfig]]:
    if (
        summary.get("schema_version") != 1
        or summary.get("purpose") != "baseline_multi_seed_training"
    ):
        raise NALMSCNetReportError("Invalid NA-LMSCNet multi-seed summary purpose/schema")
    if summary.get("run_count") != 5 or summary.get("test_accessed") is not False:
        raise NALMSCNetReportError("NA-LMSCNet summary must contain five validation-only runs")
    bindings = _mapping(summary.get("bindings"), "summary bindings")
    expected_bindings = {
        "project_commit": project_commit,
        "split_manifest_sha256": split_sha,
        "assignment_sha256": assignment_sha,
        "seeds": list(EXPECTED_SEEDS),
    }
    if bindings != expected_bindings:
        raise NALMSCNetReportError("NA-LMSCNet summary bindings differ from requested artifacts")
    result = []
    for seed in EXPECTED_SEEDS:
        run_id = f"na_lmscnet-seed-{seed}"
        record = next((item for item in summary["runs"] if item.get("run_id") == run_id), None)
        if not isinstance(record, dict):
            raise NALMSCNetReportError(f"Missing summary record {run_id}")
        config_path = output_root / "configs" / str(record["config_filename"])
        run_dir = output_root / run_id
        config = load_experiment_config(config_path)
        metrics = _load_json(run_dir / "metrics.json", f"metrics {run_id}")
        if config.model["name"] != "na_lmscnet" or config.test_access != "forbidden":
            raise NALMSCNetReportError(f"Invalid NA-LMSCNet config {run_id}")
        if experiment_config_sha256(config_path) != record.get("config_sha256"):
            raise NALMSCNetReportError(f"Config digest mismatch {run_id}")
        if _sha256_file(run_dir / "best.pt") != record.get("checkpoint_sha256") or _sha256_file(
            run_dir / "best.pt"
        ) != _mapping(metrics.get("artifacts"), f"artifacts {run_id}").get("checkpoint_sha256"):
            raise NALMSCNetReportError(f"Checkpoint digest mismatch {run_id}")
        expected_run_bindings = {
            "assignment_sha256": assignment_sha,
            "experiment_config_sha256": record["config_sha256"],
            "project_commit": project_commit,
            "seed": seed,
            "split_manifest_sha256": split_sha,
        }
        if (
            metrics.get("bindings") != expected_run_bindings
            or metrics.get("test_accessed") is not False
            or (run_dir / "last.pt").exists()
        ):
            raise NALMSCNetReportError(f"Run binding or isolation mismatch {run_id}")
        result.append((record, config_path, run_dir, metrics, config))
    return result


@torch.inference_mode()
def _replay_run(
    model: nn.Module,
    checkpoint_path: Path,
    dataset: Dataset[ModulationSample],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 1
        or not isinstance(checkpoint.get("model_state_dict"), dict)
    ):
        raise NALMSCNetReportError(f"Invalid checkpoint {checkpoint_path.name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    confusion = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)
    snr_confusions: dict[int, torch.Tensor] = {}
    correct_by_snr: dict[int, int] = defaultdict(int)
    count_by_snr: dict[int, int] = defaultdict(int)
    weights_by_snr: dict[int, list[np.ndarray]] = defaultdict(list)
    snr_errors: list[torch.Tensor] = []
    snr_error_sum: dict[int, float] = defaultdict(float)
    loader = _build_loader(dataset, config)
    amp = bool(config.training["amp"]) and device.type == "cuda"
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(iq)
        logits = outputs["logits"]
        predictions = logits.argmax(dim=1).cpu()
        targets = batch["modulation"].cpu()
        snr = batch["snr"].to(torch.int64)
        flat = targets * len(LABELS) + predictions
        confusion += torch.bincount(flat, minlength=len(LABELS) ** 2).reshape(
            len(LABELS), len(LABELS)
        )
        batch_errors = (outputs["snr_hat"].cpu() - batch["snr"].to(torch.float32)).abs()
        snr_errors.append(batch_errors)
        weights = outputs["scale_weights"].cpu().numpy()
        for index, value in enumerate(snr.tolist()):
            correct_by_snr[int(value)] += int(predictions[index] == targets[index])
            count_by_snr[int(value)] += 1
            snr_error_sum[int(value)] += float(batch_errors[index])
            weights_by_snr[int(value)].append(weights[index])
        for value in sorted(int(item) for item in snr.unique().tolist()):
            mask = snr == value
            snr_flat = targets[mask] * len(LABELS) + predictions[mask]
            matrix = snr_confusions.setdefault(
                value, torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)
            )
            matrix += torch.bincount(snr_flat, minlength=len(LABELS) ** 2).reshape(
                len(LABELS), len(LABELS)
            )
    if not snr_errors:
        raise NALMSCNetReportError("Validation replay consumed zero samples")
    true_positive = confusion.diag().to(torch.float64)
    false_positive = confusion.sum(dim=0).to(torch.float64) - true_positive
    false_negative = confusion.sum(dim=1).to(torch.float64) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = torch.where(
        denominator > 0, 2 * true_positive / denominator, torch.zeros_like(denominator)
    )
    per_snr = {value: correct_by_snr[value] / count_by_snr[value] for value in sorted(count_by_snr)}
    scale_rows = []
    for value in sorted(weights_by_snr):
        matrix = np.stack(weights_by_snr[value], axis=0)
        if matrix.ndim != 3 or matrix.shape[1:] != (6, 3):
            raise NALMSCNetReportError("Scale-weight replay shape is invalid")
        for block in range(matrix.shape[1]):
            for kernel_index, kernel in enumerate(KERNELS):
                values = matrix[:, block, kernel_index]
                scale_rows.append(
                    {
                        "snr_db": value,
                        "block": block + 1,
                        "kernel": kernel,
                        "mean_weight": float(values.mean()),
                        "std_weight": float(values.std(ddof=1)),
                        "sample_count": len(values),
                    }
                )
    return {
        "accuracy": float(confusion.diag().sum() / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "sample_count": int(confusion.sum()),
        "per_snr_accuracy": per_snr,
        "per_snr_snr_mae_db": {
            value: snr_error_sum[value] / count_by_snr[value] for value in sorted(count_by_snr)
        },
        "snr_mae_db": float(torch.cat(snr_errors).mean()),
        "confusion": confusion.numpy(),
        "snr_confusions": {key: value.numpy() for key, value in snr_confusions.items()},
        "scale_weights": scale_rows,
    }


def _aggregate_metric(rows: list[dict[str, object]], field: str) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def _aggregate_scale_weights(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for snr in ALL_SNRS:
        for block in range(1, 7):
            for kernel in KERNELS:
                selected = [
                    row
                    for row in rows
                    if row["snr_db"] == snr and row["block"] == block and row["kernel"] == kernel
                ]
                if len(selected) != len(EXPECTED_SEEDS):
                    raise NALMSCNetReportError("Scale-weight evidence is incomplete")
                means = np.asarray([row["mean_weight"] for row in selected], dtype=np.float64)
                result.append(
                    {
                        "snr_db": snr,
                        "block": block,
                        "kernel": kernel,
                        "mean_weight": float(means.mean()),
                        "std_across_seeds": float(means.std(ddof=1)),
                        "seed_count": len(means),
                        "samples_per_seed": int(selected[0]["sample_count"]),
                    }
                )
    return result


def _plot_scale_weights(path: Path, rows: list[dict[str, object]]) -> None:
    colors = {3: "#146c94", 7: "#c44536", 15: "#2d7d46"}
    panels = []
    for block in (1, 6):
        series = []
        for kernel in KERNELS:
            values = [row for row in rows if row["block"] == block and row["kernel"] == kernel]
            series.append(
                (
                    f"k={kernel}",
                    colors[kernel],
                    [(float(row["snr_db"]), float(row["mean_weight"])) for row in values],
                )
            )
        panels.append((f"Block {block}", series))
    _line_plot(
        path,
        title="NA-LMSCNet noise-conditioned scale weights",
        panels=panels,
        y_label="Mean scale weight",
    )


def _write_report_markdown(
    path: Path,
    summary: dict[str, object],
    efficiency: dict[str, object],
    scale_summary: list[dict[str, object]],
) -> None:
    block6 = {
        (int(row["snr_db"]), int(row["kernel"])): float(row["mean_weight"])
        for row in scale_summary
        if row["block"] == 6 and row["snr_db"] in {-20, 0}
    }
    lines = [
        "# NA-LMSCNet RadioML 2016.10A Validation Report",
        "",
        "This report uses only the frozen validation split and neither constructs nor accesses a test dataset. Config, checkpoint, split, assignment, commit, and seed bindings were verified for all five runs.",
        "",
        f"- training commit: `{summary['training_project_commit']}`",
        f"- run count: `{summary['run_count']}`",
        f"- validation samples per run: `{summary['validation_sample_count']}`",
        "",
        "## Summary",
        "",
        "| Metric | Mean +/- std |",
        "| --- | ---: |",
    ]
    for label, field, suffix in [
        ("Accuracy", "accuracy", ""),
        ("Macro-F1", "macro_f1", ""),
        ("Low-SNR accuracy", "low_snr_accuracy", ""),
        ("Validation loss", "validation_loss", ""),
        ("SNR MAE", "snr_mae_db", " dB"),
    ]:
        lines.append(
            f"| {label} | {summary[field + '_mean']:.4f} +/- {summary[field + '_std']:.4f}{suffix} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `per_snr_accuracy.csv` and `figures/per_snr_accuracy.png`",
            "- `confusion_matrices/`: each run and aggregate overall/`-10`/`-6`/`0 dB` matrices",
            "- `snr_metrics.csv`: per-run overall and per-SNR SNR MAE",
            "- `scale_weights_by_snr.csv` and `figures/scale_weights_by_snr.png`: six blocks, kernels `[3,7,15]`",
            "- `efficiency.csv`: parameters, MACs, checkpoint size, GPU/CPU latency",
            "",
            "## Efficiency",
            "",
            f"- parameters: `{efficiency['parameter_count']:,}`",
            f"- MACs: `{efficiency['macs']:,}`",
            f"- checkpoint size mean: `{efficiency['checkpoint_size_mean']:.0f}` bytes",
            f"- GPU latency mean: `{efficiency['gpu_latency_mean_ms']:.4f}` ms",
            f"- CPU single-thread latency mean: `{efficiency['cpu_latency_mean_ms']:.4f}` ms",
            "",
            "## Scale-weight observation",
            "",
            f"- Block 6 at `-20 dB`, kernels `[3,7,15]`: `[{block6[(-20, 3)]:.3f}, {block6[(-20, 7)]:.3f}, {block6[(-20, 15)]:.3f}]`",
            f"- Block 6 at `0 dB`, kernels `[3,7,15]`: `[{block6[(0, 3)]:.3f}, {block6[(0, 7)]:.3f}, {block6[(0, 15)]:.3f}]`",
            "- The observation does not directly support the directional hypothesis that low SNR increases weight on the larger-receptive-field branch. All blocks and ablations must be considered; a local weight trend is not sufficient evidence for the mechanism.",
            "",
            "This is validation-only evidence, not a test result, and it does not support capture-disjoint or real OTA generalization claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate_na_lmscnet_report(
    *,
    output_root: Path,
    report_dir: Path,
    hdf5_path: Path,
    split_manifest_path: Path,
    leakage_audit_path: Path,
    validation_dataset: Dataset[ModulationSample],
    project_root: Path,
    training_project_commit: str,
    report_generation_project_commit: str,
    device: torch.device,
    warmup: int = 100,
    iterations: int = 1000,
) -> dict[str, object]:
    output_root = output_root.resolve(strict=True)
    report_dir = report_dir.resolve()
    project_root = project_root.resolve(strict=True)
    if report_dir == project_root or project_root in report_dir.parents:
        raise NALMSCNetReportError("Report output must be outside repository")
    if warmup < 0 or iterations < 1:
        raise NALMSCNetReportError("Latency warmup must be non-negative and iterations positive")
    if report_dir.exists():
        raise NALMSCNetReportError(f"Refusing to overwrite report directory: {report_dir}")
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    split_sha = _sha256_file(split_manifest_path)
    assignment_sha = validation_dataset.assignment_sha256
    summary = _load_json(output_root / "multi-seed-summary.json", "NA-LMSCNet multi-seed summary")
    inputs = _load_run_inputs(
        output_root, summary, training_project_commit, split_sha, assignment_sha
    )
    if (
        validation_dataset.split_manifest_sha256
        != _mapping(_load_json(leakage_audit_path, "leakage audit"), "leakage audit")[
            "split_manifest_sha256"
        ]
    ):
        raise NALMSCNetReportError("Validation dataset split digest differs from leakage audit")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise NALMSCNetReportError("CUDA requested but unavailable")
    staging = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=report_dir.parent))
    (staging / "figures").mkdir(parents=True)
    (staging / "confusion_matrices").mkdir()
    old_threads = torch.get_num_threads()
    try:
        replay_rows = []
        scale_rows = []
        snr_rows = []
        snr_metric_rows = []
        matrices: dict[str, np.ndarray] = {}
        snr_matrices: dict[tuple[int, str], np.ndarray] = {}
        efficiency_rows = []
        for record, _config_path, run_dir, metrics, config in inputs:
            model = build_model(
                "na_lmscnet", num_classes=11, dropout=float(config.model["dropout"])
            )
            replay = _replay_run(model, run_dir / "best.pt", validation_dataset, config, device)
            best = _best_epoch_record(metrics)
            validation = _mapping(best["validation"], f"best validation {record['run_id']}")
            for field in ("accuracy", "macro_f1", "snr_mae_db"):
                expected = replay[field]
                actual = validation[field] if field != "macro_f1" else validation[field]
                if not math.isclose(float(expected), float(actual), rel_tol=3e-5, abs_tol=3e-5):
                    raise NALMSCNetReportError(
                        f"Replay metric differs for {record['run_id']}: {field}"
                    )
            metrics_per_snr = _mapping(
                validation.get("per_snr_accuracy"),
                f"per-SNR validation {record['run_id']}",
            )
            for snr, accuracy in replay["per_snr_accuracy"].items():
                if not math.isclose(
                    accuracy,
                    _float(metrics_per_snr.get(f"{snr:+d}"), f"per-SNR accuracy {snr}"),
                    rel_tol=3e-5,
                    abs_tol=3e-5,
                ):
                    raise NALMSCNetReportError(
                        f"Replay per-SNR accuracy differs for {record['run_id']}: {snr:+d}"
                    )
            replay_rows.append(
                {
                    "run_id": record["run_id"],
                    "seed": record["seed"],
                    "best_epoch": record["best_epoch"],
                    "accuracy": replay["accuracy"],
                    "macro_f1": replay["macro_f1"],
                    "validation_loss": float(
                        validation["validation_loss"]
                        if "validation_loss" in validation
                        else best["validation_loss"]
                    ),
                    "snr_mae_db": replay["snr_mae_db"],
                    "sample_count": replay["sample_count"],
                }
            )
            for snr, accuracy in replay["per_snr_accuracy"].items():
                snr_rows.append(
                    {
                        "run_id": record["run_id"],
                        "seed": record["seed"],
                        "snr_db": snr,
                        "accuracy": accuracy,
                    }
                )
                snr_metric_rows.append(
                    {
                        "run_id": record["run_id"],
                        "seed": record["seed"],
                        "snr_db": snr,
                        "accuracy": accuracy,
                        "snr_mae_db": replay["per_snr_snr_mae_db"][snr],
                    }
                )
            scale_rows.extend(
                [
                    {"run_id": record["run_id"], "seed": record["seed"], **row}
                    for row in replay["scale_weights"]
                ]
            )
            matrices[record["run_id"]] = replay["confusion"]
            for snr in REPRESENTATIVE_SNRS:
                snr_matrices[(snr, record["run_id"])] = replay["snr_confusions"][snr]
            parameters = count_parameters(model)
            macs = count_macs(model, (1, 2, 128), device)
            gpu_latency = (
                _measure_latency(model, device, warmup, iterations)
                if device.type == "cuda"
                else float("nan")
            )
            torch.set_num_threads(1)
            cpu_latency = _measure_latency(model, torch.device("cpu"), warmup, iterations)
            torch.set_num_threads(old_threads)
            efficiency_rows.append(
                {
                    "run_id": record["run_id"],
                    "seed": record["seed"],
                    "parameter_count": parameters,
                    "macs": macs,
                    "checkpoint_size_bytes": (run_dir / "best.pt").stat().st_size,
                    "gpu_latency_ms": gpu_latency,
                    "gpu_throughput_samples_per_s": 1000 / gpu_latency
                    if math.isfinite(gpu_latency)
                    else float("nan"),
                    "cpu_latency_ms": cpu_latency,
                    "gpu_warmup": warmup,
                    "gpu_iterations": iterations,
                    "cpu_threads": 1,
                }
            )
        snr_summary = []
        for snr in ALL_SNRS:
            values = np.asarray([row["accuracy"] for row in snr_rows if row["snr_db"] == snr])
            snr_summary.append(
                {
                    "snr_db": snr,
                    "mean_accuracy": float(values.mean()),
                    "std_accuracy": float(values.std(ddof=1)),
                    "run_count": len(values),
                }
            )
        metric_summary = {}
        for field in ("accuracy", "macro_f1", "validation_loss", "snr_mae_db"):
            values = np.asarray([row[field] for row in replay_rows], dtype=np.float64)
            metric_summary[field + "_mean"] = float(values.mean())
            metric_summary[field + "_std"] = float(values.std(ddof=1))
        low_snr_values = np.asarray(
            [
                np.mean(
                    [
                        row["accuracy"]
                        for row in snr_rows
                        if row["seed"] == seed and row["snr_db"] in LOW_SNR_VALUES
                    ]
                )
                for seed in EXPECTED_SEEDS
            ],
            dtype=np.float64,
        )
        metric_summary["low_snr_accuracy_mean"] = float(low_snr_values.mean())
        metric_summary["low_snr_accuracy_std"] = float(low_snr_values.std(ddof=1))
        scale_summary = _aggregate_scale_weights(scale_rows)
        efficiency = {
            "parameter_count": efficiency_rows[0]["parameter_count"],
            "macs": efficiency_rows[0]["macs"],
            "checkpoint_size_mean": float(
                np.mean([row["checkpoint_size_bytes"] for row in efficiency_rows])
            ),
            "gpu_latency_mean_ms": float(
                np.nanmean([row["gpu_latency_ms"] for row in efficiency_rows])
            ),
            "cpu_latency_mean_ms": float(
                np.mean([row["cpu_latency_ms"] for row in efficiency_rows])
            ),
        }
        _write_csv(staging / "summary_runs.csv", list(replay_rows[0]), replay_rows)
        summary_model_row = {"model": "na_lmscnet", "run_count": 5, **metric_summary}
        _write_csv(staging / "summary_models.csv", list(summary_model_row), [summary_model_row])
        _write_csv(staging / "per_snr_accuracy.csv", list(snr_rows[0]), snr_rows)
        _write_csv(staging / "per_snr_accuracy_summary.csv", list(snr_summary[0]), snr_summary)
        _write_csv(staging / "snr_metrics.csv", list(snr_metric_rows[0]), snr_metric_rows)
        _write_csv(staging / "scale_weights_by_snr.csv", list(scale_rows[0]), scale_rows)
        _write_csv(
            staging / "scale_weights_by_snr_summary.csv",
            list(scale_summary[0]),
            scale_summary,
        )
        _write_csv(staging / "efficiency.csv", list(efficiency_rows[0]), efficiency_rows)
        efficiency_model_row = {"model": "na_lmscnet", "run_count": 5, **efficiency}
        _write_csv(
            staging / "efficiency_models.csv",
            list(efficiency_model_row),
            [efficiency_model_row],
        )
        _plot_snr(staging / "figures" / "per_snr_accuracy.png", snr_summary)
        _plot_scale_weights(staging / "figures" / "scale_weights_by_snr.png", scale_summary)
        for run_id, matrix in matrices.items():
            _write_matrix_csv(staging / "confusion_matrices" / f"{run_id}_overall.csv", matrix)
            _plot_confusion(
                staging / "confusion_matrices" / f"{run_id}_overall.png",
                matrix,
                f"{run_id} validation confusion",
            )
            for snr in REPRESENTATIVE_SNRS:
                matrix = snr_matrices[(snr, run_id)]
                _write_matrix_csv(
                    staging / "confusion_matrices" / f"{run_id}_snr_{snr:+d}.csv", matrix
                )
                _plot_confusion(
                    staging / "confusion_matrices" / f"{run_id}_snr_{snr:+d}.png",
                    matrix,
                    f"{run_id} validation confusion ({snr:+d} dB)",
                )
        for suffix, matrix_map in [
            ("overall", matrices),
            *[
                (f"snr_{snr:+d}", {run_id: snr_matrices[(snr, run_id)] for run_id in matrices})
                for snr in REPRESENTATIVE_SNRS
            ],
        ]:
            aggregate = sum(matrix_map.values())
            _write_matrix_csv(
                staging / "confusion_matrices" / f"na_lmscnet_{suffix}.csv", aggregate
            )
            _plot_confusion(
                staging / "confusion_matrices" / f"na_lmscnet_{suffix}.png",
                aggregate,
                f"NA-LMSCNet validation confusion ({suffix})",
            )
        report_summary = {
            "training_project_commit": training_project_commit,
            "report_generation_project_commit": report_generation_project_commit,
            "split_manifest_sha256": split_sha,
            "assignment_sha256": assignment_sha,
            "run_count": 5,
            "validation_sample_count": len(validation_dataset),
            "test_accessed": False,
            **metric_summary,
        }
        _write_report_markdown(
            staging / "na-lmscnet-report.md",
            report_summary,
            efficiency,
            scale_summary,
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
        manifest = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "purpose": "na_lmscnet_validation_report",
            "test_accessed": False,
            "test_dataset_constructed": False,
            "bindings": {
                **report_summary,
                "hdf5_file_sha256": _sha256_file(hdf5_path),
                "leakage_audit_sha256": _sha256_file(leakage_audit_path),
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
        (staging / "report-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        shutil.move(str(staging), str(report_dir))
        return manifest
    except Exception:
        torch.set_num_threads(old_threads)
        shutil.rmtree(staging, ignore_errors=True)
        raise
