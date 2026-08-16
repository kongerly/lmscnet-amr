"""Generate the validation-only CNN2 preprocessing and source-aligned diagnosis report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import GlobalZScoreStatistics, RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.models import SourceVTCNN2, build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.metrics import classification_metrics  # noqa: E402
from run_cnn2_source_aligned import SourceSubset  # noqa: E402

MODES = ("raw", "per_sample_dc_power", "global_zscore", "per_sample_max_abs")
LABELS = ("8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM")
FOCUS = ("AM-DSB", "AM-SSB", "WBFM", "QAM16", "QAM64")
LOW_SNRS = (-10, -8, -6, -4, -2, 0)
MATRIX_SNRS = (-10, -6, 0, 18)
DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _confusion(targets: torch.Tensor, predictions: torch.Tensor) -> np.ndarray:
    flat = targets.to(torch.int64) * len(LABELS) + predictions.to(torch.int64)
    return torch.bincount(flat, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS)).numpy()


@torch.inference_mode()
def _replay(
    model: torch.nn.Module,
    checkpoint: Path,
    dataset,
    device: torch.device,
    batch_size: int,
    *,
    amp: bool = False,
) -> dict[str, object]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    predictions = []
    targets = []
    snrs = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(batch["iq"].to(device, non_blocking=True))
        predictions.append(logits.argmax(1).cpu())
        targets.append(batch["modulation"].cpu())
        snrs.append(batch["snr"].cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    snr = torch.cat(snrs).to(torch.int64)
    metrics = classification_metrics(prediction, target, snr, num_classes=len(LABELS))
    low_snr_mask = torch.zeros_like(snr, dtype=torch.bool)
    for value in LOW_SNRS:
        low_snr_mask |= snr == value
    low_snr_accuracy = float(
        (prediction[low_snr_mask] == target[low_snr_mask]).to(torch.float64).mean()
    )
    matrices = {"overall": _confusion(target, prediction)}
    for value in MATRIX_SNRS:
        mask = snr == value
        matrices[f"snr_{value:+d}"] = _confusion(target[mask], prediction[mask])
    class_accuracy = {}
    class_low_snr_accuracy = {}
    class_snr_accuracy = {}
    for index, label in enumerate(LABELS):
        mask = target == index
        class_accuracy[label] = float((prediction[mask] == target[mask]).to(torch.float64).mean())
        low_snr_class = mask & low_snr_mask
        class_low_snr_accuracy[label] = float(
            (prediction[low_snr_class] == target[low_snr_class]).to(torch.float64).mean()
        )
        for snr_value in sorted(int(value) for value in snr.unique()):
            stratum = mask & (snr == snr_value)
            class_snr_accuracy[(label, snr_value)] = float(
                (prediction[stratum] == target[stratum]).to(torch.float64).mean()
            )
    return {
        "metrics": metrics,
        "low_snr_accuracy": low_snr_accuracy,
        "matrices": matrices,
        "class_accuracy": class_accuracy,
        "class_low_snr_accuracy": class_low_snr_accuracy,
        "class_snr_accuracy": class_snr_accuracy,
        "prediction": prediction,
        "target": target,
        "snr": snr,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *LABELS])
        for label, row in zip(LABELS, matrix, strict=True):
            writer.writerow([label, *[int(value) for value in row]])


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _plot_matrix(path: Path, matrix: np.ndarray, title: str) -> None:
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    cell = 64
    left, top, right, bottom = 150, 80, 80, 180
    width = left + cell * len(LABELS) + right
    height = top + cell * len(LABELS) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(22, bold=True)
    label_font = _font(16)
    value_font = _font(13)
    draw.text((left, 22), title, fill="black", font=title_font)
    for row, label in enumerate(LABELS):
        y = top + row * cell
        draw.text((12, y + cell // 2 - 9), label, fill="black", font=label_font)
        for column, value in enumerate(normalized[row]):
            x = left + column * cell
            blue = (33, 113, 181)
            color = tuple(round(255 + float(value) * (channel - 255)) for channel in blue)
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline=(220, 220, 220))
            text_color = "white" if value >= 0.55 else "black"
            draw.text(
                (x + 5, y + cell // 2 - 8), f"{value:.2f}",
                fill=text_color, font=value_font,
            )
    for column, label in enumerate(LABELS):
        label_image = Image.new("RGBA", (120, 24), (255, 255, 255, 0))
        ImageDraw.Draw(label_image).text((0, 0), label, fill="black", font=label_font)
        rotated = label_image.rotate(55, expand=True)
        x = left + column * cell + cell // 2 - rotated.width // 2
        image.paste(rotated, (x, top + cell * len(LABELS) + 8), rotated)
    draw.text(
        (left + cell * len(LABELS) // 2 - 80, height - 28),
        "Predicted modulation", fill="black", font=label_font,
    )
    image.save(path)


def _plot_snr_accuracy(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 1300, 760
    left, top, right, bottom = 100, 70, 240, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(24, bold=True)
    label_font = _font(17)
    tick_font = _font(14)
    draw.text((left, 20), "CNN2 preprocessing diagnosis", fill="black", font=title_font)
    draw.line((left, top, left, top + plot_height), fill="black", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="black", width=2,
    )
    for step in range(6):
        value = step / 5
        y = round(top + plot_height * (1 - value))
        draw.line((left, y, left + plot_width, y), fill=(225, 225, 225), width=1)
        draw.text((42, y - 8), f"{value:.1f}", fill="black", font=tick_font)
    snrs = sorted({int(row["snr_db"]) for row in rows if row["experiment"] in MODES})
    x_for = {
        snr: round(left + index * plot_width / (len(snrs) - 1))
        for index, snr in enumerate(snrs)
    }
    for index, snr in enumerate(snrs):
        if index % 2 == 0 or index == len(snrs) - 1:
            x = x_for[snr]
            draw.line((x, top + plot_height, x, top + plot_height + 5), fill="black")
            draw.text((x - 12, top + plot_height + 12), str(snr), fill="black", font=tick_font)
    colors = {
        "raw": (60, 60, 60),
        "per_sample_dc_power": (213, 94, 0),
        "global_zscore": (0, 114, 178),
        "per_sample_max_abs": (0, 158, 115),
    }
    for legend_index, name in enumerate(MODES):
        values = sorted(
            (int(row["snr_db"]), float(row["accuracy"]))
            for row in rows if row["experiment"] == name
        )
        points = [(x_for[snr], round(top + plot_height * (1 - value))) for snr, value in values]
        draw.line(points, fill=colors[name], width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[name])
        legend_y = top + legend_index * 42
        draw.line((left + plot_width + 30, legend_y + 10, left + plot_width + 65, legend_y + 10), fill=colors[name], width=4)
        draw.text((left + plot_width + 75, legend_y), name, fill="black", font=tick_font)
    draw.text((left + plot_width // 2 - 35, height - 34), "SNR (dB)", fill="black", font=label_font)
    draw.text((8, top + plot_height // 2), "Accuracy", fill="black", font=label_font)
    image.save(path)


def main() -> int:
    args = parse_args()
    project_commit = _project_commit()
    device = torch.device(args.device)
    experiment_dir = args.experiment_dir.resolve(strict=True)
    report_dir = args.report_dir.resolve()
    if report_dir.exists():
        raise RuntimeError("Report directory already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=report_dir.parent))
    (staging / "confusion_matrices").mkdir()
    common = {
        "hdf5_path": args.hdf5, "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest, "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract, "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    stats_raw = json.loads((experiment_dir / "global-zscore-statistics.json").read_text(encoding="utf-8"))["statistics"]
    statistics = GlobalZScoreStatistics(
        channel_mean=tuple(stats_raw["channel_mean"]), channel_std=tuple(stats_raw["channel_std"]),
        scalar_count_per_channel=int(stats_raw["scalar_count_per_channel"]),
        split=stats_raw["split"], estimator=stats_raw["estimator"],
    )
    results = {}
    try:
        for mode in MODES:
            config = load_experiment_config(experiment_dir / "configs" / f"{mode}.yml")
            kwargs = {"preprocessing": mode, "global_zscore": statistics if mode == "global_zscore" else None}
            with RadioML2016HDF5Dataset(split="validation", **kwargs, **common) as dataset:
                model = build_model("cnn2", num_classes=11, dropout=float(config.model["dropout"]))
                results[mode] = _replay(
                    model, experiment_dir / mode / "best.pt", dataset, device,
                    int(config.data["batch_size"]), amp=bool(config.training["amp"]),
                )

        with (
            RadioML2016HDF5Dataset(split="train", preprocessing="raw", **common) as project_train,
            RadioML2016HDF5Dataset(split="validation", preprocessing="raw", **common) as project_validation,
        ):
            pool_rows = tuple(sorted((*project_train.rows, *project_validation.rows)))
            rng = np.random.RandomState(2016)
            selected = rng.choice(len(pool_rows), size=len(pool_rows) // 2, replace=False)
            evaluation_positions = np.asarray(sorted(set(range(len(pool_rows))) - set(int(value) for value in selected)), dtype=np.int64)
            source_validation = SourceSubset(project_train, project_validation, pool_rows, evaluation_positions)
            results["source-aligned"] = _replay(SourceVTCNN2(dropout=0.5), experiment_dir / "source-aligned" / "best.pt", source_validation, device, 1024)

        summary_rows = []
        class_rows = []
        class_snr_rows = []
        snr_rows = []
        for name, result in results.items():
            metrics = result["metrics"]
            summary_rows.append({"experiment": name, "accuracy": metrics.accuracy, "macro_f1": metrics.macro_f1, "low_snr_accuracy": result["low_snr_accuracy"], "sample_count": metrics.sample_count})
            for label, value in result["class_accuracy"].items():
                class_rows.append({
                    "experiment": name, "class": label, "accuracy": value,
                    "low_snr_accuracy": result["class_low_snr_accuracy"][label],
                    "focus_class": label in FOCUS,
                })
            for (label, snr), value in result["class_snr_accuracy"].items():
                class_snr_rows.append({
                    "experiment": name, "class": label, "snr_db": snr,
                    "accuracy": value, "focus_class": label in FOCUS,
                })
            for snr, value in metrics.per_snr_accuracy.items():
                snr_rows.append({"experiment": name, "snr_db": int(snr), "accuracy": value})
            for suffix, matrix in result["matrices"].items():
                base = staging / "confusion_matrices" / f"{name}_{suffix}"
                _write_matrix(base.with_suffix(".csv"), matrix)
                _plot_matrix(base.with_suffix(".png"), matrix, f"{name} validation confusion ({suffix})")
        _write_csv(staging / "summary.csv", summary_rows)
        _write_csv(staging / "class_accuracy.csv", class_rows)
        _write_csv(staging / "class_snr_accuracy.csv", class_snr_rows)
        _write_csv(staging / "per_snr_accuracy.csv", snr_rows)

        _plot_snr_accuracy(staging / "per_snr_accuracy.png", snr_rows)

        controlled = [row for row in summary_rows if row["experiment"] in MODES]
        source = next(row for row in summary_rows if row["experiment"] == "source-aligned")
        lines = [
            "# CNN2 Preprocessing and Source-Aligned Reproduction Diagnosis",
            "",
            "This report reads only the frozen train/validation data or internal repartitions of that pool; it neither constructs nor reads the project test dataset. The four preprocessing groups share the CNN2 architecture, augmentation, AdamW optimizer, cosine scheduler, training budget, early stopping, and Macro-F1 checkpoint rule. Input preprocessing is the only difference.",
            "",
            "## Four-Group Controlled Protocol",
            "",
            "| Preprocessing | Overall accuracy | Macro-F1 | Low-SNR accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in controlled:
            lines.append(f"| {row['experiment']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['low_snr_accuracy']:.6f} |")
        lines.extend(["", "Low SNR is the pooled accuracy over all samples at `-10/-8/-6/-4/-2/0 dB`. The stratified validation groups contain equal counts at each SNR, so this equals the arithmetic mean of the six per-SNR accuracies. The random source-aligned evaluation half does not assume exactly equal SNR counts. Per-SNR curves, all-class accuracy, class-by-SNR accuracy, and overall/`-10/-6/0/+18 dB` confusion matrices are available as CSV/PNG files in the same directory.", "", "### Focus Classes", "", "| Preprocessing | AM-DSB | AM-SSB | WBFM | QAM16 | QAM64 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name in MODES:
            values = results[name]["class_accuracy"]
            lines.append(f"| {name} | {values['AM-DSB']:.4f} | {values['AM-SSB']:.4f} | {values['WBFM']:.4f} | {values['QAM16']:.4f} | {values['QAM64']:.4f} |")
        lines.extend(["", "### Focus Classes (Low SNR)", "", "| Preprocessing | AM-DSB | AM-SSB | WBFM | QAM16 | QAM64 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name in MODES:
            values = results[name]["class_low_snr_accuracy"]
            lines.append(f"| {name} | {values['AM-DSB']:.4f} | {values['AM-SSB']:.4f} | {values['WBFM']:.4f} | {values['QAM16']:.4f} | {values['QAM64']:.4f} |")
        lines.extend(["", "## Source-Aligned Reproduction (Reported Separately)", "", "This group aligns with the raw I/Q input, random 50/50 split method, zero-padded VT-CNN2, dropout 0.5, Adam, batch size 1024, 100 epochs, validation-loss checkpoint, and patience 5 used in the original `radioML/examples@6c9ac6029ab1d0803442da7de8b7be04714bdebb` notebook. To preserve the project test lock, the 50/50 split is reconstructed only within the project train-plus-validation pool. The PyTorch port and explicit seed 2016 are documented adaptations. This group is not part of the controlled comparison above.", "", "| Accuracy | Macro-F1 | Low-SNR accuracy | Samples |", "| ---: | ---: | ---: | ---: |", f"| {source['accuracy']:.6f} | {source['macro_f1']:.6f} | {source['low_snr_accuracy']:.6f} | {source['sample_count']} |", "", "## Decision Boundary", "", "The final preprocessing choice must consider numerical results, per-SNR curves, and focus-class confusion together. If the seed-13 candidate difference is small or metric directions conflict, this report does not authorize a freeze based on that small difference; the issue must be recorded and evaluated across additional seeds.", ""])
        (staging / "diagnostic-report.md").write_text("\n".join(lines), encoding="utf-8")
        manifest = {
            "schema_version": 1, "purpose": "cnn2_preprocessing_diagnosis_report",
            "test_accessed": False, "report_generation_project_commit": project_commit,
            "bindings": {
                "experiment_protocol_sha256": _sha256_file(experiment_dir / "diagnostic-protocol.json"),
                "source_protocol_sha256": _sha256_file(experiment_dir / "source-aligned-protocol.json"),
                "split_manifest_sha256": _sha256_file(args.split_manifest),
            },
        }
        (staging / "report-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(report_dir))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"report_dir": str(report_dir), "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
