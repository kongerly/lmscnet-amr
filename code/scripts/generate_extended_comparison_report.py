"""Generate unified validation-only calibration and efficiency report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.evaluation import count_macs, count_parameters  # noqa: E402
from na_lmscnet.models import build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402

SEEDS = (13, 37, 73, 101, 137)
MODELS = (
    "lmscnet_s0_k15",
    "lmscnet_s1",
    "lmscnet_s2",
    "cnn2",
    "cldnn",
    "resnet1d",
    "resnet1d_macs",
    "mobilenetv2_1d",
    "mcldnn",
    "se_msfn_1d",
)
LOW_SNR = (-10, -8, -6, -4, -2, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _model_paths(
    model: str,
    seed: int,
    final_root: Path,
    extended_root: Path,
) -> tuple[Path, Path, Path]:
    if model in {"lmscnet_s0_k15", "lmscnet_s1", "lmscnet_s2"}:
        group = final_root / "final-family-multiseed"
        run_id = f"{model}-seed-{seed}"
    elif model in {"cnn2", "cldnn", "resnet1d"}:
        group = final_root / "baseline-multiseed"
        run_id = f"{model}-seed-{seed}"
    else:
        group = extended_root / "multiseed"
        run_id = f"{model}-seed-{seed}"
    return (
        group / "configs" / f"{run_id}.yml",
        group / run_id / "metrics.json",
        group / run_id / "best.pt",
    )


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, confidence.size)
    result = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if np.any(mask):
            result += float(mask.sum() / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def _replay(config: Any, checkpoint_path: Path, dataset: RadioML2016HDF5Dataset, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model.get("expansion", 1.25)),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    workers = int(config.data["num_workers"])
    loader = DataLoader(dataset, batch_size=int(config.data["batch_size"]), shuffle=False, num_workers=workers, pin_memory=bool(config.data["pin_memory"]), persistent_workers=workers > 0)
    logits_parts: list[np.ndarray] = []
    targets_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    ids: list[str] = []
    amp = bool(config.training["amp"]) and device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(batch["iq"].to(device, non_blocking=True))
            logits = output["logits"] if isinstance(output, dict) else output
            logits_parts.append(logits.float().cpu().numpy())
            targets_parts.append(batch["modulation"].numpy())
            snr_parts.append(batch["snr"].numpy())
            ids.extend(str(value) for value in batch["sample_id"])
    logits = np.concatenate(logits_parts)
    targets = np.concatenate(targets_parts).astype(np.int64)
    snr = np.concatenate(snr_parts).astype(np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    low = np.isin(snr, LOW_SNR)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    return {
        "sample_ids": ids,
        "targets": targets,
        "snr": snr,
        "predictions": predictions,
        "accuracy": float(correct.mean()),
        "low_snr_accuracy": float(correct[low].mean()),
        "macro_f1": _macro_f1(predictions, targets, probabilities.shape[1]),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean()),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": _ece(probabilities.max(axis=1), correct.astype(np.float64)),
        "model": model,
    }


def _macro_f1(predictions: np.ndarray, targets: np.ndarray, classes: int) -> float:
    matrix = np.bincount(targets * classes + predictions, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(matrix).astype(np.float64)
    denom = 2 * tp + matrix.sum(axis=0) - tp + matrix.sum(axis=1) - tp
    return float(np.divide(2 * tp, denom, out=np.zeros(classes), where=denom > 0).mean())


def _latency(model: torch.nn.Module, device: torch.device, warmup: int, iterations: int) -> float:
    model.eval().to(device)
    tensor = torch.zeros((1, 2, 128), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--extended-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise ValueError("Report output must remain outside repository")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml",
        "dataset_spec_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml",
        "conversion_contract_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml",
        "preprocessing": "per_sample_max_abs",
    }
    rows: list[dict[str, Any]] = []
    model_efficiency: dict[str, dict[str, Any]] = {}
    with RadioML2016HDF5Dataset(split="validation", **common) as dataset:
        for model_name in MODELS:
            for seed in SEEDS:
                config_path, metrics_path, checkpoint_path = _model_paths(model_name, seed, args.final_root.resolve(strict=True), args.extended_root.resolve(strict=True))
                config = load_experiment_config(config_path)
                metrics = _json(metrics_path)
                if metrics.get("test_accessed") is not False:
                    raise ValueError(f"Replay input accessed test: {model_name} seed {seed}")
                if metrics.get("artifacts", {}).get("checkpoint_sha256") != _sha256(checkpoint_path):
                    raise ValueError(f"Checkpoint hash differs: {model_name} seed {seed}")
                replay = _replay(config, checkpoint_path, dataset, device)
                best_epoch = int(metrics["best_epoch"])
                recorded = metrics["history"][best_epoch - 1]["validation"]
                for field in ("accuracy", "macro_f1"):
                    if not math.isclose(float(replay[field]), float(recorded[field]), rel_tol=3e-5, abs_tol=3e-5):
                        raise ValueError(f"Replay {field} differs: {model_name} seed {seed}")
                rows.append({"model": model_name, "seed": seed, "accuracy": replay["accuracy"], "low_snr_accuracy": replay["low_snr_accuracy"], "macro_f1": replay["macro_f1"], "nll": replay["nll"], "brier": replay["brier"], "ece": replay["ece"], "checkpoint_sha256": _sha256(checkpoint_path), "test_accessed": False})
                if model_name not in model_efficiency:
                    parameters = count_parameters(replay["model"])
                    macs = count_macs(replay["model"], (1, 2, 128), torch.device("cpu"))
                    old_threads = torch.get_num_threads()
                    try:
                        torch.set_num_threads(1)
                        cpu_latency = _latency(replay["model"], torch.device("cpu"), args.warmup, args.iterations)
                    finally:
                        torch.set_num_threads(old_threads)
                    gpu_latency = _latency(replay["model"], device, args.warmup, args.iterations) if device.type == "cuda" else float("nan")
                    model_efficiency[model_name] = {"parameters": parameters, "macs": macs, "checkpoint_size_bytes": checkpoint_path.stat().st_size, "gpu_latency_ms": gpu_latency, "cpu_latency_ms": cpu_latency, "gpu_throughput_samples_per_s": 1000.0 / gpu_latency if math.isfinite(gpu_latency) and gpu_latency > 0 else None}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    aggregate: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        aggregate.append({"model": model, **{f"{metric}_mean": statistics.fmean(float(row[metric]) for row in model_rows) for metric in ("accuracy", "low_snr_accuracy", "macro_f1", "nll", "brier", "ece")}, **{f"{metric}_sample_std": statistics.stdev(float(row[metric]) for row in model_rows) for metric in ("accuracy", "low_snr_accuracy", "macro_f1", "nll", "brier", "ece")}, **model_efficiency[model]})
    pareto_models = []
    for row in aggregate:
        dominated = any(other["macro_f1_mean"] >= row["macro_f1_mean"] and other["parameters"] <= row["parameters"] and other["macs"] <= row["macs"] and (other["macro_f1_mean"] > row["macro_f1_mean"] or other["parameters"] < row["parameters"] or other["macs"] < row["macs"]) for other in aggregate)
        if not dominated:
            pareto_models.append(row["model"])
    report = {"schema_version": 1, "purpose": "unified_validation_calibration_efficiency_pareto_report", "test_accessed": False, "preprocessing_mode": "per_sample_max_abs", "validation_sample_count": len(dataset), "seeds": list(SEEDS), "warmup": args.warmup, "iterations": args.iterations, "models": aggregate, "pareto_models": pareto_models, "rows": rows}
    output_dir.mkdir(parents=True)
    (output_dir / "comparison-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "comparison-models.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(json.dumps({"output_dir": str(output_dir), "model_count": len(MODELS), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
