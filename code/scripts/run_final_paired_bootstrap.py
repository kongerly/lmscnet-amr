"""Replay formal validation predictions and run frozen paired hierarchical bootstraps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
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
from na_lmscnet.evaluation.core_ablation_multiseed_report import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    _paired_accuracy_bootstrap_suite,
    formal_contribution_decision,
    paired_hierarchical_bootstrap,
)
from na_lmscnet.models import build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.metrics import classification_metrics  # noqa: E402

SEEDS = (13, 37, 73, 101, 137)
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)
MODELS = ("lmscnet_s0_k15", "lmscnet_s1", "lmscnet_s2", "resnet1d")
COMPARISONS = (
    ("s1_minus_s0", "lmscnet_s1", "lmscnet_s0_k15"),
    ("s2_minus_s1", "lmscnet_s2", "lmscnet_s1"),
    ("s2_minus_resnet1d", "lmscnet_s2", "resnet1d"),
)


class FinalBootstrapError(ValueError):
    """Raised when formal replay or paired statistical evidence is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalBootstrapError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise FinalBootstrapError(f"{field} must contain a JSON object")
    return value


def _build_model(config: Any) -> torch.nn.Module:
    return build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model.get("expansion", 1.25)),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
    )


def _best_validation(metrics: dict[str, Any]) -> dict[str, Any]:
    epoch = metrics.get("best_epoch")
    history = metrics.get("history")
    if not isinstance(epoch, int) or not isinstance(history, list) or not 1 <= epoch <= len(history):
        raise FinalBootstrapError("Metrics best epoch is invalid")
    record = history[epoch - 1]
    if not isinstance(record, dict) or not isinstance(record.get("validation"), dict):
        raise FinalBootstrapError("Metrics best validation record is invalid")
    return record["validation"]


@torch.inference_mode()
def _replay(
    *,
    config: Any,
    checkpoint_path: Path,
    metrics: dict[str, Any],
    dataset: RadioML2016HDF5Dataset,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise FinalBootstrapError(f"Checkpoint is invalid: {checkpoint_path}")
    model = _build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    workers = int(config.data["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(config.data["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config.data["pin_memory"]),
        persistent_workers=workers > 0,
    )
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    snr_parts: list[torch.Tensor] = []
    sample_ids: list[str] = []
    amp = bool(config.training["amp"]) and device.type == "cuda"
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(batch["iq"].to(device, non_blocking=True))
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        logits_parts.append(logits.detach().float().cpu())
        target_parts.append(batch["modulation"].to(dtype=torch.int64, device="cpu"))
        snr_parts.append(batch["snr"].to(dtype=torch.int64, device="cpu"))
        sample_ids.extend(str(value) for value in batch["sample_id"])
    if not logits_parts:
        raise FinalBootstrapError("Validation replay consumed zero samples")
    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    snr_db = torch.cat(snr_parts)
    predictions = logits.argmax(dim=1)
    replay_metrics = classification_metrics(
        predictions,
        targets,
        snr_db,
        num_classes=int(config.model["num_classes"]),
    )
    recorded = _best_validation(metrics)
    for field in ("accuracy", "macro_f1"):
        if not math.isclose(
            float(getattr(replay_metrics, field)),
            float(recorded[field]),
            rel_tol=3e-5,
            abs_tol=3e-5,
        ):
            raise FinalBootstrapError(f"Replay differs from recorded {field}")
    for snr, accuracy in replay_metrics.per_snr_accuracy.items():
        if not math.isclose(
            float(accuracy),
            float(recorded["per_snr_accuracy"][snr]),
            rel_tol=3e-5,
            abs_tol=3e-5,
        ):
            raise FinalBootstrapError(f"Replay differs from recorded accuracy at {snr} dB")
    low_mask = torch.zeros_like(snr_db, dtype=torch.bool)
    for value in LOW_SNR_VALUES:
        low_mask |= snr_db == value
    return {
        "sample_ids": sample_ids,
        "predictions": predictions.numpy(),
        "targets": targets.numpy(),
        "modulation": targets.numpy(),
        "snr_db": snr_db.numpy(),
        "logits": logits.numpy(),
        "metrics": {
            "accuracy": replay_metrics.accuracy,
            "macro_f1": replay_metrics.macro_f1,
            "low_snr_accuracy": float((predictions[low_mask] == targets[low_mask]).double().mean()),
        },
    }


def _formal_runs(queue_root: Path) -> dict[str, dict[int, dict[str, Path]]]:
    result: dict[str, dict[int, dict[str, Path]]] = {model: {} for model in MODELS}
    for group_name in ("final-family-multiseed", "baseline-multiseed"):
        group_dir = queue_root / group_name
        summary = _load_json(group_dir / "multi-seed-summary.json", f"{group_name} summary")
        for run in summary.get("runs", []):
            if not isinstance(run, dict) or run.get("model") not in result:
                continue
            model = str(run["model"])
            seed = int(run["seed"])
            result[model][seed] = {
                "config": group_dir / "configs" / str(run["config_filename"]),
                "metrics": group_dir / str(run["run_id"]) / "metrics.json",
                "checkpoint": group_dir / str(run["run_id"]) / "best.pt",
            }
    for model, runs in result.items():
        if set(runs) != set(SEEDS):
            raise FinalBootstrapError(f"Formal replay inputs are incomplete for {model}")
    return result


def _save_prediction(path: Path, replay: dict[str, Any]) -> str:
    np.savez_compressed(
        path,
        sample_ids=np.asarray(replay["sample_ids"], dtype=str),
        predictions=np.asarray(replay["predictions"], dtype=np.int8),
        targets=np.asarray(replay["targets"], dtype=np.int8),
        modulation=np.asarray(replay["modulation"], dtype=np.int8),
        snr_db=np.asarray(replay["snr_db"], dtype=np.int8),
        logits=np.asarray(replay["logits"], dtype=np.float32),
    )
    return _sha256_file(path)


def _aligned(replays: dict[str, dict[int, dict[str, Any]]]) -> None:
    baseline = replays[MODELS[0]][SEEDS[0]]
    for model in MODELS:
        for seed in SEEDS:
            replay = replays[model][seed]
            for field in ("sample_ids", "targets", "modulation", "snr_db"):
                if not np.array_equal(np.asarray(baseline[field]), np.asarray(replay[field])):
                    raise FinalBootstrapError(f"Replay alignment differs: {model} seed {seed} {field}")


def _macro_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 11) -> float:
    confusion = np.bincount(
        targets.astype(np.int64) * num_classes + predictions.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = (
        2 * true_positive
        + confusion.sum(axis=0)
        - true_positive
        + confusion.sum(axis=1)
        - true_positive
    )
    scores = np.divide(
        2 * true_positive, denominator, out=np.zeros(num_classes), where=denominator > 0
    )
    return float(scores.mean())


def _seed_differences(
    newer: dict[int, dict[str, Any]], older: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        new = newer[seed]
        old = older[seed]
        targets = np.asarray(new["targets"], dtype=np.int64)
        snr_db = np.asarray(new["snr_db"], dtype=np.int64)
        new_predictions = np.asarray(new["predictions"], dtype=np.int64)
        old_predictions = np.asarray(old["predictions"], dtype=np.int64)
        low = np.isin(snr_db, LOW_SNR_VALUES)
        rows.append(
            {
                "seed": seed,
                "overall_accuracy_difference": float(
                    np.mean(
                        (new_predictions == targets).astype(np.int8)
                        - (old_predictions == targets).astype(np.int8)
                    )
                ),
                "low_snr_accuracy_difference": float(
                    np.mean(
                        (new_predictions[low] == targets[low]).astype(np.int8)
                        - (old_predictions[low] == targets[low]).astype(np.int8)
                    )
                ),
                "macro_f1_difference": _macro_f1(new_predictions, targets)
                - _macro_f1(old_predictions, targets),
            }
        )
    return rows


def _comparison(
    *,
    newer: dict[int, dict[str, Any]],
    older: dict[int, dict[str, Any]],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    accuracy_suite = _paired_accuracy_bootstrap_suite(
        reference_replays=newer,
        variant_replays=older,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    macro_f1 = paired_hierarchical_bootstrap(
        reference_replays=newer,
        variant_replays=older,
        metric="macro_f1",
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    seed_rows = _seed_differences(newer, older)
    positive_low = sum(row["low_snr_accuracy_difference"] > 0 for row in seed_rows)
    decision = formal_contribution_decision(
        low_snr_ci=accuracy_suite["low_snr_accuracy"],
        accuracy_ci=accuracy_suite["accuracy"],
        macro_f1_ci=macro_f1,
        positive_low_snr_seed_count=positive_low,
    )
    return {
        "overall_accuracy": accuracy_suite["accuracy"],
        "low_snr_accuracy": accuracy_suite["low_snr_accuracy"],
        "macro_f1": macro_f1,
        "per_seed_differences": seed_rows,
        "decision": decision,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--summary-report", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml",
    )
    parser.add_argument(
        "--conversion-contract",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml",
    )
    parser.add_argument(
        "--split-contract",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_root = args.queue_root.resolve(strict=True)
    summary_report_path = args.summary_report.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise FinalBootstrapError("Report output must remain outside the repository")
    summary_report = _load_json(summary_report_path, "five-seed summary")
    if (
        summary_report.get("test_accessed") is not False
        or summary_report.get("strongest_s0") != "lmscnet_s0_k15"
        or summary_report.get("strongest_current_baseline") != "resnet1d"
    ):
        raise FinalBootstrapError("Selection freeze differs from formal replay models")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FinalBootstrapError("CUDA requested but unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        prediction_dir = temporary / "predictions"
        prediction_dir.mkdir()
        inputs = _formal_runs(queue_root)
        replays: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in MODELS}
        prediction_manifest = []
        common = {
            "hdf5_path": args.hdf5,
            "conversion_manifest_path": args.conversion_manifest,
            "split_manifest_path": args.split_manifest,
            "leakage_audit_path": args.leakage_audit,
            "split_contract_path": args.split_contract,
            "dataset_spec_path": args.dataset_spec,
            "conversion_contract_path": args.conversion_contract,
            "preprocessing": "per_sample_max_abs",
        }
        with RadioML2016HDF5Dataset(split="validation", **common) as validation_dataset:
            for model in MODELS:
                for seed in SEEDS:
                    paths = inputs[model][seed]
                    config = load_experiment_config(paths["config"])
                    metrics = _load_json(paths["metrics"], f"{model} seed {seed} metrics")
                    if metrics.get("test_accessed") is not False:
                        raise FinalBootstrapError("Replay input accessed test")
                    checkpoint_sha256 = _sha256_file(paths["checkpoint"])
                    if metrics.get("artifacts", {}).get("checkpoint_sha256") != checkpoint_sha256:
                        raise FinalBootstrapError("Replay checkpoint hash differs")
                    replay = _replay(
                        config=config,
                        checkpoint_path=paths["checkpoint"],
                        metrics=metrics,
                        dataset=validation_dataset,
                        device=device,
                    )
                    replays[model][seed] = replay
                    prediction_path = prediction_dir / f"{model}-seed-{seed}.npz"
                    prediction_sha256 = _save_prediction(prediction_path, replay)
                    prediction_manifest.append(
                        {
                            "model": model,
                            "seed": seed,
                            "filename": prediction_path.name,
                            "prediction_sha256": prediction_sha256,
                            "checkpoint_sha256": checkpoint_sha256,
                            "sample_count": len(replay["targets"]),
                            **replay["metrics"],
                            "test_accessed": False,
                        }
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if validation_dataset.preprocessing != "per_sample_max_abs":
                raise FinalBootstrapError("Validation preprocessing differs")
            dataset_binding = {
                "split": "validation",
                "sample_count": len(validation_dataset),
                "assignment_sha256": validation_dataset.assignment_sha256,
                "split_manifest_sha256": validation_dataset.split_manifest_sha256,
                "preprocessing_mode": validation_dataset.preprocessing,
            }
        _aligned(replays)
        comparisons = {}
        per_seed_rows = []
        for name, newer_model, older_model in COMPARISONS:
            result = _comparison(
                newer=replays[newer_model],
                older=replays[older_model],
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            comparisons[name] = {
                "newer_model": newer_model,
                "older_model": older_model,
                **result,
            }
            per_seed_rows.extend(
                {"comparison": name, **row} for row in result["per_seed_differences"]
            )
        report = {
            "schema_version": 1,
            "purpose": "final_validation_paired_hierarchical_bootstrap",
            "test_accessed": False,
            "summary_report_sha256": _sha256_file(summary_report_path),
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "stratification": ["modulation", "snr_db"],
            "low_snr_values_db": list(LOW_SNR_VALUES),
            "dataset_binding": dataset_binding,
            "prediction_manifest": prediction_manifest,
            "comparisons": comparisons,
        }
        (temporary / "paired-bootstrap-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_csv(temporary / "paired-seed-differences.csv", per_seed_rows)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "prediction_count": len(prediction_manifest),
                "test_accessed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
