"""Consume the one-shot authorization and run the frozen RadioML 2016.10A test evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
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

from na_lmscnet.data import RadioML2016FrozenTestDataset  # noqa: E402
from na_lmscnet.evaluation import (  # noqa: E402
    audit_freeze_manifest,
    authorize_frozen_test_dataset,
    consume_test_authorization,
    sha256_file,
    update_consumption_marker,
)
from na_lmscnet.evaluation.core_ablation_multiseed_report import (  # noqa: E402
    _paired_accuracy_bootstrap_suite,
    paired_hierarchical_bootstrap,
)
from na_lmscnet.models import build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.metrics import classification_metrics  # noqa: E402


class FrozenTestError(ValueError):
    """Raised when the frozen test-only evaluation cannot be completed exactly once."""


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenTestError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise FrozenTestError(f"{field} must contain a JSON object")
    return value


def _model(config: Any) -> torch.nn.Module:
    return build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model.get("expansion", 1.25)),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
    )


def _macro_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 11) -> float:
    flat = targets.astype(np.int64) * num_classes + predictions.astype(np.int64)
    confusion = np.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = (
        2 * true_positive
        + confusion.sum(axis=0)
        - true_positive
        + confusion.sum(axis=1)
        - true_positive
    )
    return float(
        np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros(num_classes, dtype=np.float64),
            where=denominator > 0,
        ).mean()
    )


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


@torch.inference_mode()
def _replay(
    *,
    config: Any,
    checkpoint_path: Path,
    dataset: RadioML2016FrozenTestDataset,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = _model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
        raise FrozenTestError("Test replay consumed zero samples")
    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    snr_db = torch.cat(snr_parts)
    predictions = logits.argmax(dim=1)
    metrics = classification_metrics(predictions, targets, snr_db, num_classes=11)
    probabilities = torch.softmax(logits, dim=1).numpy().astype(np.float64)
    targets_np = targets.numpy()
    predictions_np = predictions.numpy()
    snr_np = snr_db.numpy()
    correct = predictions_np == targets_np
    low = np.isin(snr_np, (-10, -8, -6, -4, -2, 0))
    one_hot = np.eye(11, dtype=np.float64)[targets_np]
    return {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "predictions": predictions_np.astype(np.int8),
        "targets": targets_np.astype(np.int8),
        "modulation": targets_np.astype(np.int8),
        "snr_db": snr_np.astype(np.int8),
        "logits": logits.numpy().astype(np.float32),
        "metrics": {
            "overall_accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "low_snr_accuracy": float(correct[low].mean()),
            "per_snr_accuracy": metrics.per_snr_accuracy,
            "ece_15_bin": _ece(probabilities.max(axis=1), correct.astype(np.float64)),
            "nll": float(
                -np.log(
                    np.maximum(probabilities[np.arange(len(targets_np)), targets_np], 1e-12)
                ).mean()
            ),
            "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "sample_count": len(targets_np),
        },
    }


def _save_prediction(path: Path, replay: dict[str, Any]) -> dict[str, Any]:
    np.savez_compressed(
        path,
        sample_ids=replay["sample_ids"],
        predictions=replay["predictions"],
        targets=replay["targets"],
        modulation=replay["modulation"],
        snr_db=replay["snr_db"],
        logits=replay["logits"],
    )
    return {"filename": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _aligned(
    replays: dict[str, dict[int, dict[str, Any]]], models: tuple[str, ...], seeds: tuple[int, ...]
) -> None:
    reference = replays[models[0]][seeds[0]]
    for model in models:
        for seed in seeds:
            replay = replays[model][seed]
            for field in ("sample_ids", "targets", "modulation", "snr_db"):
                if not np.array_equal(reference[field], replay[field]):
                    raise FrozenTestError(
                        f"Test replay alignment differs: {model} seed {seed} {field}"
                    )


def _seed_differences(
    newer: dict[int, dict[str, Any]], older: dict[int, dict[str, Any]], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        targets = newer[seed]["targets"].astype(np.int64)
        snr = newer[seed]["snr_db"].astype(np.int64)
        new_predictions = newer[seed]["predictions"].astype(np.int64)
        old_predictions = older[seed]["predictions"].astype(np.int64)
        low = np.isin(snr, (-10, -8, -6, -4, -2, 0))
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


def _aggregate(rows: list[dict[str, Any]], models: tuple[str, ...]) -> list[dict[str, Any]]:
    metrics = (
        "overall_accuracy",
        "macro_f1",
        "low_snr_accuracy",
        "ece_15_bin",
        "nll",
        "brier_score",
    )
    result = []
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        result.append(
            {
                "model": model,
                "run_count": len(selected),
                **{
                    f"{metric}_mean": statistics.fmean(float(row[metric]) for row in selected)
                    for metric in metrics
                },
                **{
                    f"{metric}_sample_std": statistics.stdev(float(row[metric]) for row in selected)
                    for metric in metrics
                },
            }
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
    manifest_path = args.manifest.resolve(strict=True)
    audit_freeze_manifest(manifest_path, project_root=PROJECT_ROOT, require_unconsumed=True)
    consume_test_authorization(manifest_path)
    update_consumption_marker(manifest_path, status="running")
    manifest = _load_json(manifest_path, "experiment freeze manifest")
    output_dir = Path(manifest["test_authorization"]["output_dir"])
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    try:
        authorization = authorize_frozen_test_dataset(manifest_path)
        dataset_binding = manifest["dataset"]
        with RadioML2016FrozenTestDataset(
            authorization=authorization,
            hdf5_path=Path(dataset_binding["hdf5"]["path"]),
            conversion_manifest_path=Path(dataset_binding["conversion_manifest"]["path"]),
            split_manifest_path=Path(dataset_binding["split_manifest"]["path"]),
            leakage_audit_path=Path(dataset_binding["leakage_audit"]["path"]),
            split_contract_path=args.split_contract,
            dataset_spec_path=args.dataset_spec,
            conversion_contract_path=args.conversion_contract,
            preprocessing=dataset_binding["preprocessing_mode"],
        ) as dataset:
            update_consumption_marker(
                manifest_path, test_dataset_constructed=True, test_sample_count=len(dataset)
            )
            device = torch.device(args.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise FrozenTestError("CUDA was requested but is unavailable")
            models = tuple(manifest["test_protocol"]["models"])
            seeds = tuple(int(seed) for seed in manifest["test_protocol"]["seeds"])
            run_bindings = {
                (run["model"], int(run["seed"])): run for run in manifest["selection"]["runs"]
            }
            replays: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in models}
            rows: list[dict[str, Any]] = []
            predictions_dir = temporary / "predictions"
            predictions_dir.mkdir()
            prediction_manifest = []
            for model in models:
                for seed in seeds:
                    binding = run_bindings[(model, seed)]
                    config = load_experiment_config(Path(binding["config"]["path"]))
                    replay = _replay(
                        config=config,
                        checkpoint_path=Path(binding["checkpoint"]["path"]),
                        dataset=dataset,
                        device=device,
                    )
                    replays[model][seed] = replay
                    prediction = _save_prediction(
                        predictions_dir / f"{model}-seed-{seed}.npz", replay
                    )
                    prediction_manifest.append({"model": model, "seed": seed, **prediction})
                    rows.append({"model": model, "seed": seed, **replay["metrics"]})
        _aligned(replays, models, seeds)
        bootstrap_seed = int(manifest["test_protocol"]["bootstrap_seed"])
        bootstrap_resamples = int(manifest["test_protocol"]["bootstrap_resamples"])
        comparison = {
            "overall_and_per_snr_accuracy": _paired_accuracy_bootstrap_suite(
                reference_replays=replays["lmscnet_s2"],
                variant_replays=replays["se_msfn_1d"],
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
            "macro_f1": paired_hierarchical_bootstrap(
                reference_replays=replays["lmscnet_s2"],
                variant_replays=replays["se_msfn_1d"],
                metric="macro_f1",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
            "per_seed_differences": _seed_differences(
                replays["lmscnet_s2"], replays["se_msfn_1d"], seeds
            ),
        }
        report = {
            "schema_version": 1,
            "purpose": "single_frozen_test_only_evaluation",
            "test_accessed": True,
            "test_access_count": 1,
            "freeze_manifest_sha256": sha256_file(manifest_path),
            "implementation_commit": manifest["implementation_commit"],
            "dataset": {
                "dataset_id": "radioml_2016_10a",
                "split": "test",
                "sample_count": rows[0]["sample_count"],
                "assignment_sha256": dataset_binding["assignment_sha256"],
                "split_manifest_sha256": dataset_binding["split_manifest"]["sha256"],
                "preprocessing_mode": dataset_binding["preprocessing_mode"],
            },
            "protocol": manifest["test_protocol"],
            "runs": rows,
            "model_summary": _aggregate(rows, models),
            "comparison": comparison,
            "prediction_manifest": prediction_manifest,
            "post_test_policy": {
                "tuning": "forbidden",
                "model_loss_preprocessing_split_metric_changes": "forbidden",
                "design_revision_from_test": "forbidden",
            },
        }
        report_path = temporary / "test-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_manifest = {
            "schema_version": 1,
            "purpose": "frozen_test_result_artifact_manifest",
            "freeze_manifest_sha256": sha256_file(manifest_path),
            "files": [
                {
                    "path": str(path.relative_to(temporary)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            ],
        }
        artifact_path = temporary / "result-manifest.json"
        artifact_path.write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
        update_consumption_marker(
            manifest_path,
            status="complete",
            result_report_sha256=sha256_file(output_dir / "test-report.json"),
            result_manifest_sha256=sha256_file(output_dir / "result-manifest.json"),
        )
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "test_access_count": 1,
                    "run_count": len(rows),
                    "status": "complete",
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        update_consumption_marker(
            manifest_path,
            status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
