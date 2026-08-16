"""Replay the Phase R2 trained models on the frozen validation split.

Replays S1-static, S1-wide-static, SKNet-1D and AFNet adaptation checkpoints
on validation, verifies replay-vs-recorded metric equality, and writes
predictions in the shared npz schema for the primary contrast analysis.
Validation-only: test construction stays forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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
from na_lmscnet.models import build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.metrics import classification_metrics  # noqa: E402

SEEDS = (13, 37, 73, 101, 137)
R2_MODELS = ("lmscnet_s1_static", "lmscnet_s1_wide_static", "sknet_1d_adaptation", "afnet_adaptation")
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)


class R2ReplayError(RuntimeError):
    """Raised when an R2 replay violates the frozen protocol."""


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
        raise R2ReplayError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise R2ReplayError(f"{field} must contain a JSON object")
    return value


def _best_validation(metrics: dict[str, Any]) -> dict[str, Any]:
    best_epoch = metrics.get("best_epoch")
    history = metrics.get("history")
    if not isinstance(best_epoch, int) or not isinstance(history, list):
        raise R2ReplayError("Metrics lacks best epoch history")
    record = history[best_epoch - 1]
    if not isinstance(record, dict) or not isinstance(record.get("validation"), dict):
        raise R2ReplayError("Best validation record is invalid")
    return record["validation"]


def _replay_model(
    *,
    config: Any,
    checkpoint_path: Path,
    metrics: dict[str, Any],
    dataset: RadioML2016HDF5Dataset,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise R2ReplayError(f"Checkpoint is invalid: {checkpoint_path}")
    model = build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model.get("expansion", 1.25)),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
        permutation_seed=int(config.model.get("permutation_seed", 13)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    loader = DataLoader(
        dataset,
        batch_size=int(config.data["batch_size"]),
        shuffle=False,
        num_workers=int(config.data["num_workers"]),
        pin_memory=bool(config.data["pin_memory"]),
        persistent_workers=int(config.data["num_workers"]) > 0,
    )
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    snr_parts: list[torch.Tensor] = []
    sample_ids: list[str] = []
    amp = bool(config.training["amp"]) and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(batch["iq"].to(device, non_blocking=True))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            logits_parts.append(logits.detach().float().cpu())
            target_parts.append(batch["modulation"].to(dtype=torch.int64, device="cpu"))
            snr_parts.append(batch["snr"].to(dtype=torch.int64, device="cpu"))
            sample_ids.extend(str(value) for value in batch["sample_id"])
    if not logits_parts:
        raise R2ReplayError("Validation replay consumed zero samples")
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
            raise R2ReplayError(f"Replay differs from recorded {field}")
    for snr, accuracy in replay_metrics.per_snr_accuracy.items():
        if not math.isclose(
            float(accuracy),
            float(recorded["per_snr_accuracy"][snr]),
            rel_tol=3e-5,
            abs_tol=3e-5,
        ):
            raise R2ReplayError(f"Replay differs from recorded accuracy at {snr} dB")
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
            "accuracy": float(replay_metrics.accuracy),
            "macro_f1": float(replay_metrics.macro_f1),
            "low_snr_accuracy": float((predictions[low_mask] == targets[low_mask]).double().mean()),
        },
    }


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_root = args.queue_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise R2ReplayError("Output must remain outside the repository")
    summary = _load_json(queue_root / "multi-seed-summary.json", "multi-seed summary")
    if summary.get("test_accessed") is not False:
        raise R2ReplayError("Training queue accessed test")
    device = torch.device(args.device)
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
    with RadioML2016HDF5Dataset(split="validation", **common) as dataset:
        assignment_sha = dataset.assignment_sha256
        output_dir.mkdir(parents=True)
        prediction_manifest: list[dict[str, Any]] = []
        for model in R2_MODELS:
            for seed in SEEDS:
                run_id = f"{model}-seed-{seed}"
                config_path = queue_root / "configs" / f"{run_id}.yml"
                metrics_path = queue_root / run_id / "metrics.json"
                checkpoint_path = queue_root / run_id / "best.pt"
                if not (config_path.is_file() and metrics_path.is_file() and checkpoint_path.is_file()):
                    raise R2ReplayError(f"Incomplete run artifacts for {run_id}")
                config = load_experiment_config(config_path)
                metrics = _load_json(metrics_path, f"{run_id} metrics")
                replay = _replay_model(
                    config=config,
                    checkpoint_path=checkpoint_path,
                    metrics=metrics,
                    dataset=dataset,
                    device=device,
                )
                pred_path = output_dir / f"{model}-seed-{seed}.npz"
                prediction_manifest.append(
                    {
                        "model": model,
                        "seed": seed,
                        "filename": pred_path.name,
                        "prediction_sha256": _save_prediction(pred_path, replay),
                        "checkpoint_sha256": _sha256_file(checkpoint_path),
                        "config_sha256": _sha256_file(config_path),
                        "sample_count": len(replay["targets"]),
                        **replay["metrics"],
                        "test_accessed": False,
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        report = {
            "schema_version": 1,
            "purpose": "phase_r2_validation_prediction_replay",
            "test_accessed": False,
            "assignment_sha256": assignment_sha,
            "split_manifest_sha256": _sha256_file(args.split_manifest),
            "preprocessing_mode": dataset.preprocessing,
            "prediction_manifest": prediction_manifest,
        }
        (output_dir / "replay-manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
