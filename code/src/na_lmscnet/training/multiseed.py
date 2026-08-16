"""Reproducible multi-seed baseline training orchestration."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset

from na_lmscnet.data.contracts import ModulationSample
from na_lmscnet.training.engine import (
    ExperimentConfig,
    experiment_config_sha256,
    load_experiment_config,
    run_training,
)

SEEDS = (13, 37, 73, 101, 137)
MAX_MULTI_SEED_CONFIG_BYTES = 32 * 1024


class MultiSeedError(ValueError):
    """Raised when multi-seed training is incomplete or inconsistent."""


@dataclass(frozen=True)
class MultiSeedRunSpec:
    model: str
    seed: int
    run_id: str
    config_filename: str
    output_directory: str


def _validate_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise MultiSeedError("seeds must be non-empty, unique, non-negative integers")
    return seeds


def multi_seed_run_specs(
    models: tuple[str, ...], seeds: tuple[int, ...] = SEEDS
) -> list[MultiSeedRunSpec]:
    """Return one spec per model and seed in stable model-then-seed order."""

    seeds = _validate_seeds(seeds)
    specs = []
    for model in models:
        for seed in seeds:
            run_id = f"{model}-seed-{seed}"
            specs.append(
                MultiSeedRunSpec(
                    model=model,
                    seed=seed,
                    run_id=run_id,
                    config_filename=f"{run_id}.yml",
                    output_directory=run_id,
                )
            )
    return specs


def _derived_config(base: ExperimentConfig, spec: MultiSeedRunSpec) -> dict[str, object]:
    config = copy.deepcopy(asdict(base))
    config["experiment_id"] = f"{spec.model}_{base.data['dataset_id']}_seed{spec.seed}"
    config["training"]["seed"] = spec.seed
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MultiSeedError(f"{field} must be a string-keyed mapping")
    return value


def _config_bytes(config: dict[str, object]) -> bytes:
    return yaml.safe_dump(config, sort_keys=False).encode()


def _write_or_validate_config(path: Path, config: dict[str, object]) -> None:
    expected = _config_bytes(config)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected:
            raise MultiSeedError(f"Existing derived config differs: {path.name}")
        return
    path.write_bytes(expected)


def _load_metrics(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MultiSeedError("Completed run metrics must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MultiSeedError(f"Could not read completed run metrics: {error}") from error
    return _mapping(value, "completed run metrics")


def _best_epoch_record(metrics: dict[str, Any]) -> dict[str, Any]:
    best_epoch = metrics.get("best_epoch")
    history = metrics.get("history")
    if (
        not isinstance(best_epoch, int)
        or not isinstance(history, list)
        or not 1 <= best_epoch <= len(history)
    ):
        raise MultiSeedError("Run metrics do not contain a valid best epoch")
    return _mapping(history[best_epoch - 1], "best epoch record")


def _validate_completed_run(
    metrics: dict[str, Any],
    *,
    spec: MultiSeedRunSpec,
    config_path: Path,
    split_manifest_sha256: str,
    assignment_sha256: str,
    project_commit: str,
    data_protocol: dict[str, object] | None = None,
) -> None:
    bindings = _mapping(metrics.get("bindings"), "run bindings")
    expected = {
        "experiment_config_sha256": experiment_config_sha256(config_path),
        "split_manifest_sha256": split_manifest_sha256,
        "assignment_sha256": assignment_sha256,
        "project_commit": project_commit,
        "seed": spec.seed,
    }
    if data_protocol is not None:
        expected["data_protocol"] = data_protocol
    if bindings != expected or metrics.get("test_accessed") is not False:
        raise MultiSeedError("Completed run bindings or test isolation differ")
    _best_epoch_record(metrics)


def summarize_run(spec: MultiSeedRunSpec, config_path: Path, run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    metrics = _load_metrics(metrics_path)
    best = _best_epoch_record(metrics)
    validation = _mapping(best.get("validation"), "best validation metrics")
    artifacts = _mapping(metrics.get("artifacts"), "run artifacts")
    if not checkpoint_path.is_file() or artifacts.get("checkpoint_sha256") != _sha256_file(
        checkpoint_path
    ):
        raise MultiSeedError("Run checkpoint is missing or differs from metrics")
    return {
        "run_id": spec.run_id,
        "model": spec.model,
        "seed": spec.seed,
        "config_filename": spec.config_filename,
        "config_sha256": experiment_config_sha256(config_path),
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "epochs_completed": metrics["epochs_completed"],
        "best_epoch": metrics["best_epoch"],
        "best_validation_macro_f1": metrics["best_validation_macro_f1"],
        "best_validation_accuracy": validation["accuracy"],
        "best_validation_loss": best["validation_loss"],
        "test_accessed": metrics["test_accessed"],
    }


def _atomic_json(value: object, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_multi_seed(
    *,
    base_config_paths: list[Path],
    train_dataset: Dataset[ModulationSample],
    validation_dataset: Dataset[ModulationSample],
    output_dir: Path,
    project_root: Path,
    project_commit: str,
    split_manifest_sha256: str,
    assignment_sha256: str,
    device: torch.device,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    batch_callback: Callable[[dict[str, object]], None] | None = None,
    data_protocol: dict[str, object] | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, object]:
    """Run or resume explicit seeds per model on frozen train and validation splits."""

    seeds = _validate_seeds(seeds)
    if len(base_config_paths) == 0 or len(base_config_paths) != len(
        {path.resolve() for path in base_config_paths}
    ):
        raise MultiSeedError("base_config_paths must be non-empty and unique")
    base_configs = [(path, load_experiment_config(path)) for path in base_config_paths]
    for _, config in base_configs:
        if config.data["assignment_sha256"] != assignment_sha256:
            raise MultiSeedError("Base config assignment differs from the loaded dataset")
        if config.test_access != "forbidden":
            raise MultiSeedError("Base config must forbid test access")
    output_dir = output_dir.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    if not output_dir.is_dir() or output_dir == project_root or project_root in output_dir.parents:
        raise MultiSeedError("Output directory must exist outside the repository")
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    summaries = []
    for _, base_config in base_configs:
        model = str(base_config.model["name"])
        for spec in multi_seed_run_specs((model,), seeds):
            config_path = configs_dir / spec.config_filename
            _write_or_validate_config(config_path, _derived_config(base_config, spec))
            config = load_experiment_config(config_path)
            run_dir = output_dir / spec.output_directory
            run_dir.mkdir(exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                metrics = _load_metrics(metrics_path)
                _validate_completed_run(
                    metrics,
                    spec=spec,
                    config_path=config_path,
                    split_manifest_sha256=split_manifest_sha256,
                    assignment_sha256=assignment_sha256,
                    project_commit=project_commit,
                    data_protocol=data_protocol,
                )
                if progress_callback is not None:
                    progress_callback({"event": "run_resumed", "run_id": spec.run_id})
            else:
                resume = (run_dir / "last.pt").is_file()
                if any(run_dir.iterdir()) and not resume:
                    raise MultiSeedError(f"Incomplete run directory lacks last.pt: {spec.run_id}")

                def epoch_callback(record: dict[str, object], run_id: str = spec.run_id) -> None:
                    if progress_callback is not None:
                        progress_callback({"event": "epoch_complete", "run_id": run_id, **record})

                run_training(
                    config=config,
                    config_path=config_path,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    output_dir=run_dir,
                    project_root=project_root,
                    project_commit=project_commit,
                    split_manifest_sha256=split_manifest_sha256,
                    device=device,
                    epoch_callback=epoch_callback,
                    batch_callback=batch_callback,
                    resume=resume,
                    data_protocol=data_protocol,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            summaries.append(summarize_run(spec, config_path, run_dir))

    summary = {
        "schema_version": 1,
        "purpose": "baseline_multi_seed_training",
        "test_accessed": False,
        "bindings": {
            "split_manifest_sha256": split_manifest_sha256,
            "assignment_sha256": assignment_sha256,
            "project_commit": project_commit,
            "seeds": list(seeds),
            **({"data_protocol": data_protocol} if data_protocol is not None else {}),
        },
        "runs": summaries,
        "run_count": len(summaries),
    }
    summary_path = output_dir / "multi-seed-summary.json"
    if summary_path.exists():
        existing = _load_metrics(summary_path)
        if existing != summary:
            raise MultiSeedError("Existing multi-seed summary differs from completed runs")
    else:
        _atomic_json(summary, summary_path)
    return summary
