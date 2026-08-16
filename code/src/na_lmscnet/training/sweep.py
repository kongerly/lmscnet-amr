"""Reproducible validation-sweep orchestration for the RadioML baselines."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
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

MAX_SWEEP_CONTRACT_BYTES = 32 * 1024

SUPPORTED_SWEEP_MODELS = (
    "cnn2",
    "cldnn",
    "mcldnn",
    "mobilenetv2_1d",
    "na_lmscnet",
    "resnet1d",
    "resnet1d_macs",
    "se_msfn_1d",
)


class SweepError(ValueError):
    """Raised when the validation sweep is incomplete or inconsistent."""


@dataclass(frozen=True)
class SweepRunSpec:
    run_id: str
    learning_rate: float
    dropout: float
    config_filename: str
    output_directory: str


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SweepError(f"{field} must be a string-keyed mapping")
    return value


def _float_list(value: object, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise SweepError(f"{field} must be a non-empty list")
    if any(type(item) is not float or not 0.0 <= item < float("inf") for item in value):
        raise SweepError(f"{field} must contain finite non-negative YAML floats")
    if len(value) != len(set(value)):
        raise SweepError(f"{field} must not contain duplicates")
    return value


def _basename(value: object, field: str, suffix: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SweepError(f"{field} must be a non-empty trimmed string")
    if (
        PurePosixPath(value).name != value
        or PureWindowsPath(value).name != value
        or not value.endswith(suffix)
    ):
        raise SweepError(f"{field} must be a path-free {suffix} filename")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sweep_contract(path: Path) -> dict[str, Any]:
    """Load a fixed 2 x 2 validation-tuning contract for a supported baseline."""

    if path.is_symlink() or not path.is_file():
        raise SweepError("Sweep contract must be a regular file")
    if path.stat().st_size > MAX_SWEEP_CONTRACT_BYTES:
        raise SweepError(f"Sweep contract exceeds {MAX_SWEEP_CONTRACT_BYTES} bytes")
    with path.open(encoding="utf-8") as stream:
        contract = _mapping(yaml.safe_load(stream), "sweep contract")
    if set(contract) != {
        "schema_version",
        "sweep_id",
        "base_config",
        "seed",
        "grid",
        "selection",
        "test_access",
        "execution",
    }:
        raise SweepError("Sweep contract fields differ from schema version 1")
    allowed_sweep_ids = {
        f"{model}_radioml_2016_10a_validation_sweep_v1" for model in SUPPORTED_SWEEP_MODELS
    }
    if (
        contract["schema_version"] != 1
        or contract["sweep_id"] not in allowed_sweep_ids
        or contract["seed"] != 13
        or contract["test_access"] != "forbidden"
    ):
        raise SweepError("Sweep identity, seed, or test isolation changed")
    _basename(contract["base_config"], "base_config", ".yml")
    grid = _mapping(contract["grid"], "grid")
    if set(grid) != {"learning_rate", "dropout"}:
        raise SweepError("Sweep grid fields are invalid")
    learning_rates = _float_list(grid["learning_rate"], "grid.learning_rate")
    dropouts = _float_list(grid["dropout"], "grid.dropout")
    if learning_rates != [0.001, 0.0003] or dropouts != [0.0, 0.2]:
        raise SweepError("Sweep grid differs from the research plan")
    if contract["selection"] != {
        "primary": "validation_macro_f1_desc",
        "tie_break": ["validation_loss_asc", "run_id_asc"],
    }:
        raise SweepError("Sweep selection policy changed")
    if contract["execution"] != {
        "mode": "sequential",
        "resume_completed_runs": True,
        "output_outside_repository": True,
        "overwrite_completed_runs": False,
    }:
        raise SweepError("Sweep execution policy changed")
    return contract


def _float_id(value: float) -> str:
    return format(value, ".10g").replace(".", "p")


def sweep_run_specs(contract: dict[str, Any]) -> list[SweepRunSpec]:
    """Return the four runs in stable learning-rate/dropout order."""

    specs = []
    for learning_rate in contract["grid"]["learning_rate"]:
        for dropout in contract["grid"]["dropout"]:
            run_id = f"lr-{_float_id(learning_rate)}_dropout-{_float_id(dropout)}_seed-13"
            specs.append(
                SweepRunSpec(
                    run_id=run_id,
                    learning_rate=learning_rate,
                    dropout=dropout,
                    config_filename=f"{run_id}.yml",
                    output_directory=run_id,
                )
            )
    return specs


def _derived_config(base: ExperimentConfig, spec: SweepRunSpec) -> dict[str, object]:
    config = copy.deepcopy(asdict(base))
    model_name = str(base.model["name"])
    config["experiment_id"] = f"{model_name}_radioml_2016_10a_{spec.run_id}"
    config["purpose"] = "publication_candidate"
    config["model"]["dropout"] = spec.dropout
    config["optimizer"]["learning_rate"] = spec.learning_rate
    config["training"]["seed"] = 13
    config["training"]["max_train_batches"] = None
    config["training"]["max_validation_batches"] = None
    return config


def _config_bytes(config: dict[str, object]) -> bytes:
    return yaml.safe_dump(config, sort_keys=False).encode()


def _write_or_validate_config(path: Path, config: dict[str, object]) -> None:
    expected = _config_bytes(config)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected:
            raise SweepError(f"Existing derived config differs: {path.name}")
        return
    path.write_bytes(expected)


def _load_metrics(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SweepError("Completed run metrics must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SweepError(f"Could not read completed run metrics: {error}") from error
    return _mapping(value, "completed run metrics")


def _best_epoch_record(metrics: dict[str, Any]) -> dict[str, Any]:
    best_epoch = metrics.get("best_epoch")
    history = metrics.get("history")
    if (
        not isinstance(best_epoch, int)
        or not isinstance(history, list)
        or not 1 <= best_epoch <= len(history)
    ):
        raise SweepError("Run metrics do not contain a valid best epoch")
    return _mapping(history[best_epoch - 1], "best epoch record")


def _validate_completed_run(
    metrics: dict[str, Any],
    *,
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
        "seed": 13,
    }
    if data_protocol is not None:
        expected["data_protocol"] = data_protocol
    if bindings != expected or metrics.get("test_accessed") is not False:
        raise SweepError("Completed run bindings or test isolation differ from the sweep")
    _best_epoch_record(metrics)


def summarize_run(spec: SweepRunSpec, config_path: Path, run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    metrics = _load_metrics(metrics_path)
    best = _best_epoch_record(metrics)
    validation = _mapping(best.get("validation"), "best validation metrics")
    artifacts = _mapping(metrics.get("artifacts"), "run artifacts")
    if not checkpoint_path.is_file() or artifacts.get("checkpoint_sha256") != _sha256_file(
        checkpoint_path
    ):
        raise SweepError("Run checkpoint is missing or differs from metrics")
    return {
        "run_id": spec.run_id,
        "learning_rate": spec.learning_rate,
        "dropout": spec.dropout,
        "config_filename": spec.config_filename,
        "config_sha256": experiment_config_sha256(config_path),
        "output_directory": spec.output_directory,
        "metrics_sha256": _sha256_file(metrics_path),
        "checkpoint_sha256": artifacts["checkpoint_sha256"],
        "epochs_completed": metrics["epochs_completed"],
        "best_epoch": metrics["best_epoch"],
        "best_validation_macro_f1": metrics["best_validation_macro_f1"],
        "best_validation_accuracy": validation["accuracy"],
        "best_validation_loss": best["validation_loss"],
        "test_accessed": metrics["test_accessed"],
    }


def select_best_run(runs: list[dict[str, object]]) -> dict[str, object]:
    """Select by macro F1, then validation loss, then stable run ID."""

    if len(runs) != 4:
        raise SweepError("A complete sweep must contain exactly four runs")
    if any(run.get("test_accessed") is not False for run in runs):
        raise SweepError("Sweep selection must not use a run that accessed test")
    return min(
        runs,
        key=lambda run: (
            -float(run["best_validation_macro_f1"]),
            float(run["best_validation_loss"]),
            str(run["run_id"]),
        ),
    )


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


def run_validation_sweep(
    *,
    sweep_contract_path: Path,
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
) -> dict[str, object]:
    """Run or resume all four validation-only tuning runs for one baseline."""

    contract = load_sweep_contract(sweep_contract_path)
    base_config_path = sweep_contract_path.parent / contract["base_config"]
    base_config = load_experiment_config(base_config_path)
    if base_config.data["assignment_sha256"] != assignment_sha256:
        raise SweepError("Base config assignment differs from the loaded dataset")
    output_dir = output_dir.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    if not output_dir.is_dir() or output_dir == project_root or project_root in output_dir.parents:
        raise SweepError("Sweep output directory must exist outside the repository")
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    summaries = []
    for spec in sweep_run_specs(contract):
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
                config_path=config_path,
                split_manifest_sha256=split_manifest_sha256,
                assignment_sha256=assignment_sha256,
                project_commit=project_commit,
                data_protocol=data_protocol,
            )
            event = {"event": "run_resumed", "run_id": spec.run_id}
            if progress_callback is not None:
                progress_callback(event)
        else:
            resume = (run_dir / "last.pt").is_file()
            if any(run_dir.iterdir()) and not resume:
                raise SweepError(f"Incomplete run directory lacks last.pt: {spec.run_id}")

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

    best = select_best_run(summaries)
    summary = {
        "schema_version": 1,
        "sweep_id": contract["sweep_id"],
        "purpose": "validation_model_selection",
        "test_accessed": False,
        "bindings": {
            "sweep_contract_sha256": _sha256_file(sweep_contract_path),
            "base_config_sha256": experiment_config_sha256(base_config_path),
            "split_manifest_sha256": split_manifest_sha256,
            "assignment_sha256": assignment_sha256,
            "project_commit": project_commit,
            "seed": 13,
            **({"data_protocol": data_protocol} if data_protocol is not None else {}),
        },
        "selection": contract["selection"],
        "runs": summaries,
        "selected_run_id": best["run_id"],
        "selected_config_sha256": best["config_sha256"],
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
    }
    summary_path = output_dir / "sweep-summary.json"
    if summary_path.exists():
        existing = _load_metrics(summary_path)
        if existing != summary:
            raise SweepError("Existing sweep summary differs from completed runs")
    else:
        _atomic_json(summary, summary_path)
    return summary
