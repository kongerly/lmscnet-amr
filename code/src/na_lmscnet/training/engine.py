"""Configuration-driven train/validation engine with external artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from na_lmscnet.data.contracts import ModulationSample
from na_lmscnet.models import build_model
from na_lmscnet.training.losses import NoiseAwareJointLoss
from na_lmscnet.training.metrics import ClassificationMetrics, classification_metrics

MAX_EXPERIMENT_CONFIG_BYTES = 64 * 1024
SNR_AUXILIARY_MODEL_NAMES = {
    "na_lmscnet",
    "na_lmscnet_fixed_average",
    "na_lmscnet_wo_multi_scale",
}
DICTIONARY_CLASSIFIER_MODEL_NAMES = {
    "na_lmscnet_wo_snr_auxiliary",
    "lmscnet_s0_k3",
    "lmscnet_s0_k7",
    "lmscnet_s0_k15",
    "lmscnet_s0_wide",
    "lmscnet_s1",
    "lmscnet_s2",
    "lmscnet_s1_static",
    "lmscnet_s1_wide_static",
    "lmscnet_s2_mean",
    "lmscnet_s2_shuffled",
    "sknet_1d_adaptation",
    "afnet_adaptation",
}
SUPPORTED_DATASET_IDS = {"radioml_2016_10a", "radioml_2018_01a"}


class TrainingError(ValueError):
    """Raised when training configuration or artifact publication is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    purpose: Literal["publication_candidate", "infrastructure_smoke_only"]
    model: dict[str, object]
    data: dict[str, object]
    optimizer: dict[str, object]
    scheduler: dict[str, object]
    training: dict[str, object]
    selection_metric: str
    test_access: str
    artifacts: dict[str, object]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TrainingError(f"{field} must be a string-keyed mapping")
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingError(f"{field} must be a non-negative integer")
    return value


def _float(value: object, field: str, *, minimum: float = 0.0) -> float:
    if type(value) is not float or not math.isfinite(value) or value < minimum:
        raise TrainingError(f"{field} must be a finite float of at least {minimum}")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingError(f"{field} must be a boolean")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_batch_limit(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, field)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate a CNN2 training configuration."""

    if path.is_symlink() or not path.is_file():
        raise TrainingError("Experiment config must be a regular file")
    if path.stat().st_size > MAX_EXPERIMENT_CONFIG_BYTES:
        raise TrainingError(f"Experiment config exceeds {MAX_EXPERIMENT_CONFIG_BYTES} bytes")
    with path.open(encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "experiment config")
    expected = {
        "schema_version",
        "experiment_id",
        "purpose",
        "model",
        "data",
        "optimizer",
        "scheduler",
        "training",
        "selection_metric",
        "test_access",
        "artifacts",
    }
    if set(raw) != expected or raw["schema_version"] != 1:
        raise TrainingError("Experiment config fields or schema version are invalid")
    if (
        not isinstance(raw["experiment_id"], str)
        or not raw["experiment_id"]
        or raw["purpose"] not in {"publication_candidate", "infrastructure_smoke_only"}
    ):
        raise TrainingError("Experiment identity or purpose is invalid")

    model = _mapping(raw["model"], "model")
    if not isinstance(model.get("name"), str):
        raise TrainingError("model.name must be a supported model name")
    if model["name"] in {
        "cnn2",
        "cldnn",
        "mcldnn",
        "mobilenetv2_1d",
        "na_lmscnet_wo_snr_auxiliary",
        "resnet1d",
        "resnet1d_macs",
        "se_msfn_1d",
    }:
        if set(model) != {"name", "num_classes", "dropout"}:
            raise TrainingError("Baseline model config fields are invalid")
    elif model["name"] in {
        "lmscnet_s0_k3",
        "lmscnet_s0_k7",
        "lmscnet_s0_k15",
        "lmscnet_s0_wide",
    }:
        if set(model) != {"name", "num_classes", "dropout", "kernel", "expansion"}:
            raise TrainingError("S0 model config fields are invalid")
        if model["kernel"] not in {3, 7, 15}:
            raise TrainingError("S0 kernel must be one of 3, 7, or 15")
        expected_kernel = {
            "lmscnet_s0_k3": 3,
            "lmscnet_s0_k7": 7,
            "lmscnet_s0_k15": 15,
            "lmscnet_s0_wide": 7,
        }[str(model["name"])]
        if int(model["kernel"]) != expected_kernel:
            raise TrainingError(f"{model['name']} is frozen to kernel {expected_kernel}")
        expansion = _float(model["expansion"], "model.expansion", minimum=1.0)
        if expansion > 3.0:
            raise TrainingError("model.expansion must be at most three")
        expected_expansion = 1.42 if model["name"] == "lmscnet_s0_wide" else 1.25
        if expansion != expected_expansion:
            raise TrainingError(
                f"{model['name']} expansion must remain frozen at {expected_expansion}"
            )
    elif model["name"] in {
        "lmscnet_s1",
        "lmscnet_s2",
        "lmscnet_s1_static",
        "lmscnet_s1_wide_static",
        "lmscnet_s2_mean",
        "lmscnet_s2_shuffled",
        "sknet_1d_adaptation",
        "afnet_adaptation",
    }:
        expected_fields = {"name", "num_classes", "dropout", "expansion"}
        if model["name"] == "lmscnet_s2_shuffled":
            expected_fields.add("permutation_seed")
        if set(model) != expected_fields:
            raise TrainingError("Final multi-scale model config fields are invalid")
        expansion = _float(model["expansion"], "model.expansion", minimum=1.0)
        if expansion > 3.0:
            raise TrainingError("model.expansion must be at most three")
        expected_expansion = 1.8 if model["name"] == "lmscnet_s1_wide_static" else 1.25
        if expansion != expected_expansion:
            raise TrainingError(
                f"{model['name']} expansion must remain frozen at {expected_expansion}"
            )
        if model["name"] == "lmscnet_s2_shuffled":
            _nonnegative_integer(model["permutation_seed"], "model.permutation_seed")
    elif model["name"] in SNR_AUXILIARY_MODEL_NAMES:
        if set(model) != {"name", "num_classes", "dropout", "snr_loss_weight"}:
            raise TrainingError("NA-LMSCNet model config fields are invalid")
        _float(model["snr_loss_weight"], "model.snr_loss_weight")
        if float(model["snr_loss_weight"]) > 1.0:
            raise TrainingError("model.snr_loss_weight must be at most one")
    else:
        raise TrainingError("Model config must select a supported model")
    _positive_integer(model["num_classes"], "model.num_classes")
    dropout = _float(model["dropout"], "model.dropout")
    if dropout >= 1.0:
        raise TrainingError("model.dropout must be below one")

    data = _mapping(raw["data"], "data")
    if (
        set(data)
        != {
            "dataset_id",
            "assignment_sha256",
            "batch_size",
            "num_workers",
            "pin_memory",
            "train_augmentation",
        }
        or data["dataset_id"] not in SUPPORTED_DATASET_IDS
    ):
        raise TrainingError("Data configuration is incomplete or uses an unsupported dataset")
    _sha256(data["assignment_sha256"], "data.assignment_sha256")
    _positive_integer(data["batch_size"], "data.batch_size")
    _nonnegative_integer(data["num_workers"], "data.num_workers")
    _boolean(data["pin_memory"], "data.pin_memory")
    augmentation = _mapping(data["train_augmentation"], "data.train_augmentation")
    if set(augmentation) != {"random_phase_rotation", "random_circular_shift"}:
        raise TrainingError("Training augmentation fields are incomplete")
    for field in augmentation:
        _boolean(augmentation[field], f"data.train_augmentation.{field}")

    optimizer = _mapping(raw["optimizer"], "optimizer")
    if set(optimizer) != {"name", "learning_rate", "weight_decay"} or optimizer["name"] != "adamw":
        raise TrainingError("Optimizer must be AdamW")
    _float(optimizer["learning_rate"], "optimizer.learning_rate", minimum=1e-12)
    _float(optimizer["weight_decay"], "optimizer.weight_decay")
    scheduler = _mapping(raw["scheduler"], "scheduler")
    if scheduler != {"name": "cosine_annealing"}:
        raise TrainingError("Scheduler must be cosine annealing")

    training = _mapping(raw["training"], "training")
    required_training_fields = {
        "seed",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "amp",
        "deterministic",
        "max_train_batches",
        "max_validation_batches",
    }
    selection_metric = raw["selection_metric"]
    expected_training_fields = set(required_training_fields)
    if selection_metric == "fixed_epoch":
        expected_training_fields.add("checkpoint_epoch")
    if set(training) != expected_training_fields:
        raise TrainingError("Training configuration fields are incomplete")
    _nonnegative_integer(training["seed"], "training.seed")
    _positive_integer(training["max_epochs"], "training.max_epochs")
    _positive_integer(training["early_stopping_patience"], "training.early_stopping_patience")
    _float(training["early_stopping_min_delta"], "training.early_stopping_min_delta")
    _boolean(training["amp"], "training.amp")
    _boolean(training["deterministic"], "training.deterministic")
    _optional_batch_limit(training["max_train_batches"], "training.max_train_batches")
    _optional_batch_limit(training["max_validation_batches"], "training.max_validation_batches")

    if selection_metric not in {"validation_macro_f1", "fixed_epoch"}:
        raise TrainingError("Selection metric is invalid")
    if selection_metric == "fixed_epoch":
        checkpoint_epoch = _positive_integer(
            training["checkpoint_epoch"], "training.checkpoint_epoch"
        )
        if checkpoint_epoch != training["max_epochs"]:
            raise TrainingError("fixed_epoch checkpoint must equal max_epochs")
        if training["early_stopping_patience"] != training["max_epochs"]:
            raise TrainingError("fixed_epoch training must disable validation early stopping")
        if training["early_stopping_min_delta"] != 0.0:
            raise TrainingError("fixed_epoch training must not use a validation improvement delta")
    if raw["test_access"] != "forbidden":
        raise TrainingError("Selection metric or test isolation policy is invalid")
    artifacts = _mapping(raw["artifacts"], "artifacts")
    if artifacts != {"output_outside_repository": True, "overwrite": False}:
        raise TrainingError("Training artifacts must remain external and no-overwrite")
    return ExperimentConfig(**raw)


def experiment_config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def augment_iq_batch(
    iq: torch.Tensor,
    *,
    generator: torch.Generator,
    phase_rotation: bool,
    circular_shift: bool,
) -> torch.Tensor:
    """Apply deterministic batch-wise random phase and circular shift augmentation."""

    if iq.ndim != 3 or iq.shape[1] != 2:
        raise TrainingError("Augmentation expects [batch, 2, length]")
    result = iq
    batch_size, _, length = iq.shape
    if phase_rotation:
        phase = torch.rand(batch_size, generator=generator, dtype=iq.dtype) * (2.0 * math.pi)
        cosine = phase.cos().view(-1, 1)
        sine = phase.sin().view(-1, 1)
        i_channel = result[:, 0]
        q_channel = result[:, 1]
        result = torch.stack(
            [i_channel * cosine - q_channel * sine, i_channel * sine + q_channel * cosine],
            dim=1,
        )
    if circular_shift:
        shifts = torch.randint(length, (batch_size,), generator=generator)
        indices = (torch.arange(length).view(1, -1) - shifts.view(-1, 1)) % length
        result = result.gather(2, indices.view(batch_size, 1, length).expand(-1, 2, -1))
    return result


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _data_loader(
    dataset: Dataset[ModulationSample],
    config: ExperimentConfig,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader[ModulationSample]:
    workers = int(config.data["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config.data["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config.data["pin_memory"]),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _batch_limit_reached(batch_index: int, limit: int | None) -> bool:
    return limit is not None and batch_index >= limit


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[ModulationSample],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    augmentation_generator: torch.Generator,
    config: ExperimentConfig,
    epoch: int,
    batch_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    augmentation = config.data["train_augmentation"]
    limit = config.training["max_train_batches"]
    total_batches = len(loader) if limit is None else int(limit)
    for batch_index, batch in enumerate(loader):
        if _batch_limit_reached(batch_index, limit):
            break
        iq = augment_iq_batch(
            batch["iq"],
            generator=augmentation_generator,
            phase_rotation=bool(augmentation["random_phase_rotation"]),
            circular_shift=bool(augmentation["random_circular_shift"]),
        ).to(device, non_blocking=True)
        targets = batch["modulation"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(iq)
            if str(config.model["name"]) in SNR_AUXILIARY_MODEL_NAMES:
                loss, _, _ = NoiseAwareJointLoss(snr_weight=float(config.model["snr_loss_weight"]))(
                    outputs, targets, batch["snr"].to(device, non_blocking=True)
                )
            elif str(config.model["name"]) in DICTIONARY_CLASSIFIER_MODEL_NAMES:
                loss = nn.functional.cross_entropy(outputs["logits"], targets)
            else:
                loss = nn.functional.cross_entropy(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = len(targets)
        loss_sum += float(loss.detach()) * count
        sample_count += count
        if batch_callback is not None:
            batch_callback(
                {
                    "event": "batch_complete",
                    "epoch": epoch,
                    "batch": batch_index + 1,
                    "total_batches": total_batches,
                    "max_epochs": int(config.training["max_epochs"]),
                    "train_loss": loss_sum / sample_count,
                }
            )
    if sample_count == 0:
        raise TrainingError("Training epoch consumed zero samples")
    return loss_sum / sample_count, sample_count


@torch.inference_mode()
def _validate_epoch(
    model: nn.Module,
    loader: DataLoader[ModulationSample],
    device: torch.device,
    config: ExperimentConfig,
) -> tuple[float, ClassificationMetrics]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    predictions: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    snr_all: list[torch.Tensor] = []
    snr_predictions: list[torch.Tensor] = []
    limit = config.training["max_validation_batches"]
    amp_enabled = bool(config.training["amp"]) and device.type == "cuda"
    for batch_index, batch in enumerate(loader):
        if _batch_limit_reached(batch_index, limit):
            break
        iq = batch["iq"].to(device, non_blocking=True)
        targets = batch["modulation"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(iq)
            if str(config.model["name"]) in SNR_AUXILIARY_MODEL_NAMES:
                loss, _, _ = NoiseAwareJointLoss(snr_weight=float(config.model["snr_loss_weight"]))(
                    outputs, targets, batch["snr"].to(device, non_blocking=True)
                )
                logits = outputs["logits"]
                snr_predictions.append(outputs["snr_hat"].detach().cpu())
            elif str(config.model["name"]) in DICTIONARY_CLASSIFIER_MODEL_NAMES:
                logits = outputs["logits"]
                loss = nn.functional.cross_entropy(logits, targets)
            else:
                logits = outputs
                loss = nn.functional.cross_entropy(logits, targets)
        count = len(targets)
        loss_sum += float(loss) * count
        sample_count += count
        predictions.append(logits.argmax(dim=1).cpu())
        targets_all.append(targets.cpu())
        snr_all.append(batch["snr"].cpu())
    if sample_count == 0:
        raise TrainingError("Validation epoch consumed zero samples")
    metrics = classification_metrics(
        torch.cat(predictions),
        torch.cat(targets_all),
        torch.cat(snr_all),
        num_classes=int(config.model["num_classes"]),
        snr_prediction_db=torch.cat(snr_predictions) if snr_predictions else None,
    )
    return loss_sum / sample_count, metrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: object, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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


def _resume_bindings(
    *,
    config_digest: str,
    split_manifest_sha256: str,
    assignment_sha256: object,
    project_commit: str,
    seed: int,
    data_protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    bindings = {
        "experiment_config_sha256": config_digest,
        "split_manifest_sha256": split_manifest_sha256,
        "assignment_sha256": assignment_sha256,
        "project_commit": project_commit,
        "seed": seed,
    }
    if data_protocol is not None:
        bindings["data_protocol"] = data_protocol
    return bindings


def _load_resume_state(path: Path, expected_bindings: dict[str, object]) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingError(f"Could not load resumable training state: {error}") from error
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise TrainingError("Resumable training state schema is invalid")
    if state.get("bindings") != expected_bindings:
        raise TrainingError("Resumable training state bindings differ from this run")
    required = {
        "schema_version",
        "bindings",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "history",
        "best_macro_f1",
        "best_epoch",
        "best_checkpoint_sha256",
        "stale_epochs",
        "torch_rng_state",
        "cuda_rng_states",
        "loader_generator_state",
        "augmentation_generator_state",
    }
    if set(state) != required:
        raise TrainingError("Resumable training state fields are incomplete")
    if (
        isinstance(state["epoch"], bool)
        or not isinstance(state["epoch"], int)
        or state["epoch"] < 1
        or not isinstance(state["history"], list)
        or len(state["history"]) != state["epoch"]
    ):
        raise TrainingError("Resumable training state epoch history is invalid")
    return state


def run_training(
    *,
    config: ExperimentConfig,
    config_path: Path,
    train_dataset: Dataset[ModulationSample],
    validation_dataset: Dataset[ModulationSample],
    output_dir: Path,
    project_root: Path,
    project_commit: str,
    split_manifest_sha256: str,
    device: torch.device,
    epoch_callback: Callable[[dict[str, object]], None] | None = None,
    batch_callback: Callable[[dict[str, object]], None] | None = None,
    resume: bool = False,
    data_protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    """Train a baseline and publish resumable, manifest-bound artifacts."""

    output_dir = output_dir.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    if not output_dir.is_dir() or output_dir == project_root or project_root in output_dir.parents:
        raise TrainingError("Output directory must exist outside the repository")
    contents = {path.name for path in output_dir.iterdir()}
    if contents and not resume:
        raise TrainingError("Output directory must be empty unless resume is requested")
    if resume and contents - {"best.pt", "last.pt"}:
        raise TrainingError("Resume directory contains unexpected or completed artifacts")
    if resume and "last.pt" not in contents:
        raise TrainingError("Resume requires a last.pt training state")
    if len(project_commit) != 40 or any(c not in "0123456789abcdef" for c in project_commit):
        raise TrainingError("project_commit must be a full lowercase Git commit")
    _sha256(split_manifest_sha256, "split_manifest_sha256")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA training requested but CUDA is unavailable")

    seed = int(config.training["seed"])
    deterministic = bool(config.training["deterministic"])
    _set_reproducibility(seed, deterministic)
    loader_generator = torch.Generator().manual_seed(seed)
    augmentation_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = _data_loader(train_dataset, config, shuffle=True, generator=loader_generator)
    validation_loader = _data_loader(
        validation_dataset,
        config,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2),
    )

    model = build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model.get("expansion", 1.25)),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
        permutation_seed=int(config.model.get("permutation_seed", 13)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optimizer["learning_rate"]),
        weight_decay=float(config.optimizer["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.training["max_epochs"])
    )
    amp_enabled = bool(config.training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    checkpoint_path = output_dir / "best.pt"
    resume_path = output_dir / "last.pt"
    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    selection_metric = config.selection_metric
    checkpoint_epoch = (
        int(config.training["checkpoint_epoch"]) if selection_metric == "fixed_epoch" else None
    )
    patience = int(config.training["early_stopping_patience"])
    min_delta = float(config.training["early_stopping_min_delta"])
    config_digest = experiment_config_sha256(config_path)
    bindings = _resume_bindings(
        config_digest=config_digest,
        split_manifest_sha256=split_manifest_sha256,
        assignment_sha256=config.data["assignment_sha256"],
        project_commit=project_commit,
        seed=seed,
        data_protocol=data_protocol,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    start_epoch = 1

    if resume:
        state = _load_resume_state(resume_path, bindings)
        completed_epoch = int(state["epoch"])
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        history = state["history"]
        best_macro_f1 = float(state["best_macro_f1"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        loader_generator.set_state(state["loader_generator_state"])
        augmentation_generator.set_state(state["augmentation_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        start_epoch = completed_epoch + 1
        if best_epoch > 0:
            if not checkpoint_path.is_file():
                raise TrainingError("Resumable state is missing its bound selected checkpoint")
            if state["best_checkpoint_sha256"] != _sha256_file(checkpoint_path):
                raise TrainingError("Selected checkpoint differs from the resumable training state")
        elif checkpoint_path.exists() or state["best_checkpoint_sha256"] is not None:
            raise TrainingError("Resume state contains an unexpected pre-selection checkpoint")
        if selection_metric == "validation_macro_f1" and stale_epochs >= patience:
            start_epoch = int(config.training["max_epochs"]) + 1

    for epoch in range(start_epoch, int(config.training["max_epochs"]) + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_samples = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            augmentation_generator,
            config,
            epoch,
            batch_callback,
        )
        validation_loss, metrics = _validate_epoch(model, validation_loader, device, config)
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_samples": train_samples,
            "validation_loss": validation_loss,
            "validation": asdict(metrics),
        }
        history.append(record)
        select_checkpoint = (
            selection_metric == "validation_macro_f1"
            and metrics.macro_f1 > best_macro_f1 + min_delta
        ) or (selection_metric == "fixed_epoch" and epoch == checkpoint_epoch)
        if select_checkpoint:
            best_macro_f1 = metrics.macro_f1
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "schema_version": 1,
                "model_name": config.model["name"],
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation": asdict(metrics),
                "bindings": {
                    **bindings,
                },
            }
            _atomic_torch_save(checkpoint, checkpoint_path)
        elif selection_metric == "validation_macro_f1":
            stale_epochs += 1
        resume_state = {
            "schema_version": 1,
            "bindings": bindings,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": (
                _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
            ),
            "stale_epochs": stale_epochs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "loader_generator_state": loader_generator.get_state(),
            "augmentation_generator_state": augmentation_generator.get_state(),
        }
        _atomic_torch_save(resume_state, resume_path)
        if epoch_callback is not None:
            epoch_callback({**record, "max_epochs": int(config.training["max_epochs"])})
        if selection_metric == "validation_macro_f1" and stale_epochs >= patience:
            break

    if best_epoch == 0 or not checkpoint_path.is_file():
        raise TrainingError("Training completed without publishing the selected checkpoint")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    result = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "purpose": config.purpose,
        "test_accessed": False,
        "bindings": {
            **bindings,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "amp_enabled": amp_enabled,
            "deterministic_algorithms": deterministic,
        },
        "model": {**config.model, "parameter_count": parameter_count},
        "selection_metric": selection_metric,
        "selected_checkpoint_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "history": history,
        "artifacts": {
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "absolute_paths_recorded": False,
        },
    }
    _atomic_json(result, output_dir / "metrics.json")
    resume_path.unlink(missing_ok=True)
    return result
