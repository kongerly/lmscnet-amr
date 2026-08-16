from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import Dataset

from na_lmscnet.data.contracts import ModulationSample, make_sample
from na_lmscnet.training.engine import (
    TrainingError,
    augment_iq_batch,
    load_experiment_config,
    run_training,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = PROJECT_ROOT / "code/configs/experiments/cnn2_radioml_2016_10a_smoke.yml"
NA_SMOKE_CONFIG = PROJECT_ROOT / "code/configs/experiments/na_lmscnet_radioml_2016_10a_smoke.yml"
NA_WO_SNR_SMOKE_CONFIG = (
    PROJECT_ROOT / "code/configs/experiments/na_lmscnet_wo_snr_auxiliary_radioml_2016_10a_smoke.yml"
)
NA_CORE_ABLATION_SMOKE_CONFIGS = (
    PROJECT_ROOT / "code/configs/experiments/na_lmscnet_wo_multi_scale_radioml_2016_10a_smoke.yml",
    PROJECT_ROOT / "code/configs/experiments/na_lmscnet_fixed_average_radioml_2016_10a_smoke.yml",
)
FULL_EPOCH_SMOKE_CONFIG = (
    PROJECT_ROOT / "code/configs/experiments/cnn2_radioml_2016_10a_full_epoch_smoke.yml"
)


class FixtureDataset(Dataset[ModulationSample]):
    def __init__(self, count: int) -> None:
        generator = torch.Generator().manual_seed(13)
        self.samples = [
            make_sample(
                iq=torch.randn((2, 128), generator=generator),
                modulation=index % 11,
                snr=float(-10 + 2 * (index % 6)),
                sample_id=f"fixture:{index:04d}",
            )
            for index in range(count)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ModulationSample:
        return self.samples[index]


def test_loads_repository_smoke_config() -> None:
    config = load_experiment_config(SMOKE_CONFIG)

    assert config.experiment_id == "cnn2_radioml_2016_10a_smoke_v1"
    assert config.training["max_train_batches"] == 8
    assert config.test_access == "forbidden"


def test_full_epoch_smoke_uses_all_train_and_validation_batches() -> None:
    config = load_experiment_config(FULL_EPOCH_SMOKE_CONFIG)

    assert config.purpose == "infrastructure_smoke_only"
    assert config.training["max_epochs"] == 1
    assert config.training["max_train_batches"] is None
    assert config.training["max_validation_batches"] is None
    assert config.test_access == "forbidden"


def test_na_lmscnet_training_smoke_records_snr_metric(tmp_path: Path) -> None:
    config = load_experiment_config(NA_SMOKE_CONFIG)
    output = tmp_path / "na-output"
    output.mkdir()

    result = run_training(
        config=config,
        config_path=NA_SMOKE_CONFIG,
        train_dataset=FixtureDataset(8),
        validation_dataset=FixtureDataset(8),
        output_dir=output,
        project_root=PROJECT_ROOT,
        project_commit="1" * 40,
        split_manifest_sha256="2" * 64,
        device=torch.device("cpu"),
    )

    assert result["test_accessed"] is False
    assert result["model"]["parameter_count"] <= 500_000
    assert result["history"][0]["validation"]["snr_mae_db"] is not None


def test_without_snr_auxiliary_training_smoke_omits_snr_metric(tmp_path: Path) -> None:
    config = load_experiment_config(NA_WO_SNR_SMOKE_CONFIG)
    output = tmp_path / "na-wo-snr-output"
    output.mkdir()

    result = run_training(
        config=config,
        config_path=NA_WO_SNR_SMOKE_CONFIG,
        train_dataset=FixtureDataset(16),
        validation_dataset=FixtureDataset(16),
        output_dir=output,
        project_root=PROJECT_ROOT,
        project_commit="1" * 40,
        split_manifest_sha256="2" * 64,
        device=torch.device("cpu"),
    )

    assert result["model"]["name"] == "na_lmscnet_wo_snr_auxiliary"
    assert result["history"][0]["validation"]["snr_mae_db"] is None
    assert result["test_accessed"] is False


@pytest.mark.parametrize("config_path", NA_CORE_ABLATION_SMOKE_CONFIGS)
def test_core_ablation_training_smoke_records_snr_metric(
    tmp_path: Path, config_path: Path
) -> None:
    config = load_experiment_config(config_path)
    output = tmp_path / str(config.model["name"])
    output.mkdir()

    result = run_training(
        config=config,
        config_path=config_path,
        train_dataset=FixtureDataset(16),
        validation_dataset=FixtureDataset(16),
        output_dir=output,
        project_root=PROJECT_ROOT,
        project_commit="1" * 40,
        split_manifest_sha256="2" * 64,
        device=torch.device("cpu"),
    )

    assert result["model"]["name"] == config.model["name"]
    assert result["history"][0]["validation"]["snr_mae_db"] is not None
    assert result["test_accessed"] is False


def test_rejects_config_that_allows_test_access(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["test_access"] = "allowed"
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingError, match="test isolation"):
        load_experiment_config(path)


def test_fixed_epoch_config_requires_frozen_final_epoch(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["training"]["max_epochs"] = 2
    raw["training"]["early_stopping_patience"] = 2
    raw["training"]["checkpoint_epoch"] = 2
    raw["selection_metric"] = "fixed_epoch"
    path = tmp_path / "fixed.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(path)

    assert config.selection_metric == "fixed_epoch"
    assert config.training["checkpoint_epoch"] == 2


def test_fixed_epoch_config_rejects_early_checkpoint(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["training"]["max_epochs"] = 2
    raw["training"]["early_stopping_patience"] = 2
    raw["training"]["checkpoint_epoch"] = 1
    raw["selection_metric"] = "fixed_epoch"
    path = tmp_path / "fixed.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingError, match="checkpoint must equal max_epochs"):
        load_experiment_config(path)


def test_augmentation_is_seeded_and_preserves_complex_power() -> None:
    iq = torch.randn((4, 2, 128), generator=torch.Generator().manual_seed(1))
    first = augment_iq_batch(
        iq,
        generator=torch.Generator().manual_seed(2),
        phase_rotation=True,
        circular_shift=True,
    )
    second = augment_iq_batch(
        iq,
        generator=torch.Generator().manual_seed(2),
        phase_rotation=True,
        circular_shift=True,
    )

    assert torch.equal(first, second)
    assert first.square().sum(dim=1).mean(dim=1) == pytest.approx(
        iq.square().sum(dim=1).mean(dim=1), rel=1e-6
    )


def test_training_writes_bound_checkpoint_and_metrics(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["data"]["batch_size"] = 4
    raw["training"]["max_train_batches"] = 1
    raw["training"]["max_validation_batches"] = 1
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    output = tmp_path / "output"
    output.mkdir()

    result = run_training(
        config=config,
        config_path=config_path,
        train_dataset=FixtureDataset(8),
        validation_dataset=FixtureDataset(4),
        output_dir=output,
        project_root=PROJECT_ROOT,
        project_commit="1" * 40,
        split_manifest_sha256="2" * 64,
        device=torch.device("cpu"),
    )

    assert result["test_accessed"] is False
    assert result["epochs_completed"] == 1
    assert (output / "best.pt").is_file()
    assert (output / "metrics.json").is_file()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["bindings"]["project_commit"] == "1" * 40
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["bindings"]["split_manifest_sha256"] == "2" * 64


def test_training_reports_batch_and_epoch_progress(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["data"]["batch_size"] = 4
    raw["training"]["max_epochs"] = 1
    raw["training"]["max_train_batches"] = 1
    raw["training"]["max_validation_batches"] = 1
    config_path = tmp_path / "progress.yml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    output = tmp_path / "progress-output"
    output.mkdir()
    batches: list[dict[str, object]] = []
    epochs: list[dict[str, object]] = []

    run_training(
        config=config,
        config_path=config_path,
        train_dataset=FixtureDataset(8),
        validation_dataset=FixtureDataset(4),
        output_dir=output,
        project_root=PROJECT_ROOT,
        project_commit="1" * 40,
        split_manifest_sha256="2" * 64,
        device=torch.device("cpu"),
        epoch_callback=epochs.append,
        batch_callback=batches.append,
    )

    assert len(batches) == 1
    assert batches[0]["event"] == "batch_complete"
    assert batches[0]["epoch"] == 1
    assert batches[0]["batch"] == 1
    assert batches[0]["total_batches"] == 1
    assert batches[0]["max_epochs"] == 1
    assert isinstance(batches[0]["train_loss"], float)
    assert len(epochs) == 1
    assert epochs[0]["epoch"] == 1
    assert epochs[0]["max_epochs"] == 1


def test_training_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(TrainingError, match="empty"):
        run_training(
            config=load_experiment_config(SMOKE_CONFIG),
            config_path=SMOKE_CONFIG,
            train_dataset=FixtureDataset(4),
            validation_dataset=FixtureDataset(4),
            output_dir=output,
            project_root=PROJECT_ROOT,
            project_commit="1" * 40,
            split_manifest_sha256="2" * 64,
            device=torch.device("cpu"),
        )


def test_training_resumes_from_last_completed_epoch(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["batch_size"] = 4
    raw["training"]["max_epochs"] = 2
    raw["training"]["early_stopping_patience"] = 2
    raw["training"]["max_train_batches"] = 1
    raw["training"]["max_validation_batches"] = 1
    config_path = tmp_path / "resume.yml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    output = tmp_path / "resume-output"
    output.mkdir()
    kwargs = {
        "config": config,
        "config_path": config_path,
        "train_dataset": FixtureDataset(8),
        "validation_dataset": FixtureDataset(4),
        "output_dir": output,
        "project_root": PROJECT_ROOT,
        "project_commit": "1" * 40,
        "split_manifest_sha256": "2" * 64,
        "device": torch.device("cpu"),
    }

    def interrupt_after_epoch(record: dict[str, object]) -> None:
        assert record["epoch"] == 1
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        run_training(**kwargs, epoch_callback=interrupt_after_epoch)
    assert (output / "last.pt").is_file()

    result = run_training(**kwargs, resume=True)

    assert result["epochs_completed"] == 2
    assert [record["epoch"] for record in result["history"]] == [1, 2]
    assert not (output / "last.pt").exists()


def test_fixed_epoch_training_selects_final_epoch_and_resumes_before_selection(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["batch_size"] = 4
    raw["training"]["max_epochs"] = 2
    raw["training"]["early_stopping_patience"] = 2
    raw["training"]["checkpoint_epoch"] = 2
    raw["training"]["max_train_batches"] = 1
    raw["training"]["max_validation_batches"] = 1
    raw["selection_metric"] = "fixed_epoch"
    config_path = tmp_path / "fixed-resume.yml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    output = tmp_path / "fixed-resume-output"
    output.mkdir()
    kwargs = {
        "config": config,
        "config_path": config_path,
        "train_dataset": FixtureDataset(8),
        "validation_dataset": FixtureDataset(4),
        "output_dir": output,
        "project_root": PROJECT_ROOT,
        "project_commit": "1" * 40,
        "split_manifest_sha256": "2" * 64,
        "device": torch.device("cpu"),
    }

    def interrupt_after_epoch(record: dict[str, object]) -> None:
        assert record["epoch"] == 1
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        run_training(**kwargs, epoch_callback=interrupt_after_epoch)
    assert (output / "last.pt").is_file()
    assert not (output / "best.pt").exists()

    result = run_training(**kwargs, resume=True)

    assert result["selection_metric"] == "fixed_epoch"
    assert result["selected_checkpoint_epoch"] == 2
    assert result["best_epoch"] == 2
    assert result["epochs_completed"] == 2
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2


def test_resume_rejects_binding_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "resume-output"
    output.mkdir()
    torch.save(
        {"schema_version": 1, "bindings": {"project_commit": "wrong"}},
        output / "last.pt",
    )
    with pytest.raises(TrainingError, match="bindings"):
        run_training(
            config=load_experiment_config(SMOKE_CONFIG),
            config_path=SMOKE_CONFIG,
            train_dataset=FixtureDataset(4),
            validation_dataset=FixtureDataset(4),
            output_dir=output,
            project_root=PROJECT_ROOT,
            project_commit="1" * 40,
            split_manifest_sha256="2" * 64,
            device=torch.device("cpu"),
            resume=True,
        )
