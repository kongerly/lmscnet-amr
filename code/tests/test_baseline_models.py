from __future__ import annotations

from pathlib import Path

import pytest
import torch

from na_lmscnet.evaluation import count_macs, count_parameters
from na_lmscnet.models import (
    CLDNN,
    MCLDNN,
    SEMSFN1D,
    MobileNetV2_1D,
    NALMSCNet,
    ResNet1D,
    ResNet1DMACMatched,
    build_model,
)
from na_lmscnet.training import load_experiment_config
from na_lmscnet.training.sweep import load_sweep_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT_ROOT / "code/configs/experiments"


@pytest.mark.parametrize(
    ("name", "model_type"),
    [
        ("cldnn", CLDNN),
        ("mcldnn", MCLDNN),
        ("mobilenetv2_1d", MobileNetV2_1D),
        ("resnet1d", ResNet1D),
        ("resnet1d_macs", ResNet1DMACMatched),
        ("se_msfn_1d", SEMSFN1D),
    ],
)
def test_baseline_factory_outputs_logits(name: str, model_type: type[torch.nn.Module]) -> None:
    model = build_model(name, num_classes=11, dropout=0.2)
    logits = model(torch.randn(3, 2, 128))

    assert isinstance(model, model_type)
    assert logits.shape == (3, 11)
    assert bool(torch.isfinite(logits).all())


@pytest.mark.parametrize(
    ("filename", "name"),
    [
        ("cldnn_radioml_2016_10a.yml", "cldnn"),
        ("mcldnn_radioml_2016_10a.yml", "mcldnn"),
        ("mobilenetv2_1d_radioml_2016_10a.yml", "mobilenetv2_1d"),
        ("resnet1d_radioml_2016_10a.yml", "resnet1d"),
        ("resnet1d_macs_radioml_2016_10a.yml", "resnet1d_macs"),
        ("se_msfn_1d_radioml_2016_10a.yml", "se_msfn_1d"),
    ],
)
def test_repository_baseline_configs_are_train_validation_only(filename: str, name: str) -> None:
    config = load_experiment_config(CONFIGS / filename)

    assert config.model["name"] == name
    assert config.training["max_train_batches"] is None
    assert config.training["max_validation_batches"] is None
    assert config.test_access == "forbidden"


@pytest.mark.parametrize(
    "filename",
    [
        "cldnn_radioml_2016_10a_smoke.yml",
        "mcldnn_radioml_2016_10a_smoke.yml",
        "mobilenetv2_1d_radioml_2016_10a_smoke.yml",
        "resnet1d_radioml_2016_10a_smoke.yml",
        "resnet1d_macs_radioml_2016_10a_smoke.yml",
        "se_msfn_1d_radioml_2016_10a_smoke.yml",
    ],
)
def test_repository_smoke_configs_are_bounded(filename: str) -> None:
    config = load_experiment_config(CONFIGS / filename)

    assert config.purpose == "infrastructure_smoke_only"
    assert config.training["max_train_batches"] == 8
    assert config.training["max_validation_batches"] == 8
    assert config.test_access == "forbidden"


@pytest.mark.parametrize(
    ("filename", "name", "learning_rate", "dropout"),
    [
        ("cnn2_radioml_2016_10a_selected.yml", "cnn2", 0.0003, 0.0),
        ("cldnn_radioml_2016_10a_selected.yml", "cldnn", 0.001, 0.0),
        ("resnet1d_radioml_2016_10a_selected.yml", "resnet1d", 0.001, 0.0),
        ("na_lmscnet_radioml_2016_10a_selected.yml", "na_lmscnet", 0.001, 0.2),
    ],
)
def test_repository_selected_configs_freeze_sweep_hyperparameters(
    filename: str, name: str, learning_rate: float, dropout: float
) -> None:
    config = load_experiment_config(CONFIGS / filename)

    assert config.model["name"] == name
    assert config.purpose == "publication_candidate"
    assert config.model["dropout"] == dropout
    assert config.optimizer["learning_rate"] == learning_rate
    if name == "na_lmscnet":
        assert config.model["snr_loss_weight"] == 0.1
    assert config.training["seed"] == 13
    assert config.training["max_train_batches"] is None
    assert config.training["max_validation_batches"] is None
    assert config.test_access == "forbidden"


def test_baselines_reject_wrong_shape() -> None:
    for model in (
        CLDNN(),
        MCLDNN(),
        MobileNetV2_1D(),
        ResNet1D(),
        ResNet1DMACMatched(),
    ):
        with pytest.raises(ValueError, match=r"\[batch, 2, 128\]"):
            model(torch.randn(2, 2, 64))


def test_se_msfn_accepts_1024_but_other_extended_baselines_remain_fixed() -> None:
    assert SEMSFN1D(num_classes=24)(torch.randn(2, 2, 1024)).shape == (2, 24)
    assert SEMSFN1D(num_classes=24)(torch.randn(2, 2, 64)).shape == (2, 24)
    with pytest.raises(ValueError, match=r"\[batch, 2, 128\]"):
        MCLDNN(num_classes=24)(torch.randn(2, 2, 1024))


def test_model_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_model("unknown", num_classes=11, dropout=0.2)


def test_extended_baseline_complexity_controls_are_frozen() -> None:
    s2_macs = 4_654_792
    s2_parameters = 124_861
    mac_matched = ResNet1DMACMatched()
    mobile = MobileNetV2_1D()

    assert abs(count_macs(mac_matched, (1, 2, 128), torch.device("cpu")) - s2_macs) / s2_macs < 0.05
    assert abs(count_parameters(mobile) - s2_parameters) / s2_parameters < 0.05


@pytest.mark.parametrize(
    "model",
    ["mcldnn", "mobilenetv2_1d", "resnet1d_macs", "se_msfn_1d"],
)
def test_extended_baseline_sweeps_use_frozen_grid(model: str) -> None:
    contract = load_sweep_contract(
        CONFIGS / f"{model}_radioml_2016_10a_sweep.yml"
    )

    assert contract["seed"] == 13
    assert contract["grid"] == {
        "learning_rate": [0.001, 0.0003],
        "dropout": [0.0, 0.2],
    }
    assert contract["test_access"] == "forbidden"


def test_na_lmscnet_outputs_noise_aware_contract() -> None:
    model = NALMSCNet(num_classes=11, dropout=0.2)
    outputs = model(torch.randn(4, 2, 128))

    assert set(outputs) == {"logits", "snr_hat", "scale_weights"}
    assert outputs["logits"].shape == (4, 11)
    assert outputs["snr_hat"].shape == (4,)
    assert outputs["scale_weights"].shape == (4, 6, 3)
    assert bool(torch.isfinite(outputs["logits"]).all())
    assert bool(torch.isfinite(outputs["snr_hat"]).all())
    assert bool(torch.isfinite(outputs["scale_weights"]).all())
    assert bool((outputs["snr_hat"] >= -20.0).all())
    assert bool((outputs["snr_hat"] <= 18.0).all())
    assert torch.allclose(
        outputs["scale_weights"].sum(dim=-1), torch.ones(4, 6), atol=1e-6
    )


def test_na_lmscnet_has_fixed_efficiency_budget() -> None:
    model = NALMSCNet(num_classes=11, dropout=0.2)
    assert sum(parameter.numel() for parameter in model.parameters()) <= 500_000
