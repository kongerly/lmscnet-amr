from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from na_lmscnet.evaluation import count_macs, count_parameters
from na_lmscnet.models import build_model
from na_lmscnet.training import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "code/configs/experiments"
SELECTED_CONFIGS = (
    "lmscnet_s0_k3_radioml_2016_10a_selected.yml",
    "lmscnet_s0_k7_radioml_2016_10a_selected.yml",
    "lmscnet_s0_k15_radioml_2016_10a_selected.yml",
    "lmscnet_s0_wide_radioml_2016_10a_selected.yml",
    "lmscnet_s1_radioml_2016_10a_selected.yml",
    "lmscnet_s2_radioml_2016_10a_selected.yml",
)
EXPECTED_EFFICIENCY = {
    "lmscnet_s0_k3": (78_283, 4_001_312),
    "lmscnet_s0_k7": (80_203, 4_113_952),
    "lmscnet_s0_k15": (84_043, 4_339_232),
    "lmscnet_s0_wide": (90_031, 4_647_008),
    "lmscnet_s1": (90_763, 4_620_832),
    "lmscnet_s2": (124_861, 4_654_792),
}


def _build(filename: str) -> tuple[object, torch.nn.Module]:
    config = load_experiment_config(CONFIG_ROOT / filename)
    return config, build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model["expansion"]),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
    )


@pytest.mark.parametrize("filename", SELECTED_CONFIGS)
def test_final_config_and_model_are_test_isolated_and_snr_free(filename: str) -> None:
    config, model = _build(filename)
    outputs = model(torch.zeros((3, 2, 128)))
    model_name = str(config.model["name"])

    assert config.test_access == "forbidden"
    assert config.optimizer == {
        "name": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    assert config.training["max_epochs"] == 100
    assert config.training["early_stopping_patience"] == 12
    assert set(outputs) == {"logits", "scale_weights"}
    assert outputs["logits"].shape == (3, 11)
    assert all(
        token not in name.lower()
        for name, _ in model.named_parameters()
        for token in ("snr", "noise_embedding", "constant_embedding")
    )
    expected_parameters, expected_macs = EXPECTED_EFFICIENCY[model_name]
    assert count_parameters(model) == expected_parameters
    assert count_macs(model, (1, 2, 128), torch.device("cpu")) == expected_macs


def test_s0_s1_s2_configs_share_every_non_model_training_variable() -> None:
    configs = [load_experiment_config(CONFIG_ROOT / filename) for filename in SELECTED_CONFIGS]
    protocols = []
    for config in configs:
        protocol = asdict(config)
        protocol.pop("experiment_id")
        protocol.pop("model")
        protocols.append(protocol)

    assert all(protocol == protocols[0] for protocol in protocols[1:])


def test_s1_and_s2_differ_only_by_content_gate_state() -> None:
    _, s1 = _build("lmscnet_s1_radioml_2016_10a_selected.yml")
    _, s2 = _build("lmscnet_s2_radioml_2016_10a_selected.yml")
    s1_schema = {name: tuple(value.shape) for name, value in s1.state_dict().items()}
    s2_non_gate_schema = {
        name: tuple(value.shape) for name, value in s2.state_dict().items() if ".gate." not in name
    }

    assert s1_schema == s2_non_gate_schema
    assert len([name for name, _ in s2.named_parameters() if ".gate." in name]) == 24


def test_static_and_adaptive_scale_weights_follow_the_frozen_contract() -> None:
    _, s0 = _build("lmscnet_s0_k7_radioml_2016_10a_selected.yml")
    _, s1 = _build("lmscnet_s1_radioml_2016_10a_selected.yml")
    _, s2 = _build("lmscnet_s2_radioml_2016_10a_selected.yml")
    iq = torch.randn((2, 2, 128), generator=torch.Generator().manual_seed(13))
    s0_weights = s0(iq)["scale_weights"]
    s1_weights = s1(iq)["scale_weights"]
    s2_weights = s2(iq)["scale_weights"]

    assert s0_weights.shape == (2, 6, 1)
    assert torch.equal(s0_weights, torch.ones_like(s0_weights))
    assert s1_weights.shape == (2, 6, 3)
    assert s1_weights == pytest.approx(torch.full_like(s1_weights, 1.0 / 3.0).numpy())
    assert s2_weights.shape == (2, 6, 3)
    assert torch.allclose(s2_weights.sum(dim=-1), torch.ones((2, 6)))
    assert not torch.equal(s2_weights, torch.full_like(s2_weights, 1.0 / 3.0))


def test_widened_s0_is_within_five_percent_of_s2_macs() -> None:
    _, wide = _build("lmscnet_s0_wide_radioml_2016_10a_selected.yml")
    _, s2 = _build("lmscnet_s2_radioml_2016_10a_selected.yml")
    wide_macs = count_macs(wide, (1, 2, 128), torch.device("cpu"))
    s2_macs = count_macs(s2, (1, 2, 128), torch.device("cpu"))

    assert abs(wide_macs - s2_macs) / s2_macs < 0.05


@pytest.mark.parametrize(
    "filename",
    [
        "lmscnet_s0_k15_radioml_2016_10a_selected.yml",
        "lmscnet_s1_radioml_2016_10a_selected.yml",
        "lmscnet_s2_radioml_2016_10a_selected.yml",
    ],
)
def test_final_models_accept_1024_sample_windows(filename: str) -> None:
    _, model = _build(filename)
    outputs = model(torch.zeros((2, 2, 1024)))

    assert outputs["logits"].shape == (2, 11)


def test_final_models_reject_invalid_channels_and_short_windows() -> None:
    _, model = _build("lmscnet_s2_radioml_2016_10a_selected.yml")
    with pytest.raises(ValueError, match="length >= 16"):
        model(torch.zeros((2, 1, 1024)))
    with pytest.raises(ValueError, match="length >= 16"):
        model(torch.zeros((2, 2, 15)))
