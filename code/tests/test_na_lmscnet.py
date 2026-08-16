from __future__ import annotations

from pathlib import Path

import pytest
import torch

from na_lmscnet.evaluation import count_macs, count_parameters
from na_lmscnet.models import (
    NALMSCNet,
    NALMSCNetFixedAverage,
    NALMSCNetWithoutMultiScale,
    NALMSCNetWithoutSNRAuxiliary,
)
from na_lmscnet.training import NoiseAwareJointLoss, load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "code/configs/experiments/na_lmscnet_radioml_2016_10a_smoke.yml"
ABLATION_CONFIG = (
    PROJECT_ROOT / "code/configs/experiments/na_lmscnet_wo_snr_auxiliary_radioml_2016_10a_smoke.yml"
)
WO_MULTI_SCALE_CONFIG = (
    PROJECT_ROOT / "code/configs/experiments/na_lmscnet_wo_multi_scale_radioml_2016_10a_smoke.yml"
)
FIXED_AVERAGE_CONFIG = (
    PROJECT_ROOT / "code/configs/experiments/na_lmscnet_fixed_average_radioml_2016_10a_smoke.yml"
)


def test_na_smoke_config_is_bounded_and_test_isolated() -> None:
    config = load_experiment_config(CONFIG)

    assert config.model["name"] == "na_lmscnet"
    assert config.model["snr_loss_weight"] == 0.1
    assert config.training["max_train_batches"] == 2
    assert config.training["max_validation_batches"] == 2
    assert config.test_access == "forbidden"


def test_na_lmscnet_macs_and_parameter_gates_pass() -> None:
    model = NALMSCNet(num_classes=11, dropout=0.2)

    assert count_parameters(model) <= 500_000
    assert count_macs(model, (1, 2, 128), torch.device("cpu")) <= 5_470_976


def test_without_snr_auxiliary_uses_constant_embedding_and_no_snr_prediction() -> None:
    config = load_experiment_config(ABLATION_CONFIG)
    model = NALMSCNetWithoutSNRAuxiliary(num_classes=11, dropout=0.2)

    outputs = model(torch.zeros((3, 2, 128)))

    assert config.model == {
        "name": "na_lmscnet_wo_snr_auxiliary",
        "num_classes": 11,
        "dropout": 0.2,
    }
    assert set(outputs) == {"logits", "scale_weights"}
    assert outputs["logits"].shape == (3, 11)
    assert outputs["scale_weights"].shape == (3, 6, 3)
    assert model.constant_noise_embedding.shape == (1, 8)
    assert "snr_head" not in dict(model.named_modules())
    assert count_parameters(model) < count_parameters(NALMSCNet(num_classes=11, dropout=0.2))
    assert count_macs(model, (1, 2, 128), torch.device("cpu")) < count_macs(
        NALMSCNet(num_classes=11, dropout=0.2), (1, 2, 128), torch.device("cpu")
    )


@pytest.mark.parametrize(
    ("config_path", "model", "weight_shape", "expected_weight"),
    [
        (WO_MULTI_SCALE_CONFIG, NALMSCNetWithoutMultiScale(), (2, 6, 1), 1.0),
        (FIXED_AVERAGE_CONFIG, NALMSCNetFixedAverage(), (2, 6, 3), 1.0 / 3.0),
    ],
)
def test_core_ablation_preserves_snr_auxiliary_and_uses_static_weights(
    config_path: Path,
    model: torch.nn.Module,
    weight_shape: tuple[int, int, int],
    expected_weight: float,
) -> None:
    config = load_experiment_config(config_path)

    outputs = model(torch.zeros((2, 2, 128)))

    assert config.model["snr_loss_weight"] == 0.1
    assert set(outputs) == {"logits", "snr_hat", "scale_weights"}
    assert outputs["logits"].shape == (2, 11)
    assert outputs["snr_hat"].shape == (2,)
    assert outputs["scale_weights"].shape == weight_shape
    assert outputs["scale_weights"] == pytest.approx(
        torch.full(weight_shape, expected_weight).numpy()
    )


def test_core_ablation_efficiency_matches_the_intended_structure() -> None:
    reference = NALMSCNet()
    without_multi_scale = NALMSCNetWithoutMultiScale()
    fixed_average = NALMSCNetFixedAverage()

    assert count_parameters(without_multi_scale) < count_parameters(fixed_average)
    assert count_parameters(fixed_average) < count_parameters(reference)
    assert count_macs(without_multi_scale, (1, 2, 128), torch.device("cpu")) < count_macs(
        fixed_average, (1, 2, 128), torch.device("cpu")
    )
    assert count_macs(fixed_average, (1, 2, 128), torch.device("cpu")) < count_macs(
        reference, (1, 2, 128), torch.device("cpu")
    )


def test_joint_loss_uses_normalized_snr_target() -> None:
    outputs = {
        "logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
        "snr_hat": torch.tensor([-10.0, 9.0], requires_grad=True),
        "scale_weights": torch.full((2, 6, 3), 1.0 / 3.0),
    }
    total, classification, snr = NoiseAwareJointLoss()(
        outputs, torch.tensor([0, 1]), torch.tensor([-10.0, 18.0])
    )

    expected_classification = torch.nn.functional.cross_entropy(
        outputs["logits"], torch.tensor([0, 1])
    ).item()
    expected_snr = torch.nn.functional.smooth_l1_loss(
        torch.tensor([-10.0, 9.0]).add(1).div(19),
        torch.tensor([-10.0, 18.0]).add(1).div(19),
    ).item()
    assert classification.item() == pytest.approx(expected_classification)
    assert snr.item() == pytest.approx(expected_snr)
    assert total.item() == pytest.approx(expected_classification + 0.1 * expected_snr)
    total.backward()
    assert outputs["logits"].grad is not None
    assert outputs["snr_hat"].grad is not None
