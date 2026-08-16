from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from na_lmscnet.data import make_sample, validate_sample


def valid_sample() -> dict[str, object]:
    return {
        "iq": torch.zeros((2, 128), dtype=torch.float32),
        "modulation": 3,
        "snr": -8.0,
        "sample_id": "rml2016.10a:000001",
    }


def test_make_sample_returns_exact_contract() -> None:
    sample = make_sample(
        iq=torch.zeros((2, 128), dtype=torch.float32),
        modulation=3,
        snr=-8.0,
        sample_id="rml2016.10a:000001",
    )

    assert set(sample) == {"iq", "modulation", "snr", "sample_id"}
    assert sample["iq"].shape == (2, 128)


@pytest.mark.parametrize(
    "iq",
    [
        torch.zeros(128),
        torch.zeros((1, 128)),
        torch.zeros((2, 0)),
        torch.zeros((2, 128, 1)),
    ],
)
def test_rejects_invalid_iq_shape(iq: torch.Tensor) -> None:
    sample = valid_sample()
    sample["iq"] = iq

    with pytest.raises(ValueError, match="shape"):
        validate_sample(sample)


def test_rejects_non_finite_iq() -> None:
    sample = valid_sample()
    sample["iq"] = torch.tensor([[0.0, float("nan")], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite"):
        validate_sample(sample)


@pytest.mark.parametrize("modulation", [-1, 1.5, True])
def test_rejects_invalid_modulation(modulation: object) -> None:
    sample = valid_sample()
    sample["modulation"] = modulation

    with pytest.raises((TypeError, ValueError), match="modulation"):
        validate_sample(sample)


@pytest.mark.parametrize("snr", [float("nan"), float("inf"), True, "-8"])
def test_rejects_invalid_snr(snr: object) -> None:
    sample = valid_sample()
    sample["snr"] = snr

    with pytest.raises((TypeError, ValueError), match="snr"):
        validate_sample(sample)


@pytest.mark.parametrize("sample_id", ["", "   ", 123])
def test_rejects_invalid_sample_id(sample_id: object) -> None:
    sample = valid_sample()
    sample["sample_id"] = sample_id

    with pytest.raises((TypeError, ValueError), match="sample_id"):
        validate_sample(sample)


@pytest.mark.parametrize(
    "sample",
    [
        {key: value for key, value in valid_sample().items() if key != "snr"},
        {**valid_sample(), "true_label_name": "QPSK"},
    ],
)
def test_rejects_missing_or_unexpected_keys(sample: Mapping[str, object]) -> None:
    with pytest.raises(ValueError, match="Invalid sample keys"):
        validate_sample(sample)
