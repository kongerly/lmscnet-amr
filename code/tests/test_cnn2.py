from __future__ import annotations

import pytest
import torch

from na_lmscnet.models import CNN2


def test_cnn2_returns_class_logits() -> None:
    model = CNN2(num_classes=11, dropout=0.2)

    logits = model(torch.zeros((4, 2, 128), dtype=torch.float32))

    assert logits.shape == (4, 11)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_666_587


@pytest.mark.parametrize(
    "input_tensor",
    [
        torch.zeros((2, 128)),
        torch.zeros((2, 2, 127)),
        torch.zeros((2, 2, 128), dtype=torch.int64),
    ],
)
def test_cnn2_rejects_invalid_input(input_tensor: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        CNN2()(input_tensor)


def test_cnn2_validates_constructor() -> None:
    with pytest.raises(ValueError):
        CNN2(num_classes=1)
    with pytest.raises(ValueError):
        CNN2(dropout=1.0)
