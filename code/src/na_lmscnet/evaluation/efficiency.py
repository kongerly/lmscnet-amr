"""Reproducible parameter and multiply-accumulate counting."""

from __future__ import annotations

import torch
from torch import nn


class EfficiencyError(ValueError):
    """Raised when an efficiency measurement cannot be completed."""


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters using the project reporting convention."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_macs(model: nn.Module, input_shape: tuple[int, int, int], device: torch.device) -> int:
    """Count Conv/Linear/LSTM MACs for one forward pass."""

    if input_shape[0] < 1 or input_shape[1:] != (2, 128):
        raise EfficiencyError("Efficiency input shape must be [batch, 2, 128]")
    total = 0

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> None:
        nonlocal total
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if not isinstance(output, torch.Tensor):
                raise EfficiencyError("Convolution output is not a tensor")
            kernel = int(torch.tensor(module.kernel_size).prod().item())
            total += int(output.numel()) * (module.in_channels // module.groups) * kernel
        elif isinstance(module, nn.Linear):
            if not isinstance(output, torch.Tensor):
                raise EfficiencyError("Linear output is not a tensor")
            total += int(output.numel()) * module.in_features
        elif isinstance(module, nn.LSTM):
            value = inputs[0]
            if value.ndim != 3:
                raise EfficiencyError("LSTM input must be three-dimensional")
            sequence = value.shape[1] if module.batch_first else value.shape[0]
            batch = value.shape[0] if module.batch_first else value.shape[1]
            directions = 2 if module.bidirectional else 1
            for layer in range(module.num_layers):
                input_size = module.input_size if layer == 0 else module.hidden_size * directions
                total += int(
                    batch
                    * sequence
                    * directions
                    * 4
                    * module.hidden_size
                    * (input_size + module.hidden_size)
                )

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LSTM))
    ]
    was_training = model.training
    try:
        model.eval().to(device)
        with torch.inference_mode():
            model(torch.zeros(input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


__all__ = ["EfficiencyError", "count_macs", "count_parameters"]
