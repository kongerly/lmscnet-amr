"""Validated sample contract shared by all dataset adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TypedDict, cast

import torch


class ModulationSample(TypedDict):
    """One preprocessed I/Q sample and its supervised metadata."""

    iq: torch.Tensor
    modulation: int
    snr: float
    sample_id: str


SAMPLE_KEYS = frozenset(ModulationSample.__required_keys__)


def validate_sample(sample: Mapping[str, object]) -> None:
    """Raise when a dataset sample violates the project-wide interface."""

    keys = set(sample)
    if keys != SAMPLE_KEYS:
        missing = sorted(SAMPLE_KEYS - keys)
        unexpected = sorted(keys - SAMPLE_KEYS)
        raise ValueError(f"Invalid sample keys: missing={missing}, unexpected={unexpected}")

    iq = sample["iq"]
    if not isinstance(iq, torch.Tensor):
        raise TypeError("iq must be a torch.Tensor")
    if iq.ndim != 2 or iq.shape[0] != 2 or iq.shape[1] < 1:
        raise ValueError(f"iq must have shape [2, length] with length >= 1, got {tuple(iq.shape)}")
    if not iq.dtype.is_floating_point:
        raise TypeError(f"iq must use a floating-point dtype, got {iq.dtype}")
    if not bool(torch.isfinite(iq).all()):
        raise ValueError("iq must contain only finite values")

    modulation = sample["modulation"]
    if isinstance(modulation, bool) or not isinstance(modulation, Integral):
        raise TypeError("modulation must be a non-negative integer class index")
    if modulation < 0:
        raise ValueError("modulation must be non-negative")

    snr = sample["snr"]
    if isinstance(snr, bool) or not isinstance(snr, Real):
        raise TypeError("snr must be a finite real number in dB")
    if not math.isfinite(float(snr)):
        raise ValueError("snr must be finite")

    sample_id = sample["sample_id"]
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    if not sample_id.strip():
        raise ValueError("sample_id must not be empty")


def make_sample(
    *,
    iq: torch.Tensor,
    modulation: int,
    snr: float,
    sample_id: str,
) -> ModulationSample:
    """Build and validate the canonical dataset return value."""

    sample: dict[str, object] = {
        "iq": iq,
        "modulation": modulation,
        "snr": snr,
        "sample_id": sample_id,
    }
    validate_sample(sample)
    return cast(ModulationSample, sample)
