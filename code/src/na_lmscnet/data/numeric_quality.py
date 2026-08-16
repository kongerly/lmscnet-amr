"""Numeric quality audit for statically validated RADIOML cell buffers."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from na_lmscnet.data.pickle_safety import UnsafePickleError
from na_lmscnet.data.pickle_schema import (
    _validated_expected_schema,
    validate_pickle_schema_archive,
)
from na_lmscnet.data.provenance import load_dataset_spec

MAX_AUDIT_CELLS = 4_096
MAX_AUDIT_SAMPLES = 1_000_000
MAX_AUDIT_VALUES = 500_000_000


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite_mask = np.isfinite(values)
    finite = values[finite_mask].astype(np.float64, copy=False)
    finite_count = int(finite.size)
    if finite_count:
        sum_value = float(np.sum(finite, dtype=np.float64))
        sum_square = float(np.sum(np.square(finite), dtype=np.float64))
        mean = sum_value / finite_count
        mean_square = sum_square / finite_count
        return {
            "finite_count": finite_count,
            "min_value": float(np.min(finite)),
            "max_value": float(np.max(finite)),
            "mean_value": mean,
            "mean_square": mean_square,
            "rms_value": math.sqrt(mean_square),
            "sum_value": sum_value,
            "sum_square": sum_square,
        }
    return {
        "finite_count": 0,
        "min_value": None,
        "max_value": None,
        "mean_value": None,
        "mean_square": None,
        "rms_value": None,
        "sum_value": 0.0,
        "sum_square": 0.0,
    }


def _duplicate_summary(counts: Counter[bytes]) -> dict[str, int]:
    duplicate_counts = [count for count in counts.values() if count > 1]
    return {
        "distinct_duplicate_hashes": len(duplicate_counts),
        "duplicate_occurrences_beyond_first": sum(count - 1 for count in duplicate_counts),
    }


@dataclass
class NumericQualityCollector:
    """Collect bounded statistics and exact fingerprints one validated cell at a time."""

    expected: dict[str, Any]
    cell_reports: list[dict[str, Any]] = field(default_factory=list)
    sample_hash_counts: Counter[bytes] = field(default_factory=Counter)
    cell_hash_counts: Counter[bytes] = field(default_factory=Counter)
    finite_count: int = 0
    nan_count: int = 0
    positive_infinity_count: int = 0
    negative_infinity_count: int = 0
    zero_value_count: int = 0
    zero_energy_sample_count: int = 0
    sum_value: float = 0.0
    sum_square: float = 0.0
    iq_power_sum: float = 0.0
    iq_power_count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    channel_finite_counts: list[int] = field(default_factory=lambda: [0, 0])
    channel_sums: list[float] = field(default_factory=lambda: [0.0, 0.0])

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> NumericQualityCollector:
        expected = _validated_expected_schema(spec)
        sample_values = math.prod(expected["sample_shape"])
        total_values = expected["total_samples"] * sample_values
        if expected["total_samples"] > MAX_AUDIT_SAMPLES or total_values > MAX_AUDIT_VALUES:
            raise ValueError("Numeric audit exceeds the configured dataset limit")
        channels = expected["sample_shape"][0]
        return cls(
            expected=expected,
            channel_finite_counts=[0] * channels,
            channel_sums=[0.0] * channels,
        )

    @property
    def expected_cell_keys(self) -> set[tuple[str, int]]:
        return {
            (modulation, snr)
            for modulation in self.expected["modulations"]
            for snr in self.expected["snr_db"]
        }

    def observe(self, key: tuple[str, int], payload: bytes) -> None:
        if key not in self.expected_cell_keys:
            raise UnsafePickleError(f"Numeric audit received an unexpected cell: {key!r}")
        if any(
            report["modulation"] == key[0] and report["snr_db"] == key[1]
            for report in self.cell_reports
        ):
            raise UnsafePickleError(f"Numeric audit received a duplicate cell: {key!r}")
        if len(self.cell_reports) >= MAX_AUDIT_CELLS:
            raise UnsafePickleError("Numeric audit exceeds the configured cell limit")

        samples_per_cell = self.expected["samples_per_cell"]
        channels, length = self.expected["sample_shape"]
        expected_bytes = samples_per_cell * channels * length * 4
        if len(payload) != expected_bytes:
            raise UnsafePickleError(
                f"Numeric audit cell {key!r} has {len(payload)} bytes, expected {expected_bytes}"
            )
        values = np.frombuffer(payload, dtype=np.dtype("<f4"))
        if values.flags.writeable:
            raise UnsafePickleError("Numeric audit view must be read-only")
        if values.size > MAX_AUDIT_VALUES:
            raise UnsafePickleError("Numeric audit cell exceeds the configured value limit")
        array = values.reshape(samples_per_cell, channels, length)

        summary = _finite_summary(values)
        nan_count = int(np.count_nonzero(np.isnan(values)))
        positive_infinity_count = int(np.count_nonzero(np.isposinf(values)))
        negative_infinity_count = int(np.count_nonzero(np.isneginf(values)))
        zero_value_count = int(np.count_nonzero(values == 0))
        finite_sample_mask = np.all(np.isfinite(array), axis=(1, 2))
        zero_energy_mask = finite_sample_mask & np.all(array == 0, axis=(1, 2))
        zero_energy_samples = int(np.count_nonzero(zero_energy_mask))
        finite_iq_mask = np.all(np.isfinite(array), axis=1)
        iq_power = np.sum(array.astype(np.float64, copy=False) ** 2, axis=1)
        finite_iq_power = iq_power[finite_iq_mask]
        cell_iq_power_sum = float(np.sum(finite_iq_power, dtype=np.float64))
        cell_iq_power_count = int(finite_iq_power.size)
        cell_mean_iq_power = (
            cell_iq_power_sum / cell_iq_power_count if cell_iq_power_count else None
        )

        channel_dc_means: list[float | None] = []
        for channel_index in range(channels):
            channel_values = array[:, channel_index, :]
            finite_channel = channel_values[np.isfinite(channel_values)].astype(
                np.float64, copy=False
            )
            channel_count = int(finite_channel.size)
            channel_sum = float(np.sum(finite_channel, dtype=np.float64))
            channel_dc_means.append(channel_sum / channel_count if channel_count else None)
            self.channel_finite_counts[channel_index] += channel_count
            self.channel_sums[channel_index] += channel_sum

        sample_hash_counts: Counter[bytes] = Counter()
        sample_size_bytes = channels * length * 4
        payload_view = memoryview(payload)
        for sample_index in range(samples_per_cell):
            start = sample_index * sample_size_bytes
            digest = hashlib.sha256(payload_view[start : start + sample_size_bytes]).digest()
            sample_hash_counts[digest] += 1
            self.sample_hash_counts[digest] += 1
        cell_digest = hashlib.sha256(payload).digest()
        self.cell_hash_counts[cell_digest] += 1

        finite_count = int(summary["finite_count"])
        self.finite_count += finite_count
        self.nan_count += nan_count
        self.positive_infinity_count += positive_infinity_count
        self.negative_infinity_count += negative_infinity_count
        self.zero_value_count += zero_value_count
        self.zero_energy_sample_count += zero_energy_samples
        self.sum_value += float(summary["sum_value"])
        self.sum_square += float(summary["sum_square"])
        self.iq_power_sum += cell_iq_power_sum
        self.iq_power_count += cell_iq_power_count
        cell_min = summary["min_value"]
        cell_max = summary["max_value"]
        if cell_min is not None:
            self.min_value = cell_min if self.min_value is None else min(self.min_value, cell_min)
        if cell_max is not None:
            self.max_value = cell_max if self.max_value is None else max(self.max_value, cell_max)

        self.cell_reports.append(
            {
                "modulation": key[0],
                "snr_db": key[1],
                "sha256": cell_digest.hex(),
                "value_count": int(values.size),
                "finite_count": finite_count,
                "nan_count": nan_count,
                "positive_infinity_count": positive_infinity_count,
                "negative_infinity_count": negative_infinity_count,
                "zero_value_count": zero_value_count,
                "zero_energy_sample_count": zero_energy_samples,
                "min_value": cell_min,
                "max_value": cell_max,
                "mean_value": summary["mean_value"],
                "mean_square": summary["mean_square"],
                "rms_value": summary["rms_value"],
                "mean_iq_power": cell_mean_iq_power,
                "channel_dc_means": channel_dc_means,
                "sample_duplicates": _duplicate_summary(sample_hash_counts),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        expected_cells = len(self.expected_cell_keys)
        expected_samples = self.expected["total_samples"]
        expected_values = expected_samples * math.prod(self.expected["sample_shape"])
        if len(self.cell_reports) != expected_cells:
            raise UnsafePickleError(
                f"Numeric audit observed {len(self.cell_reports)} cells, expected {expected_cells}"
            )
        if sum(self.sample_hash_counts.values()) != expected_samples:
            raise UnsafePickleError("Numeric audit sample count does not match the expected total")
        observed_values = (
            self.finite_count
            + self.nan_count
            + self.positive_infinity_count
            + self.negative_infinity_count
        )
        if observed_values != expected_values:
            raise UnsafePickleError("Numeric audit value count does not match the expected total")
        mean_value = self.sum_value / self.finite_count if self.finite_count else None
        mean_square = self.sum_square / self.finite_count if self.finite_count else None
        channel_dc_means = [
            channel_sum / channel_count if channel_count else None
            for channel_sum, channel_count in zip(
                self.channel_sums,
                self.channel_finite_counts,
                strict=True,
            )
        ]
        dataset_hasher = hashlib.sha256()
        for report in sorted(
            self.cell_reports, key=lambda item: (item["modulation"], item["snr_db"])
        ):
            dataset_hasher.update(report["modulation"].encode("ascii"))
            dataset_hasher.update(b"\0")
            dataset_hasher.update(str(report["snr_db"]).encode("ascii"))
            dataset_hasher.update(b"\0")
            dataset_hasher.update(bytes.fromhex(report["sha256"]))

        return {
            "dataset_content_sha256": dataset_hasher.hexdigest(),
            "cell_count": len(self.cell_reports),
            "sample_count": expected_samples,
            "value_count": expected_values,
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "positive_infinity_count": self.positive_infinity_count,
            "negative_infinity_count": self.negative_infinity_count,
            "zero_value_count": self.zero_value_count,
            "zero_energy_sample_count": self.zero_energy_sample_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": mean_value,
            "mean_square": mean_square,
            "rms_value": math.sqrt(mean_square) if mean_square is not None else None,
            "mean_iq_power": (
                self.iq_power_sum / self.iq_power_count if self.iq_power_count else None
            ),
            "channel_dc_means": channel_dc_means,
            "sample_duplicates": _duplicate_summary(self.sample_hash_counts),
            "cell_duplicates": _duplicate_summary(self.cell_hash_counts),
            "all_values_finite": (
                self.nan_count == 0
                and self.positive_infinity_count == 0
                and self.negative_infinity_count == 0
            ),
            "all_samples_nonzero_energy": self.zero_energy_sample_count == 0,
            "cells": sorted(
                self.cell_reports,
                key=lambda item: (item["modulation"], item["snr_db"]),
            ),
        }


def audit_numeric_quality_archive(archive_path: Path, spec_path: Path) -> dict[str, Any]:
    """Audit numeric buffers while retaining the static no-deserialization boundary."""

    spec = load_dataset_spec(spec_path)
    collector = NumericQualityCollector.from_spec(spec)
    schema_report = validate_pickle_schema_archive(
        archive_path,
        spec_path,
        buffer_observer=collector.observe,
    )
    return {
        "schema_version": 1,
        "dataset_id": schema_report["dataset_id"],
        "archive": schema_report["archive"],
        "validated_schema": schema_report["validated_schema"],
        "numeric_quality": collector.as_dict(),
        "verification": {
            **schema_report["verification"],
            "numeric_values_inspected": True,
            "exact_sample_fingerprints_computed": True,
            "near_duplicate_detection_performed": False,
        },
        "security": {
            **schema_report["security"],
            "numpy_read_only_views_constructed": True,
            "converted_dataset_written": False,
        },
    }
