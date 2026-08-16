"""Manifest-bound HDF5 dataset adapter for RadioML 2016.10A."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from na_lmscnet.data.contracts import ModulationSample, make_sample
from na_lmscnet.data.conversion_contract import (
    conversion_sample_id,
    load_conversion_contract,
)
from na_lmscnet.data.preprocessing import (
    DEFAULT_PREPROCESSING_MODE,
    GlobalZScoreStatistics,
    PreprocessingError,
    PreprocessingMode,
)
from na_lmscnet.data.preprocessing import (
    preprocess_iq as _preprocess_iq,
)
from na_lmscnet.data.split_manifest import load_split_artifacts


class RadioMLDatasetError(ValueError):
    """Raised when the manifest-bound dataset cannot be read safely."""


def preprocess_iq(
    iq: torch.Tensor,
    *,
    mode: PreprocessingMode = DEFAULT_PREPROCESSING_MODE,
    global_zscore: GlobalZScoreStatistics | None = None,
) -> torch.Tensor:
    """Backward-compatible public wrapper for explicit preprocessing modes."""

    try:
        return _preprocess_iq(iq, mode=mode, global_zscore=global_zscore)
    except PreprocessingError as error:
        raise RadioMLDatasetError(str(error)) from error


class RadioML2016HDF5Dataset(Dataset[ModulationSample]):
    """Read a frozen train or validation split with worker-local HDF5 handles."""

    def __init__(
        self,
        *,
        split: Literal["train", "validation"],
        hdf5_path: Path,
        conversion_manifest_path: Path,
        split_manifest_path: Path,
        leakage_audit_path: Path,
        split_contract_path: Path,
        dataset_spec_path: Path,
        conversion_contract_path: Path,
        preprocessing: PreprocessingMode = DEFAULT_PREPROCESSING_MODE,
        global_zscore: GlobalZScoreStatistics | None = None,
        normalize: bool | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise RadioMLDatasetError(
                "Only train and validation are available before an experiment freeze manifest"
            )
        if normalize is not None:
            if not isinstance(normalize, bool):
                raise TypeError("normalize must be a boolean when provided")
            preprocessing = "per_sample_dc_power" if normalize else "raw"
        if preprocessing == "global_zscore" and global_zscore is None:
            raise RadioMLDatasetError("global_zscore preprocessing requires training statistics")
        if preprocessing != "global_zscore" and global_zscore is not None:
            raise RadioMLDatasetError("global z-score statistics may only accompany global_zscore")
        artifacts = load_split_artifacts(
            manifest_path=split_manifest_path,
            leakage_audit_path=leakage_audit_path,
            hdf5_path=hdf5_path,
            conversion_manifest_path=conversion_manifest_path,
            split_contract_path=split_contract_path,
            dataset_spec_path=dataset_spec_path,
            conversion_contract_path=conversion_contract_path,
        )
        self.split = split
        self.hdf5_path = hdf5_path.resolve(strict=True)
        self.rows = tuple(artifacts["manifest"]["assignments"][split])
        self.assignment_sha256 = artifacts["manifest"]["digests"]["assignment_sha256"]
        self.split_manifest_sha256 = artifacts["leakage_audit"]["split_manifest_sha256"]
        self.conversion_contract = load_conversion_contract(
            conversion_contract_path, dataset_spec_path
        )
        self.preprocessing = preprocessing
        self.global_zscore = global_zscore
        self.normalize = preprocessing != "raw"
        self._file: h5py.File | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _hdf5(self) -> h5py.File:
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            self.close()
            self._file = h5py.File(self.hdf5_path, "r", libver="earliest", swmr=False)
            self._pid = pid
        return self._file

    def __getitem__(self, index: int) -> ModulationSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self.rows)
        if not 0 <= index < len(self.rows):
            raise IndexError("dataset index out of range")
        row = self.rows[index]
        iq_array = np.asarray(self._hdf5()["/iq"][row], dtype=np.float32).copy()
        return self._make_sample(row, torch.from_numpy(iq_array))

    def __getitems__(self, indices: list[int]) -> list[ModulationSample]:
        """Read a shuffled batch through one ordered HDF5 selection."""

        if not isinstance(indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise TypeError("dataset indices must be a list of integers")
        if not indices:
            return []
        normalized = [index + len(self.rows) if index < 0 else index for index in indices]
        if any(not 0 <= index < len(self.rows) for index in normalized):
            raise IndexError("dataset index out of range")
        rows = np.asarray([self.rows[index] for index in normalized], dtype=np.int64)
        order = np.argsort(rows)
        sorted_rows = rows[order]
        if len(np.unique(sorted_rows)) != len(sorted_rows):
            return [self[index] for index in normalized]
        sorted_iq = np.asarray(self._hdf5()["/iq"][sorted_rows], dtype=np.float32)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        iq_batch = sorted_iq[inverse].copy()
        return [
            self._make_sample(int(row), torch.from_numpy(iq))
            for row, iq in zip(rows, iq_batch, strict=True)
        ]

    def _make_sample(self, row: int, iq: torch.Tensor) -> ModulationSample:
        try:
            iq = _preprocess_iq(
                iq,
                mode=getattr(
                    self,
                    "preprocessing",
                    "per_sample_dc_power" if self.normalize else "raw",
                ),
                global_zscore=getattr(self, "global_zscore", None),
            )
        except PreprocessingError as error:
            raise RadioMLDatasetError(str(error)) from error

        ordering = self.conversion_contract["ordering"]
        samples_per_stratum = ordering["source_index"]["stop"]
        snr_count = len(ordering["snr_db_order"])
        modulation_index, remainder = divmod(row, snr_count * samples_per_stratum)
        snr_index, source_index = divmod(remainder, samples_per_stratum)
        modulation = ordering["modulation_order"][modulation_index]
        snr_db = ordering["snr_db_order"][snr_index]
        sample_id = conversion_sample_id(self.conversion_contract, modulation, snr_db, source_index)
        return make_sample(
            iq=iq,
            modulation=modulation_index,
            snr=float(snr_db),
            sample_id=sample_id,
        )

    def close(self) -> None:
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()
        self._file = None
        self._pid = None

    def __getstate__(self) -> dict[str, object]:
        self.close()
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __enter__(self) -> RadioML2016HDF5Dataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class RadioML2016FrozenTestDataset(RadioML2016HDF5Dataset):
    """Read the frozen test assignment only after one-shot authorization is consumed."""

    def __init__(
        self,
        *,
        authorization: Mapping[str, str],
        hdf5_path: Path,
        conversion_manifest_path: Path,
        split_manifest_path: Path,
        leakage_audit_path: Path,
        split_contract_path: Path,
        dataset_spec_path: Path,
        conversion_contract_path: Path,
        preprocessing: PreprocessingMode = DEFAULT_PREPROCESSING_MODE,
    ) -> None:
        if set(authorization) != {
            "manifest_sha256",
            "assignment_sha256",
            "preprocessing_mode",
        }:
            raise RadioMLDatasetError("Frozen test authorization fields are invalid")
        if authorization["preprocessing_mode"] != preprocessing:
            raise RadioMLDatasetError("Frozen test preprocessing differs from authorization")
        artifacts = load_split_artifacts(
            manifest_path=split_manifest_path,
            leakage_audit_path=leakage_audit_path,
            hdf5_path=hdf5_path,
            conversion_manifest_path=conversion_manifest_path,
            split_contract_path=split_contract_path,
            dataset_spec_path=dataset_spec_path,
            conversion_contract_path=conversion_contract_path,
        )
        assignment = artifacts["manifest"]["digests"]["assignment_sha256"]
        if authorization["assignment_sha256"] != assignment:
            raise RadioMLDatasetError("Frozen test assignment differs from authorization")
        self.split = "test"
        self.hdf5_path = hdf5_path.resolve(strict=True)
        self.rows = tuple(artifacts["manifest"]["assignments"]["test"])
        self.assignment_sha256 = assignment
        self.split_manifest_sha256 = artifacts["leakage_audit"]["split_manifest_sha256"]
        self.conversion_contract = load_conversion_contract(
            conversion_contract_path, dataset_spec_path
        )
        self.preprocessing = preprocessing
        self.global_zscore = None
        self.normalize = preprocessing != "raw"
        self.freeze_manifest_sha256 = authorization["manifest_sha256"]
        self._file = None
        self._pid = None
