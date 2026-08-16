from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from na_lmscnet.data.radioml_hdf5 import (
    RadioML2016HDF5Dataset,
    RadioMLDatasetError,
    preprocess_iq,
)


def test_default_preprocess_normalizes_maximum_complex_amplitude() -> None:
    iq = torch.tensor([[2.0, 4.0, 6.0, 8.0], [-3.0, -1.0, 1.0, 3.0]])

    result = preprocess_iq(iq)

    assert result.square().sum(dim=0).sqrt().max() == pytest.approx(1.0, abs=1e-7)
    assert torch.equal(result, iq / iq.square().sum(dim=0).sqrt().max())


def test_historical_dc_power_preprocess_remains_explicitly_available() -> None:
    iq = torch.tensor([[2.0, 4.0, 6.0, 8.0], [-3.0, -1.0, 1.0, 3.0]])

    result = preprocess_iq(iq, mode="per_sample_dc_power")

    assert result.mean(dim=1) == pytest.approx(torch.zeros(2), abs=1e-7)
    assert result.square().sum(dim=0).mean() == pytest.approx(1.0, abs=2e-7)


def test_preprocess_rejects_zero_power() -> None:
    with pytest.raises(RadioMLDatasetError, match="positive"):
        preprocess_iq(torch.zeros((2, 128)))


def test_dataset_rejects_test_access_before_freeze(tmp_path: Path) -> None:
    with pytest.raises(RadioMLDatasetError, match="experiment freeze"):
        RadioML2016HDF5Dataset(
            split="test",  # type: ignore[arg-type]
            hdf5_path=tmp_path / "data.h5",
            conversion_manifest_path=tmp_path / "conversion.json",
            split_manifest_path=tmp_path / "split.json",
            leakage_audit_path=tmp_path / "audit.json",
            split_contract_path=tmp_path / "split.yml",
            dataset_spec_path=tmp_path / "data.yml",
            conversion_contract_path=tmp_path / "conversion.yml",
        )


def test_dataset_reads_canonical_sample_from_worker_local_hdf5(tmp_path: Path) -> None:
    hdf5 = tmp_path / "fixture.h5"
    samples = np.arange(512, dtype=np.float32).reshape(2, 2, 128)
    with h5py.File(hdf5, "w") as file:
        file.create_dataset("iq", data=samples)

    dataset = RadioML2016HDF5Dataset.__new__(RadioML2016HDF5Dataset)
    dataset.split = "train"
    dataset.hdf5_path = hdf5
    dataset.rows = (1,)
    dataset.normalize = False
    dataset._file = None
    dataset._pid = None
    dataset.conversion_contract = {
        "dataset_id": "fixture",
        "ordering": {
            "modulation_order": ["QPSK"],
            "snr_db_order": [0],
            "source_index": {"start": 0, "stop": 2, "step": 1},
        },
        "sample_id": {"separator": ":", "source_index_width": 1},
    }
    try:
        sample = dataset[0]
    finally:
        dataset.close()

    assert sample["iq"].shape == (2, 128)
    assert torch.equal(sample["iq"], torch.from_numpy(samples[1]))
    assert sample["modulation"] == 0
    assert sample["snr"] == 0.0
    assert sample["sample_id"] == "fixture:QPSK:+00:1"


def test_dataset_batch_read_restores_shuffled_order(tmp_path: Path) -> None:
    hdf5 = tmp_path / "fixture.h5"
    samples = np.arange(1024, dtype=np.float32).reshape(4, 2, 128)
    with h5py.File(hdf5, "w") as file:
        file.create_dataset("iq", data=samples)

    dataset = RadioML2016HDF5Dataset.__new__(RadioML2016HDF5Dataset)
    dataset.split = "train"
    dataset.hdf5_path = hdf5
    dataset.rows = (0, 1, 2, 3)
    dataset.normalize = False
    dataset._file = None
    dataset._pid = None
    dataset.conversion_contract = {
        "dataset_id": "fixture",
        "ordering": {
            "modulation_order": ["QPSK"],
            "snr_db_order": [0],
            "source_index": {"start": 0, "stop": 4, "step": 1},
        },
        "sample_id": {"separator": ":", "source_index_width": 1},
    }
    try:
        batch = dataset.__getitems__([3, 0, 2])
    finally:
        dataset.close()

    assert [sample["sample_id"] for sample in batch] == [
        "fixture:QPSK:+00:3",
        "fixture:QPSK:+00:0",
        "fixture:QPSK:+00:2",
    ]
    assert torch.equal(batch[0]["iq"], torch.from_numpy(samples[3]))
