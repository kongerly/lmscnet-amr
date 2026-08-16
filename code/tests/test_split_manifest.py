from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from na_lmscnet.data.split_manifest import (
    SplitManifestError,
    _audit_exact_duplicates,
    validate_split_assignments,
)


def test_validates_complete_disjoint_sorted_assignments() -> None:
    assignments = {"train": [0, 2], "validation": [1], "test": [3]}

    assert (
        validate_split_assignments(assignments, {"train": 2, "validation": 1, "test": 1}, 4)
        == assignments
    )


@pytest.mark.parametrize(
    "assignments",
    [
        {"train": [0, 2], "validation": [1], "test": [2]},
        {"train": [2, 0], "validation": [1], "test": [3]},
        {"train": [0, 2], "validation": [1], "test": [4]},
        {"train": [0, True], "validation": [1], "test": [3]},
    ],
)
def test_rejects_invalid_assignments(assignments: dict[str, list[object]]) -> None:
    with pytest.raises(SplitManifestError):
        validate_split_assignments(
            assignments,
            {"train": 2, "validation": 1, "test": 1},
            4,  # type: ignore[arg-type]
        )


def _write_iq(path: Path, samples: np.ndarray) -> None:
    with h5py.File(path, "w") as file:
        file.create_dataset("iq", data=samples.astype("<f4"))


def test_exact_duplicate_audit_rejects_cross_split_group(tmp_path: Path) -> None:
    sample_a = np.arange(8, dtype=np.float32).reshape(2, 4)
    sample_b = np.flip(sample_a, axis=1).copy()
    hdf5 = tmp_path / "fixture.h5"
    _write_iq(hdf5, np.stack([sample_a, sample_b, sample_a]))

    audit = _audit_exact_duplicates(hdf5, {"train": [0], "validation": [1], "test": [2]})

    assert audit["passed"] is False
    assert audit["cross_split_group_count"] == 1


def test_exact_duplicate_audit_reports_within_split_group(tmp_path: Path) -> None:
    sample_a = np.arange(8, dtype=np.float32).reshape(2, 4)
    sample_b = np.flip(sample_a, axis=1).copy()
    hdf5 = tmp_path / "fixture.h5"
    _write_iq(hdf5, np.stack([sample_a, sample_a, sample_b]))

    audit = _audit_exact_duplicates(hdf5, {"train": [0, 1], "validation": [2], "test": []})

    assert audit["passed"] is True
    assert audit["within_split_group_count"] == 1
