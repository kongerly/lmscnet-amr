"""Data contracts and dataset adapters."""

from na_lmscnet.data.contracts import ModulationSample, make_sample, validate_sample
from na_lmscnet.data.conversion_contract import (
    ConversionContractError,
    conversion_contract_sha256,
    conversion_row_index,
    conversion_sample_id,
    load_conversion_contract,
)
from na_lmscnet.data.hdf5_conversion import (
    ConversionError,
    convert_archive,
    inspect_hdf5,
    logical_content_sha256,
    verify_conversion,
)
from na_lmscnet.data.near_duplicate_contract import (
    NearDuplicateContractError,
    load_near_duplicate_contract,
    max_abs_normalized_circular_correlation,
    near_duplicate_contract_sha256,
    power_normalized_complex,
)
from na_lmscnet.data.near_duplicate_fixture import (
    NearDuplicateFixtureError,
    build_near_duplicate_fixture_evidence,
    deterministic_fixture_sample,
    load_near_duplicate_fixture_contract,
    near_duplicate_fixture_contract_sha256,
)
from na_lmscnet.data.numeric_quality import audit_numeric_quality_archive
from na_lmscnet.data.pickle_safety import (
    PickleScanReport,
    UnsafePickleError,
    inspect_pickle_archive,
    scan_pickle_stream,
)
from na_lmscnet.data.pickle_schema import (
    validate_pickle_schema_archive,
    validate_pickle_schema_stream,
)
from na_lmscnet.data.preprocessing import (
    DEFAULT_PREPROCESSING_MODE,
    PREPROCESSING_MODES,
    GlobalZScoreStatistics,
    PreprocessingError,
    compute_global_zscore_statistics,
)
from na_lmscnet.data.provenance import (
    UnsafeArchiveError,
    build_archive_inventory,
    inspect_tar_bz2,
    sha256_file,
)
from na_lmscnet.data.radioml_2018 import (
    RadioML2018Error,
    RadioML2018HDF5Dataset,
    audit_radioml_2018_source,
    generate_radioml_2018_split,
)
from na_lmscnet.data.radioml_hdf5 import (
    RadioML2016FrozenTestDataset,
    RadioML2016HDF5Dataset,
    RadioMLDatasetError,
    preprocess_iq,
)
from na_lmscnet.data.split_contract import (
    SplitContractError,
    allocation_counts,
    assign_stratum_sample_ids,
    load_split_contract,
    rank_sample_ids,
    split_contract_sha256,
    split_rank_digest,
)
from na_lmscnet.data.split_manifest import (
    SplitManifestError,
    generate_split_artifacts,
    load_split_artifacts,
    validate_split_assignments,
)

__all__ = [
    "ModulationSample",
    "NearDuplicateContractError",
    "NearDuplicateFixtureError",
    "ConversionContractError",
    "ConversionError",
    "PickleScanReport",
    "PREPROCESSING_MODES",
    "DEFAULT_PREPROCESSING_MODE",
    "GlobalZScoreStatistics",
    "PreprocessingError",
    "RadioML2016HDF5Dataset",
    "RadioML2016FrozenTestDataset",
    "RadioML2018Error",
    "RadioML2018HDF5Dataset",
    "RadioMLDatasetError",
    "SplitContractError",
    "SplitManifestError",
    "UnsafeArchiveError",
    "UnsafePickleError",
    "build_archive_inventory",
    "allocation_counts",
    "assign_stratum_sample_ids",
    "conversion_contract_sha256",
    "conversion_row_index",
    "conversion_sample_id",
    "convert_archive",
    "compute_global_zscore_statistics",
    "audit_numeric_quality_archive",
    "audit_radioml_2018_source",
    "build_near_duplicate_fixture_evidence",
    "deterministic_fixture_sample",
    "generate_split_artifacts",
    "generate_radioml_2018_split",
    "inspect_tar_bz2",
    "inspect_pickle_archive",
    "inspect_hdf5",
    "load_conversion_contract",
    "logical_content_sha256",
    "load_near_duplicate_contract",
    "load_near_duplicate_fixture_contract",
    "load_split_contract",
    "load_split_artifacts",
    "make_sample",
    "max_abs_normalized_circular_correlation",
    "near_duplicate_contract_sha256",
    "near_duplicate_fixture_contract_sha256",
    "power_normalized_complex",
    "preprocess_iq",
    "sha256_file",
    "scan_pickle_stream",
    "rank_sample_ids",
    "split_contract_sha256",
    "split_rank_digest",
    "validate_sample",
    "validate_split_assignments",
    "validate_pickle_schema_archive",
    "validate_pickle_schema_stream",
    "verify_conversion",
]
