# RADIOML 2016.10A Conversion Design

## Scope

This document freezes the design boundary for converting the locally verified RADIOML 2016.10A archive into a non-executable training format. It does not perform the conversion, create a data split, implement a dataset adapter, or authorize training. The machine-readable contract is `code/configs/data/radioml_2016_10a_conversion.yml`.

## Format Decision

HDF5 is selected for the first converted dataset version. The pinned Miniconda environment already includes `h5py`, RadioML 2018.01A also requires HDF5 support, and a single output file permits straightforward atomic publication and whole-file integrity checking. The alternatives were rejected for this version:

| Candidate | Decision | Reason |
| --- | --- | --- |
| HDF5 | Selected | Existing pinned dependency, bounded row/chunk reads, typed datasets, Fletcher32 chunk checks, and one-file publication |
| Zarr | Rejected for v1 | Adds a dependency and turns publication into a multi-object directory transaction |
| Sharded NPY | Rejected for v1 | Multiplies files and manifest entries, with no transactional container or embedded checksum support |

This decision is project-specific. It does not claim that HDF5 is universally superior.

## Logical Layout

The output filename is `RML2016.10a.h5`. The HDF5 file contains only fixed-size numeric or fixed-width byte datasets; it must not contain Python objects, pickled attributes, variable-length object arrays, external links, virtual datasets, or user-defined filters.

| Dataset | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `/iq` | little-endian float32 | `[220000, 2, 128]` | Original validated I/Q values, without normalization or augmentation |
| `/modulation_index` | uint8 | `[220000]` | Index into `/modulation_names` |
| `/snr_db` | int8 | `[220000]` | Source SNR label in dB |
| `/source_index` | little-endian uint16 | `[220000]` | Original index within one `(modulation, SNR)` cell |
| `/modulation_names` | fixed-width ASCII bytes | `[11]` | Canonical class-name table |

Rows are ordered by the exact modulation list in the dataset specification, then ascending SNR, then source index `0..999`. The row formula is:

```text
row = ((modulation_index * 20) + snr_index) * 1000 + source_index
```

The stable sample identifier is derived from source coordinates rather than the physical row alone:

```text
radioml_2016_10a:<modulation>:<signed-SNR>:<four-digit-source-index>
```

For example, row `190999` is `radioml_2016_10a:QPSK:+00:0999`. A later split manifest must reference these stable identifiers or their deterministically verified row mapping. No split assignment belongs in the conversion file.

## Storage Rules

- `/iq` is chunked as `[256, 2, 128]`; the row metadata datasets use chunks of 4096 entries. Fletcher32 is enabled for all chunked datasets.
- Version 1 uses no compression or shuffle filter. This avoids optional filter dependencies and keeps logical-byte verification direct. Compression may be benchmarked only in a future contract version.
- `/modulation_names` is contiguous because it is tiny and immutable.
- The writer explicitly disables object timestamps and uses `libver='earliest'` to reduce avoidable physical-file variation. HDF5 1.14.6 does not expose the original `track_times` creation choice reliably after close/reopen, so the independent verifier does not claim to reconstruct that flag from the final file.
- The exact HDF5 file SHA-256 is an integrity identifier for one artifact. Cross-environment reproducibility is established by per-dataset canonical-byte digests and the logical-content digest, not by assuming different HDF5 library versions produce identical container bytes.

## Manifest Contract

The converter must publish `RML2016.10a.conversion-manifest.json` beside the HDF5 file. The manifest must contain relative basenames only and bind all of the following with SHA-256:

- Exact committed conversion-contract and dataset-specification bytes
- Source archive and validated source-content identities
- Final HDF5 file bytes
- Canonical logical content
- Each logical dataset in canonical C-order bytes

It must also record the project Git commit and the Python, NumPy, h5py, and HDF5 versions. The logical-content digest processes datasets in contract order. For each dataset, it hashes a record containing the UTF-8 dataset name, ASCII dtype string, unsigned 64-bit dimensions, and raw 32-byte dataset digest. Every variable-length field is preceded by its unsigned 64-bit big-endian byte length. This `length-prefixed-v1` framing prevents ambiguous concatenation. The actual manifest schema and verifier will be implemented with the converter; this design block defines the mandatory fields but does not generate a manifest.

## Transaction And Concurrency Rules

1. Refuse symlink/reparse-point inputs and outputs, unexpected source size/hash, existing final files, and output paths inside the repository.
2. Use one process and one writer. SWMR and concurrent conversion are disabled.
3. Create unpredictable temporary HDF5 and manifest files in the final output directory using exclusive creation.
4. Stream one statically validated cell at a time. Write each cell to its canonical row range; do not extract or deserialize the pickle.
5. Flush and close HDF5, reopen it read-only, then verify structure, values, labels, per-dataset digests, logical digest, and source-to-output sample identity.
6. Flush and `fsync` each temporary file before publication. Atomically replace the absent final HDF5 path first and publish the manifest last.
7. Treat the manifest as the completion marker. A final HDF5 file without its matching manifest is incomplete and must never be consumed by training code.
8. On failure, remove only temporary files created by the current process. Never overwrite or delete a pre-existing final dataset or manifest.

HDF5 does not provide a portable cross-filesystem transaction for two files. Publishing the manifest last makes interrupted states fail closed: readers require both artifacts and verify their mutual hashes before use.

## Validation Gate

Validate the committed design contract from the repository root:

```powershell
conda run -n na-lmscnet python code/scripts/validate_conversion_contract.py
```

The converter and manifest verifier may run only after this command, the unit tests, dependency audit, and repository safety checks pass. Data splitting, normalization, augmentation, model code, and training remain out of scope until conversion is independently verified.

## Conversion Commands

The output directory must already exist and must be outside the repository. Run the converter from a clean, committed project state:

```powershell
conda run -n na-lmscnet python code/scripts/convert_radioml_2016_10a.py `
  <data-dir>\RML2016.10a.tar.bz2 `
  --output-dir <data-dir>
```

Then independently verify both published artifacts against the original archive, dataset specification, and conversion contract:

```powershell
conda run -n na-lmscnet python code/scripts/verify_radioml_2016_10a_conversion.py `
  <data-dir>\RML2016.10a.tar.bz2 `
  <data-dir>\RML2016.10a.h5 `
  <data-dir>\RML2016.10a.conversion-manifest.json
```

The verifier re-runs the no-deserialization static archive interpreter and compares every source cell digest with its canonical HDF5 row range. It also verifies the exact dataset-specification and conversion-contract hashes, HDF5 structure and filters, row metadata, finite I/Q values, per-dataset logical hashes, whole-file hash, manifest schema, and source archive identity. Neither command creates a data split.

## Verified Local Conversion

The contracted archive was converted and independently verified locally on 2026-08-08. The artifacts remain outside the repository under `<data-dir>` and are not redistributed.

| Property | Verified value |
| --- | --- |
| Source archive SHA-256 | `7a1603dd61e557f45b6e113dc0c59be02a14509b77856c31bbb324a993f7974c` |
| Source dataset-content SHA-256 | `bcaf1ea9bca18db5b5e179352b18504e6f92d1db7f4cf5b12673c2e3fba9aef9` |
| HDF5 size | `226409344` bytes |
| HDF5 SHA-256 | `96120f40a9190bf24697227aaa7377a4e1cf883b3bb1b602b176f2622ebf7c63` |
| Logical-content SHA-256 | `0713dd71751ff18fa0f0de26e570afb0f18a8e00191748a3c4a10f9a3271bce4` |
| Manifest size | `3493` bytes |
| Manifest SHA-256 | `de5bcb3dc6c490dca774d18bb7f3d3fd79634b55f9e2c31af244ac55b8ea776e` |
| Bound implementation commit | `3d836ca356b2a78aa9b94bd54a2468db9bca24b9` |
| Bound dataset-spec SHA-256 | `016d42ea6555d9b00751307705b0722d7feff38636427aa3fcdb63a0d5389773` |
| Bound conversion-contract SHA-256 | `ec89644a023e909424142b22905ba9acbcfff41c94370a954f44b00b25e0eac1` |
| Verified cells / samples | `220 / 220000` |

The independent verifier returned `verified: true`. It confirmed the complete cell grid, static source schema, canonical HDF5 row order, finite numeric values, read-only HDF5 reopen, every source-cell-to-HDF5 digest comparison, all manifest bindings, and path redaction. The source archive hash was unchanged after conversion. A second full conversion in a separate temporary directory produced byte-identical HDF5 and manifest SHA-256 values and independently verified again; the temporary artifacts were then removed. This establishes a verified non-executable local representation of this exact synthetic benchmark; it does not resolve DeepSig's known errata, detect transformed near-duplicates, create a data split, or demonstrate real over-the-air generalization. The subsequent deterministic assignment and isolation boundary is defined separately in [the split design](radioml_2016_10a_split.md).
