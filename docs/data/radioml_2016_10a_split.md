# RADIOML 2016.10A Split Design

## Scope

This document freezes the deterministic split design for the verified local RADIOML 2016.10A HDF5 artifact. The machine-readable contract is `code/configs/data/radioml_2016_10a_split.yml`. This block does not generate a split manifest, inspect test samples, implement a dataset adapter, normalize or augment I/Q data, or authorize model training.

Split generation remains disabled until a separate, versioned near-duplicate audit contract defines defensible representations, thresholds, candidate coverage, validation fixtures, and resource bounds. Exact-byte duplicate checks alone are not evidence that transformed or tolerance-based duplicates are absent.

## Source Binding

The contract binds the exact dataset specification, conversion contract, source archive, validated source content, HDF5 file, HDF5 logical content, conversion manifest, and conversion implementation commit. A future generator must verify all bindings before reading rows. It must refuse substituted or modified artifacts even when their filenames match.

Generated split and leakage-audit artifacts must remain outside the repository. They contain derived row assignments rather than the dataset itself, but keeping them external prevents accidental coupling to a local data path and preserves the repository boundary. Published reproducibility records may later report their hashes and provide a generator that recreates them from the contracted source.

## Stratification And Counts

Each `(modulation, SNR)` cell is one stratum. RADIOML 2016.10A has 11 modulation classes, 20 SNR values, and 1000 samples per cell, giving 220 strata and 220000 samples.

The contracted ratio is `70/10/20` in the fixed order `train`, `validation`, `test`. Counts use integer largest-remainder allocation:

1. Compute each floor allocation as `sample_count * weight // sum(weights)`.
2. Order residual seats by descending integer remainder.
3. Resolve equal remainders by the fixed split order.

For this dataset, no residual allocation is needed because every stratum has exactly 1000 samples. Every stratum contains 700 training, 100 validation, and 200 test samples. Dataset totals are 154000, 22000, and 44000 respectively. The general rounding rule is still frozen so another implementation cannot silently choose different behavior.

## Deterministic Assignment

The seed is `2026`. Assignment does not use NumPy, Python, or framework random-number generators because their algorithms and stream semantics are an unnecessary reproducibility dependency.

Each stratum is ranked independently. For every stable source-coordinate sample ID, the generator constructs three UTF-8 fields:

```text
na-lmscnet/radioml_2016_10a/split-ranking/v1
2026
<sample_id>
```

Each field is prefixed by its unsigned 64-bit big-endian byte length. SHA-256 is computed over the concatenated framed record. Samples are ordered by raw digest bytes, with UTF-8 sample-ID bytes as the explicit collision tie-break. The first allocated range becomes training, the next validation, and the last test. Manifest row-index arrays are then stored in ascending HDF5 row order for compact, canonical consumption.

The fixed test vector for `radioml_2016_10a:QPSK:+00:0999` is:

```text
6ba3315ff624ccf05597fcc515766a3a5692c34af59a79e1e15f663f703b67bb
```

## Leakage Policy

The future generation gate has three distinct parts:

- Exact duplicates: hash canonical little-endian float32 I/Q bytes for all samples. Any digest appearing in more than one split rejects the result. Within-split duplicates are reported. The earlier source audit observed no exact duplicates, but the split generator must verify this against the bound HDF5 artifact again.
- Near duplicates: the bound `radioml_2016_10a_near_duplicate_v1` contract and deterministic fixture document the similarity metric and bounded reference behavior. A global transformed-near-duplicate production audit is not complete and is not required to generate the benchmark split; the limitation must remain visible in artifacts and publications. This split must not be described as proving transformed-near-duplicate or capture-level independence.
- Adjacent windows: RADIOML 2016.10A does not expose capture/session provenance or window start offsets. `source_index` is only a position in a `(modulation, SNR)` array and must not be interpreted as time adjacency or a recording group. The project must record this limitation and must never claim that adjacent-window leakage was verified for this dataset.

These checks do not repair DeepSig's known errata and do not make this synthetic benchmark representative of real over-the-air data.

## Generated Split And Adapter

The external split manifest contains sorted HDF5 row indices for `154000` train, `22000` validation, and `44000` test samples. The companion leakage audit records the all-sample exact-byte check and the unresolved near-duplicate and adjacent-window limitations. Neither artifact is redistributed in the source repository.
The canonical assignment SHA-256 is `0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941`; a regenerated manifest must reproduce this digest before experiments are compared.

The HDF5 adapter validates both artifacts before reading rows. Final train and validation experiments use the canonical `{iq, modulation, snr, sample_id}` interface with `per_sample_max_abs`: each I/Q window is divided by its maximum finite positive complex amplitude, without DC removal or validation/test statistics. The earlier per-channel DC removal and mean complex-power normalization is retained only for historical experiment reproduction. Test construction is rejected until an experiment-freeze manifest binds the split, configuration, project commit, run seeds, and selected checkpoints.

## Test Isolation

Only `train` may be used for parameter fitting. Hyperparameter selection and early stopping may use `train` and `validation`; they must not read test samples or test metrics.

Test access requires a future `experiment-freeze-manifest-v1` artifact binding:

- The exact split-manifest SHA-256
- The exact experiment-configuration SHA-256
- The project Git commit
- All run seeds
- The selected checkpoint identities

The freeze record is created only after the architecture, preprocessing, optimization, tuning decision, metric implementation, and checkpoint-selection rule are fixed. Test evaluation is then a one-way reporting step. A change informed by test results creates a new, explicitly disclosed exploratory cycle; it must not be presented as untouched confirmatory evaluation.

## Manifest And Publication

The future split manifest will contain canonical sorted HDF5 row-index arrays and counts for each split and stratum. Its SHA-256 bindings cover the split contract, dataset and conversion specifications, conversion manifest, source archive and logical content, HDF5 physical and logical content, and assignment content. It records relative basenames only plus the project and software environment.

Publication is single-process, no-overwrite, and same-directory transactional. The writer uses unpredictable exclusive temporary files, flushes and `fsync`s them, publishes the split manifest first, and publishes the leakage-audit report last. The leakage-audit report is the completion marker. A split manifest without its matching, passing leakage report is incomplete and must never be consumed.

## Validation Gate

Validate the design contract from the repository root:

```powershell
conda run -n na-lmscnet python code/scripts/validate_split_contract.py
```

The expected summary reports seed `2026`, per-stratum counts `700/100/200`, totals `154000/22000/44000`, and `generation_enabled: false`. The next candidate block is the near-duplicate audit design review. No split generator, adapter, normalization, augmentation, model, or training implementation may begin before that gate is separately designed and verified.
