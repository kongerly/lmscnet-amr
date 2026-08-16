# Code Workspace

This directory contains the reproducible implementation for the automatic modulation recognition study: environment definitions, data processing, model code, training, evaluation, configuration, tests, and experiment scripts.

## Environment

The standard environment is the Miniconda environment `na-lmscnet` with Python 3.11, PyTorch 2.13.0, and the CUDA 13.0 PyTorch build. Exact package versions are recorded in `environment.yml`. The setup script explicitly uses `conda-forge` for Conda packages and the official PyTorch index for the CUDA wheel.

```powershell
powershell -ExecutionPolicy Bypass -File .\code\scripts\setup_environment.ps1
```

Verify the installation with:

```powershell
python code/scripts/verify_environment.py --require-cuda
```

## Layout

```text
code/
|-- environment.yml       # Reproducible Miniconda environment definition
|-- configs/              # Experiment configurations
|-- src/                  # Data, model, training, and evaluation source
|-- scripts/              # Environment, training, and analysis scripts
|-- tests/                # Automated tests
`-- outputs/              # Local experiment outputs; excluded from source control
```

The current code stage contains the verified environment, frozen dataset adapter, baseline families, S0/S1/S2 models, learned-static and parameter-matched controls, SKNet-1D/AFNet adaptations, post-training gate interventions, fixed-epoch R6 configurations, and a resumable train/validation engine. The historical RadioML 2016.10A test is permanently locked, and this release authorizes no new test access.

Dataset archives must be acquired from their documented official source and stored outside the repository. Use `scripts/inventory_dataset.py` to record the archive SHA-256 and validate its tar metadata before any payload handling.
For the legacy RADIOML 2016.10A Python 2 payload, `scripts/inspect_pickle_payload.py` provides a bounded no-execution opcode and literal scan. It does not deserialize the dataset or establish its full schema.
After that scan succeeds, `scripts/validate_pickle_schema.py` uses a strict static protocol-0 interpreter to verify the complete modulation/SNR key grid, array shape, float32 dtype metadata, memory order, and inline buffer lengths. It does not call `pickle.load`, import or execute pickle globals, construct NumPy arrays, inspect numeric sample values, or extract files.
After schema validation succeeds, `scripts/audit_numeric_quality.py` creates read-only NumPy views over one validated cell buffer at a time. It audits finite values, range, power, channel DC, zero-energy samples, and exact SHA-256 sample/cell duplicates without deserializing pickle or writing converted data.
Before any conversion runs, `scripts/validate_conversion_contract.py` strictly cross-checks the selected HDF5 layout, canonical row and sample-ID rules, required manifest bindings, and single-writer atomic-publication policy against the dataset specification. This validation does not create an HDF5 file, manifest, or data split.
The controlled converter is `scripts/convert_radioml_2016_10a.py`. It accepts only the contracted source archive, requires an existing output directory outside the repository, and publishes the HDF5 file plus its manifest without overwriting existing artifacts. `scripts/verify_radioml_2016_10a_conversion.py` independently repeats the static source scan, checks every source cell against the HDF5 rows, verifies all logical and physical SHA-256 digests, and treats a missing or mismatched manifest as an incomplete conversion.
`scripts/validate_split_contract.py` verifies the immutable source bindings, per-`(modulation, SNR)` 70/10/20 allocation, seed-2026 SHA-256 ranking algorithm, leakage policy, and test-isolation rules. `scripts/generate_radioml_2016_10a_split.py` writes the deterministic split manifest and exact-duplicate audit outside the repository.
The near-duplicate design is frozen in `docs/data/radioml_2016_10a_near_duplicate.md` and validated by `scripts/validate_near_duplicate_contract.py`. `scripts/validate_near_duplicate_fixture.py` reproduces the synthetic threshold-calibration fixture and a 64-sample exhaustive reference audit without writing artifacts. This evidence does not enable production candidate generation, real-data audit, or split generation.
`RadioML2016HDF5Dataset` validates the HDF5, conversion manifest, split manifest, and leakage audit before exposing train or validation rows. It removes per-channel DC and normalizes mean complex power. Test rows remain unavailable until an experiment-freeze manifest is implemented and validated.

Every experiment intended for reproduction must record its software environment, random seed, dataset version and hash, split indices, frozen configuration, and output path. Reproduction artifacts must remain outside the repository. The controlled revision is validation-only and does not authorize test construction or access.

## RadioML 2018.01A Test-Independence Audit

`scripts/audit_radioml_2018_test_independence.py` scans only source code,
configuration, JSON/YAML, logs, and documentation. For
`.h5/.hdf5/.pt/.pth/.npy/.npz` files, it records metadata only. The output
directory must be outside the repository and must not already exist:

```powershell
conda run -n na-lmscnet python code/scripts/audit_radioml_2018_test_independence.py `
  --artifact-root <external-artifact-root> `
  --output-dir <new-external-audit-directory> `
  --audit-date YYYY-MM-DD
```

The formal Phase R0 conclusion on 2026-08-14 was `ineligible`: the existing
2018.01A test partition had been constructed, its indexes had been read, and
its members had been included in an all-sample exact-duplicate audit. It
therefore cannot be unlocked as a new confirmatory test. Historical
`test_accessed=false` records mean only that no test-performance evaluation was
declared; they do not satisfy the stricter independence criterion used here.

## Major Revision namespace

`configs/revision/phase_r0.yml` records H1--H5, data-use prohibitions, and the
confirmatory-test block introduced on 2026-08-14 as a machine-readable
contract. `scripts/initialize_revision_namespace.py` validates the historical
test-consumption marker, the 2018 independence report, required governance-file
hashes, and the frozen directory name before atomically publishing an external
namespace:

```powershell
conda run -n na-lmscnet python code/scripts/initialize_revision_namespace.py `
  --output-dir <new-external-revision-namespace> `
  --initialization-date 2026-08-14 `
  --independence-report <external-independence-report.json> `
  --test-consumed-marker <external-test-consumed-marker.json>
```

The R0 namespace does not authorize formal training:
`formal_runs_authorized=false`. R1 still requires a clean implementation commit
and a new phase manifest. No test directory or confirmatory dataset may be
created under this authorization state.

## Baseline Training

`scripts/train_baseline.py` trains CNN2, CLDNN, ResNet1D, NA-LMSCNet, or its registered ablation variants using only the frozen train and validation rows. Each model has a publication-candidate config and a separate bounded smoke config; smoke metrics must not be reported as model results. Training uses seeded phase rotation and circular shift on train batches, AdamW, cosine annealing, AMP on CUDA, and macro-F1 early stopping. Checkpoints and metrics are written outside the repository and bind the config, split manifest, assignment, clean project commit, and seed. Interrupted runs publish `last.pt` after each epoch and resume with `--resume` only after validating all bindings and the best checkpoint digest. `scripts/train_cnn2.py` remains as a backward-compatible entry point.

`scripts/generate_core_ablation_report.py` compares the frozen seed-13 NA-LMSCNet checkpoint with `w/o multi-scale` and `fixed-average`. It replays validation predictions, verifies checkpoint/config bindings, and publishes the required classification and measured efficiency evidence outside the repository without constructing the test dataset.
