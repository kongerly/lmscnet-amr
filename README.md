# LMSCNet-AMR

Reproducible code for:

> *A Controlled Study of Multi-Scale Gating for Low-SNR Automatic Modulation Recognition*

This repository contains the data contracts, model implementations, training code,
controlled fusion variants, statistical scripts, configurations, and tests used in the
study. `NA-LMSCNet` remains as a historical namespace for the first noise-aware prototype;
SNR conditioning is not a contribution of the current work.

## Evidence boundary

The paper is a controlled empirical study, not a claim that sample-specific content
matching is independently beneficial or that S2 is superior to learned-static fusion,
SKNet-1D, or AFNet under the evaluated protocol.

- C1 and C4 remained unresolved in both the original and fixed-epoch validation routes.
- C3 showed that parameter count alone was insufficient to explain the observed result;
  it did not establish complete capacity matching.
- Mean-gate and shuffled-gate interventions showed bounded dependence of frozen S2
  checkpoints on gate assignment. The implemented shuffle was batch-local, with about
  95.8% same-modulation and 33.8% same-SNR pairing.
- Evidence is limited to a fixed RadioML 2016.10A split, a synthetic benchmark, and five
  training seeds. It does not establish cross-split, cross-dataset, or OTA generalization.
- The previously consumed RadioML 2016.10A test is permanently locked. RadioML 2018.01A
  test is ineligible for a new confirmation. No new test is authorized by this release.

See [`docs/revision_protocol.md`](docs/revision_protocol.md) for the control matrix,
configuration map, seeds, artifact hashes, and reproducibility limits.

## Repository layout

- `code/src/`: data, model, training, and evaluation modules.
- `code/configs/data/`: dataset, conversion, split, and leakage contracts.
- `code/configs/experiments/`: historical, R2, and fixed-epoch R6 configurations.
- `code/configs/revision/`: frozen revision-stage protocols.
- `code/scripts/`: training, replay, statistics, audit, and freeze-manifest scripts.
- `code/tests/`: unit and contract tests.
- `docs/data/`: dataset acquisition and split-generation documentation.
- `docs/training/`: model and baseline implementation notes.

Datasets, split artifacts, checkpoints, predictions, metrics, generated figures, and
submission packages are intentionally excluded.

## Environment

The reference environment uses Python 3.11 and is defined in `code/environment.yml`.
From Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\code\scripts\setup_environment.ps1
python code/scripts/verify_environment.py --require-cuda
```

Run the test suite from the repository root:

```powershell
python -m pytest
python -m ruff check code
```

Audit every tracked file and the Git history before a public update:

```powershell
python code/scripts/audit_public_repository.py `
  --as-of-date YYYY-MM-DD `
  --output <external-audit-report.json>
```

The audit rejects non-English CJK text, tracked email addresses, user-specific home
paths, credential-like material, datasets, model artifacts, caches, and oversized or
unreviewed binary files. Its JSON output contains one record for every tracked file.

## Data preparation

RadioML datasets are not redistributed. Start with
[`docs/data/radioml_2016_10a.md`](docs/data/radioml_2016_10a.md), then follow the
conversion and deterministic split protocols:

1. `code/scripts/inventory_dataset.py`
2. `code/scripts/validate_pickle_schema.py`
3. `code/scripts/audit_numeric_quality.py`
4. `code/scripts/validate_conversion_contract.py`
5. `code/scripts/convert_radioml_2016_10a.py`
6. `code/scripts/verify_radioml_2016_10a_conversion.py`
7. `code/scripts/validate_split_contract.py`
8. `code/scripts/generate_radioml_2016_10a_split.py`

All generated data and manifests must be written outside the repository. The split uses
deterministic SHA-256 ranking with seed `2026`; the training seeds are
`13, 37, 73, 101, 137`.

## Revision configurations

The principal controlled-study files are:

- `code/configs/revision/phase_r1.yml`
- `code/configs/revision/phase_r6.yml`
- `code/configs/experiments/revision_r2_*_selected.yml`
- `code/configs/experiments/revision_r6_*_fixed_epoch_radioml_2016_10a.yml`

R6 uses epoch 100 checkpoints without validation-based early stopping or checkpoint
selection. The validation split is used only for post-training assessment in that route.

## Third-party code

Adapted third-party components and their licenses are documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

The original code in this repository is released under the MIT License. See
[`LICENSE`](LICENSE).
