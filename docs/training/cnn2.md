# Baseline Training

## Scope

All baselines consume normalized I/Q tensors with shape `[batch, 2, 128]`. CNN2 treats the tensor as a single-channel I/Q plane. CLDNN uses three temporal convolution blocks and an LSTM and is adapted from the fixed MIT-licensed SigDA source recorded in `THIRD_PARTY_NOTICES.md`. ResNet1D is a four-stage, two-block-per-stage one-dimensional ResNet-18 analogue. None contains the proposed multi-scale branches, noise-aware fusion, or SNR auxiliary task.

## Training Protocol

The publication-candidate configuration uses the frozen RadioML 2016.10A train and validation assignments, seed `13`, batch size `256`, AdamW with learning rate `0.001` and weight decay `0.0001`, cosine annealing, CUDA AMP, and deterministic execution. Training stops after at most 100 epochs or after validation macro F1 fails to improve for 12 epochs. Random global phase rotation and integer circular shift are applied only to train batches. No AWGN is added.

The bounded smoke configurations read eight train and validation batches for one epoch. The separate CNN2 full-epoch smoke traverses all frozen train and validation rows once. These configurations validate CUDA execution, data iteration, checkpoint publication, metrics serialization, and artifact bindings. Their metrics are not model-selection evidence and must not appear in result tables.

## Validation Tuning Sweep

`cnn2_radioml_2016_10a_sweep.yml` fixes the seed-13 grid to learning rates `{0.001, 0.0003}` and dropout values `{0.0, 0.2}`. All four runs consume the complete train and validation splits and use identical augmentation, optimizer family, scheduler, epoch limit, and early-stopping policy. Runs execute sequentially and completed run directories may be resumed only when all config, split, assignment, commit, seed, checkpoint, and metrics bindings match.

The selected configuration maximizes validation macro F1. Exact ties use lower validation loss and then lexicographically smaller run ID. The sweep does not read test data. Its selected run is model-selection evidence for later multi-seed training, not a final test result.

The seed-13 sweep completed on 2026-08-08. Validation macro F1 was `0.47715` for lr `1e-3`/dropout `0`, `0.45352` for lr `1e-3`/dropout `0.2`, `0.52201` for lr `3e-4`/dropout `0`, and `0.47065` for lr `3e-4`/dropout `0.2`. The selected publication configuration is `learning_rate=3e-4, dropout=0` (best epoch 64), frozen as `code/configs/experiments/cnn2_radioml_2016_10a_selected.yml`. The sweep summary is stored outside the repository and binds the split manifest SHA-256, assignment SHA-256, project commit, and seed. These values are model-selection evidence only and must not appear in result tables.

## Test Isolation And Artifacts

The training CLI constructs only train and validation datasets. The HDF5 adapter rejects test construction before an experiment-freeze manifest exists. The best checkpoint and `metrics.json` are written to a new empty directory outside the repository. Both record the experiment config SHA-256, split manifest SHA-256, canonical assignment SHA-256, project commit, seed, software environment, and whether AMP was enabled. Absolute local paths are not stored. The CLI refuses training from a dirty Git worktree. After each epoch, `last.pt` stores the full training and RNG state plus the best-checkpoint digest. An interrupted run resumes only when all bindings match; `last.pt` is removed after `metrics.json` is published.

Run from the repository root with local artifact paths:

```powershell
conda run -n na-lmscnet python code/scripts/train_baseline.py `
  --config code/configs/experiments/cnn2_radioml_2016_10a_selected.yml `
  --hdf5 <path-to-RML2016.10a.h5> `
  --conversion-manifest <path-to-conversion-manifest> `
  --split-manifest <path-to-split-manifest> `
  --leakage-audit <path-to-leakage-audit> `
  --output-dir <new-empty-external-directory>
```

Use the same command with `--resume` when the external output directory contains a valid `last.pt` from an interrupted run. Do not use `--resume` for a fresh or completed directory.
