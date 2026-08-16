# Controlled Revision Protocol

## Research question

The revision asks what equal weighting, learned global static preference, parameter count,
and input-conditioned gate assignment can explain under one fixed RadioML 2016.10A
protocol. It does not treat content adaptivity as an established independent contribution.

## Control matrix

| Contrast | Control | Estimand | Allowed interpretation |
| --- | --- | --- | --- |
| C1 | S1-static | S2 minus learned-static fusion | Whether a robust advantage over the evaluated learned-static control was established |
| C2a | S2-mean | Frozen-checkpoint mean-gate replacement | Sensitivity of a trained S2 checkpoint to removal of sample-varying gates |
| C2b | S2-shuffled | Frozen-checkpoint batch-local gate reassignment | Bounded dependence on input-conditioned gate assignment under the implemented shuffle |
| C3 | S1-wide-static | Parameter-count-matched static control | Whether parameter count alone was sufficient to explain the S2 result |
| C4 | SKNet-1D and AFNet adaptations | Direct adaptive-fusion neighbors | Competitiveness against the evaluated project adaptations only |

C1 and C4 were unresolved. C2a and C2b are destructive post-training interventions, not
retrained model comparisons. C3 supports only the statement that a parameter-count-only
explanation was insufficient.

## Frozen identifiers

- Dataset: RadioML 2016.10A, obtained separately.
- Split assignment SHA-256: `0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941`.
- Split ranking seed: `2026`.
- Training seeds: `13, 37, 73, 101, 137`.
- Shuffled-gate seeds: `13, 37, 73, 101, 137, 211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511, 1601`.
- Low-SNR reporting range: `{-10, -8, -6, -4, -2, 0}` dB.
- R6 checkpoint rule: fixed epoch 100; no validation-based early stopping or checkpoint selection.

## Configuration map

- `code/configs/revision/phase_r0.yml`: historical cutoff and test-isolation boundary.
- `code/configs/revision/phase_r1.yml`: controls, parameter matching, and permutation seeds.
- `code/configs/revision/phase_r6.yml`: fixed-epoch sensitivity protocol and validation freeze.
- `code/configs/experiments/revision_r2_*`: original controlled validation configurations.
- `code/configs/experiments/revision_r6_*`: fixed-epoch validation sensitivity configurations.

Historical absolute paths inside frozen YAML files are provenance records, not required
installation paths. Users must provide their own external dataset and artifact locations.

## Statistical scripts

- `code/scripts/run_r2_primary_contrasts.py`
- `code/scripts/run_r6_fixed_epoch_contrasts.py`
- `code/scripts/summarize_r2_validation.py`
- `code/scripts/summarize_r6_fixed_epoch_validation.py`
- `code/scripts/audit_r2_intervention_validity.py`
- `code/scripts/generate_r6_validation_freeze.py`

The paired bootstrap operates within the fixed split and finite training-seed set. Its
intervals do not establish uncertainty across alternative splits, datasets, or OTA capture
conditions. The shuffled-gate interval is a permutation interval, not a bootstrap CI.

## Intervention boundary

The implemented shuffle is batch-local and repeats the size-specific permutation pattern.
The audited mapping retained about 95.8% same-modulation pairing and 33.8% same-SNR
pairing. It therefore does not test arbitrary cross-sample or cross-modulation matching.

## Test isolation

The RadioML 2016.10A test was consumed before the controlled revision and is permanently
locked. RadioML 2018.01A test was ruled ineligible for a new confirmation. This release
does not authorize construction, access, replay, slicing, re-bootstrap, or evaluation of
any test dataset for the revision hypotheses.
