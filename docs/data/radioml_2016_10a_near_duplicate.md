# RADIOML 2016.10A Near-Duplicate Audit Design

## Scope And Non-Claims

This document freezes the near-duplicate leakage-audit design for the verified local RADIOML 2016.10A HDF5 artifact. The machine-readable contract is `code/configs/data/radioml_2016_10a_near_duplicate.yml`.

This block does not scan the 220000-sample artifact, generate a leakage report, generate a split, implement a dataset adapter, normalize or augment training data, or authorize model training. The production audit remains disabled until candidate-recall evidence, threshold calibration, and manual-review protocol are all complete. The reference scorer is a bounded mathematical fixture tool, not production evidence.

## Two Representations

Exact duplicates and transformed near duplicates are different questions and must not share a digest interpretation.

### Exact-Byte Reference

The exact reference hashes canonical little-endian float32 I/Q bytes in C order. It is the existing exact-duplicate check from the numeric audit. A digest appearing across proposed splits is a hard failure. It says nothing about phase rotation, circular shifts, normalization, resampling, noise, or tolerance-based similarity.

### Transformed Similarity View

For bounded fixtures, each sample maps to a complex sequence:

```text
z[t] = float32(i[t]) + j * float32(q[t])
```

The sequence must be finite and have positive RMS complex amplitude. The scorer divides by RMS amplitude and computes the maximum absolute circular cross-correlation over integer lags `0..127`:

```text
max_lag(abs(sum(conj(z_a[t]) * z_b[(t+lag) mod length]))) / (norm(z_a) * norm(z_b))
```

The reference arithmetic uses float64 after the source float32 values are loaded. This view is invariant to a global nonzero complex gain, including amplitude scaling and phase rotation, and to an integer circular time shift. It deliberately does not claim invariance to resampling, time scaling, conjugation, clipping, or additive noise. Those transformations require separate evidence and must not be silently folded into this contract.

The scorer is intentionally quadratic and fixture-bounded to at most 64 samples for exhaustive pairwise checks. Calibration may score its explicitly enumerated positive and negative pairs without taking their full Cartesian product. The scorer's presence proves the representation and numerical boundary, not production-scale candidate recall.

## Candidate Recall

The only accepted reference algorithm is exhaustive pairwise comparison on small fixtures. A production candidate generator may be introduced only with a versioned proof that every pair scoring at or above the calibrated threshold is retrieved. Blocking keys are currently `none`; a future index must provide an independently tested recall certificate against the exhaustive reference on adversarial fixtures.

Any unproven false negative is blocking. Approximate nearest-neighbor, locality-sensitive hashing, product quantization, or SNR/modulation buckets are not automatically safe: each can discard a transformed pair. They require measured recall, deterministic serialization, version binding, resource bounds, and failure tests before use.

## Threshold Calibration

No threshold is currently chosen. Calibration must use separate deterministic fixtures, not test samples:

- At least 128 positive pairs created by global nonzero complex gain and integer circular shifts, with a source float32 round trip before scoring.
- At least 1024 independently seeded nonmatching negative pairs using seed `2026`.
- Positive recall must be `1.0`; negative false-positive rate must be at most `0.001`.
- The calibration report must bind its fixture-generation contract, exact pair IDs, source and transform parameters, scorer implementation, environment, and report SHA-256.

The threshold must be selected from the calibration evidence and recorded as an exact finite decimal. A threshold that is merely plausible, borrowed from another signal domain, or tuned after viewing the real test set is invalid.

### Deterministic Fixture Evidence

The separate `radioml_2016_10a_near_duplicate_fixture_v1` contract uses SHA-256 counter-generated float32 I/Q samples and never reads RADIOML data. It constructs 128 gain-and-shift positive pairs and 1024 independently labelled negative pairs. The threshold rule floors the minimum positive score to 12 decimal places; the current result is `0.999999999999`, with positive recall `1.0` and negative false-positive rate `0.0` on this synthetic fixture.

A separate 64-sample fixture contains 16 known base/transform pairs and 32 unrelated samples. Exhaustive scoring of all 2016 unordered pairs retrieves exactly the 16 known pairs, with recall `1.0`, no false negatives, and no false positives at the fixture threshold. This proves the bounded reference implementation and fixture construction only. It is not a production-scale recall certificate and does not establish behavior under additive noise or any forbidden transform family.

## Pair Review

Candidate pairs are labelled `same_source_transform`, `unrelated`, or `ambiguous`. Any ambiguous pair blocks publication until a reviewer records the pair digest, decision, reviewer identity, UTC timestamp, and reason. Records contain relative IDs and hashes only; absolute local paths are forbidden. A cross-split candidate that cannot be resolved is treated as a failure, not as permission to continue.

The final audit report must bind the near-duplicate contract, source identities, HDF5 logical content, calibration report, candidate assignment, pair decisions, and software environment. It must record candidate counts, score distribution, threshold, false-negative/false-positive calibration evidence, unresolved count, and cross-split decisions in a deterministic schema.

## Publication And Gate

Audit outputs stay outside the repository. Publication is single-process, no-overwrite, same-directory transactional, with exclusive unpredictable temporary files, `fsync`, and an audit report as completion marker. A partial or missing report fails closed.

The current CLI verifies that `candidate_generation=reference_only`, calibration and review are `pending`, and production near-duplicate audit generation is disabled. The approved deterministic benchmark split may be generated with the unresolved global-audit limitation recorded explicitly. Validate from the repository root:

```powershell
conda run -n na-lmscnet python code/scripts/validate_near_duplicate_contract.py
conda run -n na-lmscnet python code/scripts/validate_near_duplicate_fixture.py
```

The next implementation block should prioritize the reproducible data path needed for baseline experiments while retaining the fail-closed production audit and test-isolation gates.
