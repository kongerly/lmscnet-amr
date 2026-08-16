# RADIOML 2016.10A: Source and Integrity Protocol

This document records the provenance, license, expected structure, and local integrity workflow for the primary dataset. It does not redistribute the dataset.

## Authoritative Source

- Provider: DeepSig, Inc.
- Dataset page: <https://www.deepsig.ai/datasets/>
- Dataset label on the provider page: `RADIOML 2016.10A`
- Official archive name: `RML2016.10a.tar.bz2`
- Official download endpoint: <https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2>
- Source checked: 2026-08-08

The endpoint currently presents a contact and email-verification form. This repository does not automate or bypass that process. Download the archive through the provider's workflow and keep it outside the repository.

## License and Citation

DeepSig's dataset page states that all datasets it provides are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) (`CC BY-NC-SA 4.0`) and asks users to reference the dataset page or the relevant academic papers.

This project does not redistribute the archive. Users must independently comply with the license, including attribution, non-commercial use, and ShareAlike requirements where applicable.

Relevant foundational paper:

> Timothy J. O'Shea, Johnathan Corgan, and T. Charles Clancy, "Convolutional Radio Modulation Recognition Networks," arXiv:1602.04105v3, 2016. <https://arxiv.org/abs/1602.04105>

## Expected Dataset Grid

The historical generator at commit [`4ecf612`](https://github.com/radioML/dataset/tree/4ecf612cfbc5bfc80eb8b0dbe63ed685d0a73c44) directly specifies the following grid:

| Property | Expected value |
| --- | --- |
| Modulations | `8PSK`, `AM-DSB`, `AM-SSB`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`, `QAM64`, `QPSK`, `WBFM` |
| SNR values | `-20` through `18` dB in 2 dB steps |
| Samples per modulation/SNR cell | `1000` |
| Number of cells | `220` |
| Total samples | `220000` |
| Sample shape | `[2, 128]` |
| Generator dtype | `float32` |

The provider warns that the historical generation modules are not maintained. The generator repository itself has no detected repository license; the dataset license statement above comes from DeepSig's current official dataset page.

DeepSig currently labels these datasets as early academic research releases from 2016/2017, with known errata, and states that they are not used in current DeepSig products. Treat RADIOML 2016.10A as a legacy synthetic benchmark, not as evidence of performance on real over-the-air signals. The project will not claim real-world generalization from this dataset alone.

## Local Integrity Workflow

1. Complete the official download workflow yourself.
2. Save the archive outside this Git repository, preserving the filename `RML2016.10a.tar.bz2`.
3. From the repository root, run:

```powershell
conda run -n na-lmscnet python code/scripts/inventory_dataset.py `
  <data-dir>\RML2016.10a.tar.bz2 `
  --output <data-dir>\RML2016.10a.dataset-inventory.json
```

4. Preserve the generated JSON with the archive and record the archive SHA-256 in every experiment manifest.
5. Run the no-execution opcode scan and static schema validation described below before implementing a data adapter.
6. Do not deserialize the pickle directly. A later data-conversion block must retain the same strict trust boundary and validate numeric values before training.

The official page did not publish a checksum when checked on 2026-08-08. Therefore, this project records the digest of the exact locally acquired archive rather than pretending that an unverified canonical checksum exists. Digests from different downloads must not be assumed equal without evidence.

For the local archive used during development on 2026-08-08:

- Archive size: `228491318` bytes
- Archive SHA-256: `7a1603dd61e557f45b6e113dc0c59be02a14509b77856c31bbb324a993f7974c`
- Members: `RML2016.10a_dict.pkl` (`640919653` bytes) and `LICENSE.TXT` (`20993` bytes)
- `LICENSE.TXT` SHA-256: `8d37ad62e3fde7a8dadb7e4febc749ce6255952da94120fc629569d021cc2d90`

These values identify one local download; they are not an official provider checksum.

## Security Boundary

Python pickle payloads can execute arbitrary code during deserialization. The inventory command deliberately never calls `pickle.load`, never extracts archive members, rejects links and unsafe paths, and only computes the archive digest plus declared tar member metadata. A matching filename and plausible archive structure do not prove that a pickle payload is safe.

After an archive has been inventoried, the repository provides a second no-execution scanner for this legacy Python 2 protocol-0 payload:

```powershell
conda run -n na-lmscnet python code/scripts/inspect_pickle_payload.py `
  <data-dir>\RML2016.10a.tar.bz2 `
  --output <data-dir>\RML2016.10a.pickle-scan.json
```

The scanner treats Python 2 `STRING` values as opaque escaped bytes, records bounded opcode and literal metadata, flags global/reduction opcodes without executing them, and validates that the expected modulation/SNR literals are present. It does not call `pickle.load`, import referenced globals, create NumPy arrays, or write extracted payloads. This is a preliminary check only.

After the preliminary scan succeeds, run the stricter static schema validator:

```powershell
conda run -n na-lmscnet python code/scripts/validate_pickle_schema.py `
  <data-dir>\RML2016.10a.tar.bz2 `
  --output <data-dir>\RML2016.10a.pickle-schema.json
```

The validator interprets only the 13 protocol-0 opcode names observed in the payload. It symbolically recognizes exactly `numpy.core.multiarray._reconstruct`, `numpy.ndarray`, and `numpy.dtype`; it never imports or invokes those references. All other globals and opcodes are rejected. Stack depth, memo entries, tuple size, cell count, opcode count, decoded string size, archive members, source type, and trailing bytes are bounded or checked.

The local archive identified above passed this validation with the following result:

| Verified property | Local result |
| --- | --- |
| Root container | `dict` |
| Complete key grid | 11 modulations x 20 SNR values = 220 cells |
| Array shape per cell | `[1000, 2, 128]` |
| Samples per cell | `1000` |
| Total samples | `220000` |
| Dtype metadata | little-endian float32 (`<f4`) |
| Memory order | C-contiguous |
| Inline buffer per cell | `1024000` bytes |
| Total inline array bytes | `225280000` bytes |
| Parsed opcodes | `6196` |

This establishes the complete dictionary grid and structural array metadata for this exact archive. It does not itself inspect the decoded floating-point values.

## Numeric Quality Audit

After schema validation succeeds, audit the validated inline buffers:

```powershell
conda run -n na-lmscnet python code/scripts/audit_numeric_quality.py `
  <data-dir>\RML2016.10a.tar.bz2 `
  --output <data-dir>\RML2016.10a.numeric-audit.json
```

The audit retains the static pickle boundary. It creates a read-only NumPy `<f4` view over one already validated cell buffer at a time, computes bounded numeric summaries, and releases that buffer before proceeding. It also computes SHA-256 for each full cell and each raw 1024-byte sample. It does not call `pickle.load`, import or execute pickle globals, extract the archive, or write converted arrays.

The local archive produced the following audit result:

| Audited property | Local result |
| --- | --- |
| Dataset content SHA-256 | `bcaf1ea9bca18db5b5e179352b18504e6f92d1db7f4cf5b12673c2e3fba9aef9` |
| Float32 values inspected | `56320000` |
| NaN / +Inf / -Inf | `0 / 0 / 0` |
| Exact zero values | `0` |
| Zero-energy samples | `0` |
| Global minimum / maximum | `-0.154945537447929 / 0.16422912478447` |
| Global mean / RMS | `-0.000273587765620367 / 0.00604862194143634` |
| Mean I/Q power | `0.0000731716547808503` |
| I / Q DC means | `0.000069815295869156 / -0.000616990827109889` |
| Exact duplicate raw samples | `0` |
| Exact duplicate cell buffers | `0` |

The content digest is derived deterministically from sorted `(modulation, SNR, cell SHA-256)` records. Exact duplicate detection compares SHA-256 digests of the original 1024-byte float32 sample representation; it does not establish that tolerance-based, transformed, or near-duplicate samples are absent. The provider's known errata and synthetic-data bias also remain unaffected by these checks. No data split or training step may infer real over-the-air validity from this audit.
