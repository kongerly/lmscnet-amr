"""Phase R2 gate mechanism analysis on frozen checkpoints.

Reports gate entropy, sample variance, per-SNR and per-modulation branch
weight distributions, and collapse checks for S2-aligned (content gate) and
the neighbor channel-wise gates. Descriptive evidence only; it does not
replace the mean/shuffled interventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.models import (  # noqa: E402
    AFNetAdaptation,
    LMSCNetS2,
    SKNet1DAdaptation,
)

SNR_VALUES = list(range(-20, 19, 2))


class GateAnalysisError(ValueError):
    """Raised when gate analysis cannot be completed."""


def _entropy(weights: np.ndarray) -> float:
    """Mean Shannon entropy over samples, blocks and branches (per row)."""
    rows = weights.reshape(-1, weights.shape[-1])
    epsilon = 1e-12
    return float(np.mean(-np.sum(rows * np.log2(rows + epsilon), axis=1)))


def _collapse_fraction(weights: np.ndarray, threshold: float = 0.9) -> float:
    return float(np.mean(weights.max(axis=-1) >= threshold))


def _gather(model: torch.nn.Module, dataset: Any, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    weights_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    modulation_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["iq"].to(device))
            weights_parts.append(outputs["scale_weights"].cpu().numpy())
            snr_parts.append(batch["snr"].numpy())
            modulation_parts.append(batch["modulation"].numpy())
    return {
        "weights": np.concatenate(weights_parts, axis=0),
        "snr_db": np.concatenate(snr_parts).astype(np.int64),
        "modulation": np.concatenate(modulation_parts).astype(np.int64),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--s2-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise GateAnalysisError("Output must remain outside the repository")
    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml",
        "dataset_spec_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml",
        "conversion_contract_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml",
        "preprocessing": "per_sample_max_abs",
    }
    device = torch.device(args.device)
    checkpoint_path = args.s2_checkpoint_dir / "lmscnet_s2-seed-13" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    models: dict[str, torch.nn.Module] = {
        "s2_aligned": LMSCNetS2(num_classes=11, dropout=0.2, expansion=1.25),
        "sknet_1d": SKNet1DAdaptation(num_classes=11, dropout=0.2, expansion=1.25),
        "afnet": AFNetAdaptation(num_classes=11, dropout=0.2, expansion=1.25),
    }
    models["s2_aligned"].load_state_dict(checkpoint["model_state_dict"])
    with RadioML2016HDF5Dataset(split="validation", **common) as dataset:
        reports: dict[str, dict[str, Any]] = {}
        for name, model in models.items():
            gathered = _gather(model, dataset, device, args.batch_size)
            weights = gathered["weights"]
            snr_db = gathered["snr_db"]
            modulation = gathered["modulation"]
            branch_count = weights.shape[-1]
            per_block = []
            for block_index in range(weights.shape[1]):
                block_weights = weights[:, block_index, :]
                per_block.append(
                    {
                        "block": block_index,
                        "entropy_bits": _entropy(block_weights),
                        "mean_weight": block_weights.mean(axis=0).tolist(),
                        "std_weight": block_weights.std(axis=0).tolist(),
                        "collapse_fraction": _collapse_fraction(block_weights),
                    }
                )
            by_snr = {}
            for snr in SNR_VALUES:
                mask = snr_db == snr
                if not mask.any():
                    continue
                selected = weights[mask]
                by_snr[str(snr)] = {
                    "mean_weight": selected.mean(axis=(0, 1)).tolist(),
                    "std_weight": selected.std(axis=(0, 1)).tolist(),
                    "entropy_bits": _entropy(selected),
                }
            by_modulation = {}
            for modulation_index in range(11):
                mask = modulation == modulation_index
                if not mask.any():
                    continue
                selected = weights[mask]
                by_modulation[str(modulation_index)] = {
                    "mean_weight": selected.mean(axis=(0, 1)).tolist(),
                    "entropy_bits": _entropy(selected),
                }
            reports[name] = {
                "branch_count": branch_count,
                "sample_count": int(weights.shape[0]),
                "overall_entropy_bits": _entropy(weights),
                "overall_mean_weight": weights.mean(axis=(0, 1)).tolist(),
                "overall_std_weight": weights.std(axis=(0, 1)).tolist(),
                "collapse_fraction": _collapse_fraction(weights),
                "per_block": per_block,
                "by_snr": by_snr,
                "by_modulation": by_modulation,
            }
    report = {
        "schema_version": 1,
        "purpose": "phase_r2_gate_mechanism_analysis",
        "test_accessed": False,
        "models": reports,
        "note": "Descriptive evidence only; does not replace mean/shuffled interventions.",
    }
    output_dir.mkdir(parents=True)
    (output_dir / "r2-gate-mechanism.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "model_count": len(reports), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
