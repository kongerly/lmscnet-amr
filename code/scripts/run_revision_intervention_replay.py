"""Phase R2 intervention replays on the frozen S2 checkpoints.

S2-aligned, S2-mean and S2-shuffled reuse the five frozen S2 checkpoints from
the historical validation queue; no model is retrained. S2-mean fits a fixed
average gate on the train split only. S2-shuffled reassigns sample gates with
the pre-frozen permutation seeds, preserving gate marginals per batch.

This script is validation-only: test construction stays forbidden and the
frozen RadioML 2016.10A test split is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.models import LMSCNetS2, LMSCNetS2Mean, LMSCNetS2Shuffled  # noqa: E402

PERMUTATION_SEEDS = [13, 37, 73, 101, 137, 211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511, 1601]

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


class ReplayError(RuntimeError):
    """Raised when an intervention replay violates the frozen protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(checkpoint_path: Path, expected_seed: int, expected_split_sha: str) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != "lmscnet_s2":
        raise ReplayError(f"Expected an S2 checkpoint, got {checkpoint['model_name']}")
    bindings = checkpoint["bindings"]
    if int(bindings["seed"]) != expected_seed:
        raise ReplayError(f"Seed mismatch: {bindings['seed']} != {expected_seed}")
    if str(bindings["split_manifest_sha256"]) != expected_split_sha:
        raise ReplayError("Split manifest does not match the frozen S2 queue")
    if bindings["data_protocol"]["preprocessing_mode"] != "per_sample_max_abs":
        raise ReplayError("Unexpected preprocessing protocol in the frozen checkpoint")
    return checkpoint


def _collect_gates(model: LMSCNetS2, loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    collected: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            weights = model(batch["iq"].to(device))["scale_weights"]
            collected.append(weights.cpu())
    return torch.cat(collected, dim=0)


def _run_model(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    modulation: list[np.ndarray] = []
    snr: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            outputs = model(iq)
            batch_logits = outputs["logits"].float().cpu().numpy()
            sample_ids.extend(batch["sample_id"])
            predictions.append(batch_logits.argmax(axis=1).astype(np.int8))
            targets.append(batch["modulation"].numpy().astype(np.int8))
            modulation.append(batch["modulation"].numpy().astype(np.int8))
            snr.append(batch["snr"].numpy().astype(np.int8))
            logits.append(batch_logits.astype(np.float32))
    return {
        "sample_ids": np.asarray(sample_ids),
        "predictions": np.concatenate(predictions),
        "targets": np.concatenate(targets),
        "modulation": np.concatenate(modulation),
        "snr_db": np.concatenate(snr),
        "logits": np.concatenate(logits),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--s2-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73, 101, 137])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve(strict=True)
    split_sha = _sha256_file(args.split_manifest)
    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    device = torch.device(args.device)
    permutation_digests: dict[str, str] = {}
    results: list[dict[str, object]] = []
    with (
        RadioML2016HDF5Dataset(split="train", **common) as train_dataset,
        RadioML2016HDF5Dataset(split="validation", **common) as validation_dataset,
    ):
        if train_dataset.assignment_sha256 != validation_dataset.assignment_sha256:
            raise ReplayError("Train and validation assignments differ")
        assignment_sha = train_dataset.assignment_sha256
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=256, shuffle=False)
        for seed in args.seeds:
            checkpoint_path = args.s2_checkpoint_dir / f"lmscnet_s2-seed-{seed}" / "best.pt"
            checkpoint = _load_checkpoint(checkpoint_path, seed, split_sha)
            checkpoint_sha = _sha256_file(checkpoint_path)

            aligned = LMSCNetS2(num_classes=11, dropout=0.2, expansion=1.25)
            aligned.load_state_dict(checkpoint["model_state_dict"])
            aligned_pred = _run_model(aligned.to(device), validation_loader, device)
            aligned_out = output_dir / f"lmscnet_s2_aligned-seed-{seed}.npz"
            np.savez_compressed(aligned_out, **aligned_pred)
            results.append(
                {
                    "run": "s2_aligned",
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_sha,
                    "output": aligned_out.name,
                }
            )

            mean_model = LMSCNetS2Mean(num_classes=11, dropout=0.2, expansion=1.25)
            mean_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            mean_model.fit_mean_gate(train_loader)
            mean_gate_std = float(mean_model.mean_gate.std())
            mean_gate_rows = mean_model.mean_gate.numpy().round(4).tolist()
            mean_out = output_dir / f"lmscnet_s2_mean-seed-{seed}.npz"
            np.savez_compressed(mean_out, **_run_model(mean_model.to(device), validation_loader, device))
            results.append(
                {
                    "run": "s2_mean",
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_sha,
                    "mean_gate_fitted_on": "train_only",
                    "mean_gate_std": mean_gate_std,
                    "mean_gate_rows": mean_gate_rows,
                    "output": mean_out.name,
                }
            )

            for permutation_seed in PERMUTATION_SEEDS:
                shuffled_model = LMSCNetS2Shuffled(num_classes=11, dropout=0.2, expansion=1.25, permutation_seed=permutation_seed)
                shuffled_model.load_state_dict(checkpoint["model_state_dict"])
                shuffled_out = output_dir / f"lmscnet_s2_shuffled-seed-{seed}-perm-{permutation_seed}.npz"
                np.savez_compressed(shuffled_out, **_run_model(shuffled_model.to(device), validation_loader, device))
                permutation_digests[f"s2-{seed}-perm-{permutation_seed}"] = shuffled_model.last_permutation_hash
            results.append(
                {
                    "run": "s2_shuffled",
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_sha,
                    "permutation_seed_count": len(PERMUTATION_SEEDS),
                    "permutation_seeds": PERMUTATION_SEEDS,
                }
            )
    manifest = {
        "schema_version": 1,
        "purpose": "phase_r2_intervention_replay",
        "s2_checkpoint_commit": "f5760d85ff0bbcf28b1f6005f3ef5dad1e615de6",
        "split_manifest_sha256": split_sha,
        "assignment_sha256": assignment_sha,
        "seeds": args.seeds,
        "permutation_seeds": PERMUTATION_SEEDS,
        "test_accessed": False,
        "results": results,
        "permutation_digests": permutation_digests,
    }
    manifest_path = output_dir / "replay-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "replay_complete", "runs": len(results), "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
