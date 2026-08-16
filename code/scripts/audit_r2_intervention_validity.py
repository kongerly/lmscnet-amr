"""Phase R2.5 intervention-validity and auditability audit.

Runs on the frozen S2 checkpoints and the frozen train/validation splits only:

1. aligned gate_override identity: with gate_override equal to the aligned
   scale weights, logits, predictions and scale weights must be numerically
   identical to the un-overridden S2 forward.
2. independent mean-gate recomputation on the train split only, cross-checked
   against the values recorded in the Phase R2 replay manifest, plus a
   block-wise replacement diagnosis of the S2-mean drop.
3. complete sample-level A-to-B gate mapping manifest for S2-shuffled,
   including fixed points, same-modulation/SNR pairing proportions and
   gate mismatch, with every recorded permutation digest verified.

This is validation-only. The frozen RadioML 2016.10A test split is never
opened and the report carries test_accessed=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from na_lmscnet.models import LMSCNetS2  # noqa: E402
from na_lmscnet.models.final_lmscnet import NUM_FINAL_BLOCKS  # noqa: E402

PERMUTATION_SEEDS = [13, 37, 73, 101, 137, 211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511, 1601]
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)
REPLAY_BATCH_SIZE = 256

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


class AuditError(RuntimeError):
    """Raised when the intervention-validity audit cannot be completed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{field} must contain a JSON object")
    return value


def _load_checkpoint(checkpoint_path: Path, expected_seed: int, expected_split_sha: str) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != "lmscnet_s2":
        raise AuditError(f"Expected an S2 checkpoint, got {checkpoint['model_name']}")
    bindings = checkpoint["bindings"]
    if int(bindings["seed"]) != expected_seed:
        raise AuditError(f"Seed mismatch: {bindings['seed']} != {expected_seed}")
    if str(bindings["split_manifest_sha256"]) != expected_split_sha:
        raise AuditError("Split manifest does not match the frozen S2 queue")
    if bindings["data_protocol"]["preprocessing_mode"] != "per_sample_max_abs":
        raise AuditError("Unexpected preprocessing protocol in the frozen checkpoint")
    return checkpoint


def _aligned_forward(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    logits: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    modulation: list[np.ndarray] = []
    snr: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["iq"].to(device))
            sample_ids.extend(batch["sample_id"])
            logits.append(outputs["logits"].float().cpu().numpy().astype(np.float32))
            weights.append(outputs["scale_weights"].float().cpu().numpy().astype(np.float32))
            modulation.append(batch["modulation"].numpy().astype(np.int8))
            snr.append(batch["snr"].numpy().astype(np.int8))
    return {
        "sample_ids": np.asarray(sample_ids),
        "logits": np.concatenate(logits),
        "scale_weights": np.concatenate(weights),
        "modulation": np.concatenate(modulation),
        "snr_db": np.concatenate(snr),
    }


def _override_forward(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    gate_override: np.ndarray,
) -> dict[str, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    with torch.no_grad():
        start = 0
        for batch in loader:
            size = len(batch["iq"])
            override = torch.from_numpy(gate_override[start : start + size])
            outputs = model(batch["iq"].to(device), override.to(device))
            logits.append(outputs["logits"].float().cpu().numpy().astype(np.float32))
            weights.append(outputs["scale_weights"].float().cpu().numpy().astype(np.float32))
            start += size
    return {
        "logits": np.concatenate(logits),
        "scale_weights": np.concatenate(weights),
    }


def _accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(logits.argmax(axis=1) == targets))


def _low_snr_accuracy(logits: np.ndarray, targets: np.ndarray, snr_db: np.ndarray) -> float:
    mask = np.isin(snr_db, LOW_SNR_VALUES)
    if not mask.any():
        return float("nan")
    return float(np.mean(logits[mask].argmax(axis=1) == targets[mask]))


def _macro_f1(logits: np.ndarray, targets: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(targets, logits.argmax(axis=1), average="macro", zero_division=0))


def _fit_mean_gate_independent(
    model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, int]:
    model.eval()
    totals_f32 = np.zeros((NUM_FINAL_BLOCKS, 3), dtype=np.float32)
    totals_f64 = np.zeros((NUM_FINAL_BLOCKS, 3), dtype=np.float64)
    count = 0
    with torch.no_grad():
        for batch in train_loader:
            weights = model(batch["iq"].to(device))["scale_weights"].float().cpu().numpy()
            totals_f32 += weights.sum(axis=0).astype(np.float32)
            totals_f64 += weights.sum(axis=0).astype(np.float64)
            count += int(weights.shape[0])
    if count <= 0:
        raise AuditError("Train loader yielded no samples")
    return totals_f32 / count, totals_f64 / count, count


def _fit_mean_gate_cpu(
    model: torch.nn.Module, train_loader: torch.utils.data.DataLoader
) -> np.ndarray:
    cpu_model = type(model)(num_classes=11, dropout=0.2, expansion=1.25)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()
    totals = torch.zeros((NUM_FINAL_BLOCKS, 3), dtype=torch.float32)
    count = 0
    with torch.no_grad():
        for batch in train_loader:
            weights = cpu_model(batch["iq"])["scale_weights"].float()
            totals += weights.sum(dim=0).to(totals)
            count += int(weights.shape[0])
    if count <= 0:
        raise AuditError("Train loader yielded no samples")
    return (totals / count).numpy()


def _batch_permutation(batch_size: int, permutation_seed: int, device: torch.device) -> tuple[np.ndarray, str]:
    generator = torch.Generator(device=device.type).manual_seed(int(permutation_seed))
    permutation = torch.randperm(batch_size, generator=generator, device=device).detach().cpu()
    digest = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
    return permutation.numpy().astype(np.int32), digest


def _build_shuffled_mapping(
    validation_length: int, batch_size: int, permutation_seed: int, device: torch.device
) -> dict[str, Any]:
    full_batches = validation_length // batch_size
    last_batch_size = validation_length - full_batches * batch_size
    full_permutation, digest_full_batch = _batch_permutation(batch_size, permutation_seed, device)
    last_permutation, digest_last_batch = (
        _batch_permutation(last_batch_size, permutation_seed, device) if last_batch_size > 0 else (None, "")
    )
    batch_count = full_batches + (1 if last_batch_size > 0 else 0)
    sizes = np.full(batch_count, batch_size, dtype=np.int64)
    if last_batch_size > 0:
        sizes[-1] = last_batch_size
    receiver_batch = np.repeat(np.arange(batch_count, dtype=np.int16), sizes)
    receiver_pos = np.concatenate([np.arange(size, dtype=np.int16) for size in sizes])
    donor_pos = np.empty(receiver_batch.shape, dtype=np.int32)
    for batch_index in range(batch_count):
        mask = receiver_batch == batch_index
        donor_pos[mask] = full_permutation if batch_index < full_batches else last_permutation
    donor_global = receiver_batch.astype(np.int64) * batch_size + donor_pos
    return {
        "batch_index": receiver_batch,
        "batch_position": receiver_pos,
        "donor_position": donor_pos.astype(np.int16),
        "donor_global_index": donor_global.astype(np.int32),
        "full_batch_size": batch_size,
        "full_batch_count": full_batches,
        "last_batch_size": last_batch_size,
        "digest_full_batch": digest_full_batch,
        "digest_last_batch": digest_last_batch,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--s2-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73, 101, 137])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=REPLAY_BATCH_SIZE)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size != REPLAY_BATCH_SIZE:
        raise AuditError(f"Audit must reuse the replay batch size {REPLAY_BATCH_SIZE}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise AuditError("Output must remain outside the repository")
    split_sha = _sha256_file(args.split_manifest)
    replay_manifest = _load_json(args.replay_dir / "replay-manifest.json", "replay manifest")
    if replay_manifest.get("test_accessed") is not False:
        raise AuditError("Replay manifest reports test access")
    recorded_mean_rows: dict[int, list[list[float]]] = {}
    for result in replay_manifest["results"]:
        if result["run"] == "s2_mean":
            recorded_mean_rows[int(result["seed"])] = result["mean_gate_rows"]
    recorded_digests: dict[str, str] = {
        key: str(digest) for key, digest in replay_manifest["permutation_digests"].items()
    }

    device = torch.device(args.device)
    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "phase_r2_5_intervention_validity_audit",
        "test_accessed": False,
        "s2_checkpoint_commit": "f5760d85ff0bbcf28b1f6005f3ef5dad1e615de6",
        "split_manifest_sha256": split_sha,
        "replay_manifest_sha256": _sha256_file(args.replay_dir / "replay-manifest.json"),
        "seeds": args.seeds,
        "permutation_seeds": PERMUTATION_SEEDS,
        "batch_size": args.batch_size,
        "device": args.device,
        "seeds_report": {},
        "permutation_mapping": {},
        "passed": False,
    }
    evidence: list[dict[str, Any]] = []

    with (
        RadioML2016HDF5Dataset(split="train", **common) as train_dataset,
        RadioML2016HDF5Dataset(split="validation", **common) as validation_dataset,
    ):
        if train_dataset.assignment_sha256 != validation_dataset.assignment_sha256:
            raise AuditError("Train and validation assignments differ")
        report["assignment_sha256"] = validation_dataset.assignment_sha256
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        validation_length = len(validation_dataset)
        expected_train_count = len(train_dataset)

        aligned_by_seed: dict[int, dict[str, np.ndarray]] = {}
        models_by_seed: dict[int, torch.nn.Module] = {}
        for seed in args.seeds:
            checkpoint_path = args.s2_checkpoint_dir / f"lmscnet_s2-seed-{seed}" / "best.pt"
            checkpoint = _load_checkpoint(checkpoint_path, seed, split_sha)
            checkpoint_sha = _sha256_file(checkpoint_path)

            model = LMSCNetS2(num_classes=11, dropout=0.2, expansion=1.25)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            models_by_seed[seed] = model

            aligned = _aligned_forward(model, validation_loader, device)
            aligned_by_seed[seed] = aligned
            recorded_npz = np.load(args.replay_dir / f"lmscnet_s2_aligned-seed-{seed}.npz")
            recorded_logits = recorded_npz["logits"].astype(np.float32)
            recorded_predictions = recorded_npz["predictions"].astype(np.int64)
            replay_logits_max_diff = float(np.max(np.abs(aligned["logits"] - recorded_logits)))
            replay_prediction_mismatch = int(np.sum(aligned["logits"].argmax(axis=1) != recorded_predictions))

            overridden = _override_forward(model, validation_loader, device, aligned["scale_weights"])
            identity_logits_max_diff = float(np.max(np.abs(overridden["logits"] - aligned["logits"])))
            identity_weights_max_diff = float(np.max(np.abs(overridden["scale_weights"] - aligned["scale_weights"])))
            identity_prediction_mismatch = int(
                np.sum(overridden["logits"].argmax(axis=1) != aligned["logits"].argmax(axis=1))
            )
            identity_logits_exact = bool(np.array_equal(overridden["logits"], aligned["logits"]))
            identity_weights_exact = bool(np.array_equal(overridden["scale_weights"], aligned["scale_weights"]))

            mean_cpu = _fit_mean_gate_cpu(model, train_loader)
            mean_gpu, mean_f64, mean_count = _fit_mean_gate_independent(model, train_loader, device)
            recorded_rows = np.asarray(recorded_mean_rows[seed], dtype=np.float32).round(4)
            mean_cpu_rows_max_diff = float(np.max(np.abs(mean_cpu.round(4) - recorded_rows)))
            mean_gpu_rows_max_diff = float(np.max(np.abs(mean_gpu.round(4) - recorded_rows)))
            mean_gpu_vs_cpu_max_diff = float(np.max(np.abs(mean_gpu - mean_cpu)))
            mean_f64_vs_f32_max_diff = float(np.max(np.abs(mean_f64.astype(np.float32) - mean_gpu)))
            recorded_std = float(np.asarray(recorded_mean_rows[seed], dtype=np.float32).std())
            recomputed_std = float(mean_cpu.std())

            with torch.no_grad():
                mean_repro_logits = _override_forward(
                    model, validation_loader, device, np.repeat(mean_cpu[np.newaxis, :, :], validation_length, axis=0)
                )["logits"]
            recorded_mean_npz = np.load(args.replay_dir / f"lmscnet_s2_mean-seed-{seed}.npz")
            mean_replay_max_diff = float(
                np.max(np.abs(mean_repro_logits - recorded_mean_npz["logits"].astype(np.float32)))
            )
            mean_prediction_mismatch = int(
                np.sum(mean_repro_logits.argmax(axis=1) != recorded_mean_npz["predictions"].astype(np.int64))
            )

            aligned_accuracy = _accuracy(aligned["logits"], aligned["modulation"])
            aligned_low_snr = _low_snr_accuracy(aligned["logits"], aligned["modulation"], aligned["snr_db"])
            aligned_macro_f1 = _macro_f1(aligned["logits"], aligned["modulation"])
            full_mean_override = np.repeat(mean_cpu[np.newaxis, :, :], validation_length, axis=0)
            full_mean_out = _override_forward(model, validation_loader, device, full_mean_override)
            full_mean_logits = full_mean_out["logits"]
            full_mean_accuracy = _accuracy(full_mean_logits, aligned["modulation"])
            full_mean_low_snr = _low_snr_accuracy(full_mean_logits, aligned["modulation"], aligned["snr_db"])
            full_mean_mismatch = int(
                np.sum(full_mean_logits.argmax(axis=1) != np.load(args.replay_dir / f"lmscnet_s2_mean-seed-{seed}.npz")["predictions"].astype(np.int64))
            )
            block_diagnostics: list[dict[str, Any]] = []
            for block_index in range(NUM_FINAL_BLOCKS):
                override = aligned["scale_weights"].copy()
                override[:, block_index, :] = mean_cpu[block_index]
                block_out = _override_forward(model, validation_loader, device, override)
                block_accuracy = _accuracy(block_out["logits"], aligned["modulation"])
                block_low_snr = _low_snr_accuracy(block_out["logits"], aligned["modulation"], aligned["snr_db"])
                block_diagnostics.append(
                    {
                        "block": block_index,
                        "accuracy": block_accuracy,
                        "low_snr_accuracy": block_low_snr,
                        "accuracy_delta_pp": (block_accuracy - aligned_accuracy) * 100,
                        "low_snr_delta_pp": (block_low_snr - aligned_low_snr) * 100,
                    }
                )
            aligned_mean_gate = aligned["scale_weights"].mean(axis=0)
            train_val_gate_deviation = float(np.max(np.abs(mean_cpu - aligned_mean_gate)))

            recorded_checkpoint_sha = replay_manifest["results"][
                next(
                    index
                    for index, result in enumerate(replay_manifest["results"])
                    if result["seed"] == seed and result["run"] == "s2_aligned"
                )
            ]["checkpoint_sha256"]
            seed_report = {
                "seed": seed,
                "checkpoint_sha256": checkpoint_sha,
                "recorded_checkpoint_sha256": recorded_checkpoint_sha,
                "aligned_override_identity": {
                    "logits_max_abs_diff": identity_logits_max_diff,
                    "scale_weights_max_abs_diff": identity_weights_max_diff,
                    "prediction_mismatch_count": identity_prediction_mismatch,
                    "logits_exact_equal": identity_logits_exact,
                    "scale_weights_exact_equal": identity_weights_exact,
                },
                "replay_cross_check": {
                    "logits_max_abs_diff": replay_logits_max_diff,
                    "prediction_mismatch_count": replay_prediction_mismatch,
                },
                "mean_gate_independent_recompute": {
                    "train_sample_count": mean_count,
                    "expected_train_sample_count": expected_train_count,
                    "eval_mode_during_fit": True,
                    "accumulation_dtype": "float32_buffer_float64_reference",
                    "fit_device_matches_recorded_protocol": "cpu",
                    "cpu_fit_recorded_rows_max_abs_diff_after_round4": mean_cpu_rows_max_diff,
                    "gpu_fit_recorded_rows_max_abs_diff_after_round4": mean_gpu_rows_max_diff,
                    "gpu_vs_cpu_max_abs_diff": mean_gpu_vs_cpu_max_diff,
                    "float64_vs_float32_max_abs_diff": mean_f64_vs_f32_max_diff,
                    "recomputed_mean_gate_rows_cpu_round4": mean_cpu.round(4).tolist(),
                    "recorded_mean_gate_rows": recorded_rows.tolist(),
                    "recomputed_std": recomputed_std,
                    "recorded_std": recorded_std,
                    "mean_replay_logits_max_abs_diff": mean_replay_max_diff,
                    "mean_replay_prediction_mismatch_count": mean_prediction_mismatch,
                },
                "mean_gate_block_diagnosis": {
                    "aligned_accuracy": aligned_accuracy,
                    "aligned_low_snr_accuracy": aligned_low_snr,
                    "aligned_macro_f1": aligned_macro_f1,
                    "full_mean_gate_accuracy": full_mean_accuracy,
                    "full_mean_gate_low_snr_accuracy": full_mean_low_snr,
                    "full_mean_replay_prediction_mismatch_count": full_mean_mismatch,
                    "train_vs_validation_mean_gate_max_abs_diff": train_val_gate_deviation,
                    "per_block": block_diagnostics,
                },
            }
            report["seeds_report"][str(seed)] = seed_report
            evidence.append({"event": "seed_audit", "seed": seed, **seed_report})

        order_metadata = aligned_by_seed[args.seeds[0]]
        receiver_modulation = order_metadata["modulation"]
        receiver_snr = order_metadata["snr_db"]
        receiver_sample_ids = order_metadata["sample_ids"]

        for seed in args.seeds:
            aligned = aligned_by_seed[seed]
            aligned_weights = aligned["scale_weights"]
            model = models_by_seed[seed]
            for permutation_seed in PERMUTATION_SEEDS:
                mapping = _build_shuffled_mapping(validation_length, args.batch_size, permutation_seed, device)
                donor_global = mapping["donor_global_index"]
                assigned = aligned_weights[donor_global]
                gate_mean_abs_diff = np.abs(aligned_weights - assigned).mean(axis=2)
                fixed_point = mapping["batch_position"] == mapping["donor_position"]
                same_modulation = receiver_modulation[donor_global] == receiver_modulation
                same_snr = receiver_snr[donor_global] == receiver_snr
                same_stratum = same_modulation & same_snr
                argmax_flip = (assigned.argmax(axis=2) != aligned_weights.argmax(axis=2)).any(axis=1)
                donor_sample_ids = receiver_sample_ids[donor_global]
                mapping_npz = {
                    "receiver_sample_id": receiver_sample_ids,
                    "donor_sample_id": donor_sample_ids,
                    "batch_index": mapping["batch_index"],
                    "batch_position": mapping["batch_position"],
                    "donor_position": mapping["donor_position"],
                    "donor_global_index": donor_global,
                    "fixed_point": fixed_point,
                    "same_modulation": same_modulation,
                    "same_snr": same_snr,
                    "same_stratum": same_stratum,
                    "argmax_flip": argmax_flip,
                    "gate_mean_abs_diff": gate_mean_abs_diff.astype(np.float16),
                    "assigned_scale_weights": assigned.astype(np.float16),
                    "aligned_scale_weights": aligned_weights.astype(np.float16),
                }
                mapping_dir = output_dir / "shuffled-mapping"
                mapping_dir.mkdir(parents=True, exist_ok=True)
                mapping_path = mapping_dir / f"lmscnet_s2_shuffled-seed-{seed}-perm-{permutation_seed}-mapping.npz"
                np.savez_compressed(mapping_path, **mapping_npz)

                recorded_key = f"s2-{seed}-perm-{permutation_seed}"
                recorded_digest = recorded_digests[recorded_key]
                digest_matches = recorded_digest == mapping["digest_last_batch"]

                shuffled_repro_logits = _override_forward(model, validation_loader, device, assigned)["logits"]
                recorded_shuffled_npz = np.load(
                    args.replay_dir / f"lmscnet_s2_shuffled-seed-{seed}-perm-{permutation_seed}.npz"
                )
                shuffled_replay_max_diff = float(
                    np.max(np.abs(shuffled_repro_logits - recorded_shuffled_npz["logits"].astype(np.float32)))
                )
                shuffled_prediction_mismatch = int(
                    np.sum(
                        shuffled_repro_logits.argmax(axis=1)
                        != recorded_shuffled_npz["predictions"].astype(np.int64)
                    )
                )

                permutation_report = {
                    "seed": seed,
                    "permutation_seed": permutation_seed,
                    "recorded_digest": recorded_digest,
                    "recomputed_last_batch_digest": mapping["digest_last_batch"],
                    "digest_match": digest_matches,
                    "sample_count": int(validation_length),
                    "full_batch_count": int(mapping["full_batch_count"]),
                    "full_batch_size": int(mapping["full_batch_size"]),
                    "last_batch_size": int(mapping["last_batch_size"]),
                    "batch_local_fixed_seed_repeats_same_permutation_in_full_batches": True,
                    "fixed_point_count": int(fixed_point.sum()),
                    "fixed_point_fraction": float(fixed_point.mean()),
                    "same_modulation_count": int(same_modulation.sum()),
                    "same_modulation_fraction": float(same_modulation.mean()),
                    "same_snr_count": int(same_snr.sum()),
                    "same_snr_fraction": float(same_snr.mean()),
                    "same_stratum_count": int(same_stratum.sum()),
                    "same_stratum_fraction": float(same_stratum.mean()),
                    "argmax_flip_fraction": float(argmax_flip.mean()),
                    "gate_mean_abs_diff_mean": float(gate_mean_abs_diff.mean()),
                    "gate_mean_abs_diff_max": float(gate_mean_abs_diff.max()),
                    "gate_changed_fraction": float(np.mean(gate_mean_abs_diff.max(axis=1) > 1e-6)),
                    "mapping_file": str(mapping_path.relative_to(output_dir)),
                    "replay_reconstruction": {
                        "logits_max_abs_diff": shuffled_replay_max_diff,
                        "prediction_mismatch_count": shuffled_prediction_mismatch,
                    },
                }
                report["permutation_mapping"].setdefault(str(seed), {})[
                    str(permutation_seed)
                ] = permutation_report
                evidence.append({"event": "permutation_mapping", **permutation_report})

    identity_passed = all(
        report["seeds_report"][str(seed)]["aligned_override_identity"]["prediction_mismatch_count"] == 0
        and report["seeds_report"][str(seed)]["aligned_override_identity"]["logits_max_abs_diff"] <= 1e-6
        and report["seeds_report"][str(seed)]["aligned_override_identity"]["scale_weights_max_abs_diff"] <= 1e-6
        for seed in args.seeds
    )
    replay_cross_check_passed = all(
        report["seeds_report"][str(seed)]["replay_cross_check"]["prediction_mismatch_count"] == 0
        and report["seeds_report"][str(seed)]["replay_cross_check"]["logits_max_abs_diff"] <= 1e-6
        for seed in args.seeds
    )
    mean_gate_passed = all(
        report["seeds_report"][str(seed)]["mean_gate_independent_recompute"][
            "cpu_fit_recorded_rows_max_abs_diff_after_round4"
        ]
        <= 1e-6
        and report["seeds_report"][str(seed)]["mean_gate_independent_recompute"][
            "mean_replay_prediction_mismatch_count"
        ]
        == 0
        and report["seeds_report"][str(seed)]["mean_gate_independent_recompute"]["train_sample_count"]
        == report["seeds_report"][str(seed)]["mean_gate_independent_recompute"]["expected_train_sample_count"]
        for seed in args.seeds
    )
    mapping_passed = all(
        report["permutation_mapping"][str(seed)][str(permutation_seed)]["digest_match"]
        and report["permutation_mapping"][str(seed)][str(permutation_seed)]["sample_count"] == validation_length
        and report["permutation_mapping"][str(seed)][str(permutation_seed)]["replay_reconstruction"][
            "prediction_mismatch_count"
        ]
        == 0
        and report["permutation_mapping"][str(seed)][str(permutation_seed)]["replay_reconstruction"][
            "logits_max_abs_diff"
        ]
        <= 1e-6
        for seed in args.seeds
        for permutation_seed in PERMUTATION_SEEDS
    )
    report["checkmarks"] = {
        "aligned_override_identity": identity_passed,
        "replay_cross_check": replay_cross_check_passed,
        "mean_gate_independent_recompute": mean_gate_passed,
        "shuffled_mapping_and_digest": mapping_passed,
    }
    report["passed"] = all(report["checkmarks"].values())
    report["test_accessed"] = False
    report["audit_script_sha256"] = _sha256_file(Path(__file__).resolve())
    report["audit_project_commit"] = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "r25-intervention-validity-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "r25-intervention-validity-evidence.jsonl").open("w", encoding="utf-8") as stream:
        for line in evidence:
            stream.write(json.dumps(line, sort_keys=True) + "\n")

    summary = {
        "conclusion": "passed" if report["passed"] else "failed",
        "checkmarks": report["checkmarks"],
        "output_dir": str(output_dir),
        "test_accessed": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
