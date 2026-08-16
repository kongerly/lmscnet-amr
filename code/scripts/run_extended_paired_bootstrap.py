"""Replay S2 and the strongest extended baseline on identical validation samples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.evaluation.core_ablation_multiseed_report import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
)
from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.engine import experiment_config_sha256  # noqa: E402
from run_final_paired_bootstrap import (  # noqa: E402
    _comparison,
    _load_json,
    _replay,
    _save_prediction,
    _sha256_file,
)

SEEDS = (13, 37, 73, 101, 137)
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)


def _paths(queue_root: Path, group: str, model: str, seed: int) -> dict[str, Path]:
    group_dir = queue_root / group
    run_id = f"{model}-seed-{seed}"
    return {
        "config": group_dir / "configs" / f"{run_id}.yml",
        "metrics": group_dir / run_id / "metrics.json",
        "checkpoint": group_dir / run_id / "best.pt",
    }


def _aligned(left: dict[str, Any], right: dict[str, Any]) -> None:
    for field in ("sample_ids", "targets", "modulation", "snr_db"):
        if not np.array_equal(np.asarray(left[field]), np.asarray(right[field])):
            raise ValueError(f"Validation sample alignment differs for {field}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2-queue-root", type=Path, required=True)
    parser.add_argument("--extended-queue-root", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--s2-split-manifest", type=Path, required=True)
    parser.add_argument("--extended-split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise ValueError("Bootstrap output must remain outside repository")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    s2_root = args.s2_queue_root.resolve(strict=True)
    ext_root = args.extended_queue_root.resolve(strict=True)
    s2_protocol = _load_json(s2_root / "queue-protocol.json", "S2 queue protocol")
    ext_protocol = _load_json(ext_root / "queue-protocol.json", "extended queue protocol")
    if s2_protocol.get("test_accessed") is not False or ext_protocol.get("test_accessed") is not False:
        raise ValueError("Training queue protocol accessed test")
    if s2_protocol.get("assignment_sha256") != ext_protocol.get("assignment_sha256"):
        raise ValueError("Training queue assignments differ")
    s2_replays: dict[int, dict[str, Any]] = {}
    ext_replays: dict[int, dict[str, Any]] = {}
    prediction_manifest: list[dict[str, Any]] = []
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        prediction_dir = temporary / "predictions"
        prediction_dir.mkdir()
        common = {
            "hdf5_path": args.hdf5,
            "conversion_manifest_path": args.conversion_manifest,
            "leakage_audit_path": args.leakage_audit,
            "split_contract_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml",
            "dataset_spec_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml",
            "conversion_contract_path": PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml",
            "preprocessing": "per_sample_max_abs",
        }
        with (
            RadioML2016HDF5Dataset(split="validation", split_manifest_path=args.s2_split_manifest, **common) as s2_dataset,
            RadioML2016HDF5Dataset(split="validation", split_manifest_path=args.extended_split_manifest, **common) as ext_dataset,
        ):
            if s2_dataset.assignment_sha256 != ext_dataset.assignment_sha256:
                raise ValueError("S2 and extended validation assignments differ")
            for seed in SEEDS:
                s2_paths = _paths(s2_root, "final-family-multiseed", "lmscnet_s2", seed)
                ext_paths = _paths(ext_root, "multiseed", "se_msfn_1d", seed)
                s2_config = load_experiment_config(s2_paths["config"])
                ext_config = load_experiment_config(ext_paths["config"])
                s2_metrics = _load_json(s2_paths["metrics"], f"S2 seed {seed} metrics")
                ext_metrics = _load_json(ext_paths["metrics"], f"SE-MSFN seed {seed} metrics")
                s2_replay = _replay(config=s2_config, checkpoint_path=s2_paths["checkpoint"], metrics=s2_metrics, dataset=s2_dataset, device=device)
                ext_replay = _replay(config=ext_config, checkpoint_path=ext_paths["checkpoint"], metrics=ext_metrics, dataset=ext_dataset, device=device)
                _aligned(s2_replay, ext_replay)
                s2_replays[seed] = s2_replay
                ext_replays[seed] = ext_replay
                for name, replay, paths, _config in (("lmscnet_s2", s2_replay, s2_paths, s2_config), ("se_msfn_1d", ext_replay, ext_paths, ext_config)):
                    pred_path = prediction_dir / f"{name}-seed-{seed}.npz"
                    prediction_manifest.append({
                        "model": name,
                        "seed": seed,
                        "filename": pred_path.name,
                        "prediction_sha256": _save_prediction(pred_path, replay),
                        "checkpoint_sha256": _sha256_file(paths["checkpoint"]),
                        "config_sha256": experiment_config_sha256(paths["config"]),
                        "sample_count": len(replay["targets"]),
                        **replay["metrics"],
                        "test_accessed": False,
                    })
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            dataset_binding = {
                "s2_training_split_manifest_sha256": s2_protocol["split_manifest_sha256"],
                "extended_training_split_manifest_sha256": ext_protocol["split_manifest_sha256"],
                "s2_split_manifest_sha256": _sha256_file(args.s2_split_manifest),
                "extended_split_manifest_sha256": _sha256_file(args.extended_split_manifest),
                "assignment_sha256": s2_dataset.assignment_sha256,
                "sample_count": len(s2_dataset),
                "preprocessing_mode": s2_dataset.preprocessing,
                "test_accessed": False,
            }
        comparison = _comparison(
            newer=s2_replays,
            older=ext_replays,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        report = {
            "schema_version": 1,
            "purpose": "s2_vs_strongest_extended_baseline_paired_hierarchical_bootstrap",
            "test_accessed": False,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "stratification": ["modulation", "snr_db"],
            "low_snr_values_db": list(LOW_SNR_VALUES),
            "reference_model": "lmscnet_s2",
            "variant_model": "se_msfn_1d",
            "dataset_binding": dataset_binding,
            "prediction_manifest": prediction_manifest,
            "comparison": comparison,
        }
        (temporary / "paired-bootstrap-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output_dir), "prediction_count": 10, "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
