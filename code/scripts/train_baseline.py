"""Train one supported baseline on the frozen RadioML train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.training import load_experiment_config, run_training  # noqa: E402
from na_lmscnet.training.progress import ProgressReporter  # noqa: E402

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline without accessing the test split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _project_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Training requires a clean Git worktree for an exact code binding")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


reporter = ProgressReporter()


def _progress(event: dict[str, object]) -> None:
    reporter.on_epoch(event)


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    with (
        RadioML2016HDF5Dataset(split="train", **common) as train_dataset,
        RadioML2016HDF5Dataset(split="validation", **common) as validation_dataset,
    ):
        if (
            config.data["assignment_sha256"] != train_dataset.assignment_sha256
            or train_dataset.assignment_sha256 != validation_dataset.assignment_sha256
        ):
            raise ValueError("Experiment config assignment SHA-256 differs from the split manifest")
        result = run_training(
            config=config,
            config_path=args.config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            output_dir=args.output_dir,
            project_root=PROJECT_ROOT,
            project_commit=_project_commit(),
            split_manifest_sha256=_sha256_file(args.split_manifest),
            device=torch.device(args.device),
            epoch_callback=_progress,
            batch_callback=reporter.on_batch,
            resume=args.resume,
            data_protocol={"preprocessing_mode": train_dataset.preprocessing},
        )
    reporter.finish()
    summary = {
        "experiment_id": result["experiment_id"],
        "purpose": result["purpose"],
        "epochs_completed": result["epochs_completed"],
        "best_epoch": result["best_epoch"],
        "best_validation_macro_f1": result["best_validation_macro_f1"],
        "test_accessed": result["test_accessed"],
        "checkpoint_sha256": result["artifacts"]["checkpoint_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
