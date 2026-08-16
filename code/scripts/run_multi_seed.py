"""Run or resume multi-seed baseline training without test access."""

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
from na_lmscnet.training import run_multi_seed  # noqa: E402
from na_lmscnet.training.progress import ProgressReporter  # noqa: E402

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume frozen-config multi-seed baseline training."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        required=True,
        help="Frozen selected configs, one per baseline",
    )
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73, 101, 137])
    return parser.parse_args(argv)


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
    if event.get("event") == "epoch_complete":
        reporter.on_epoch(event, run_id=event.get("run_id"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        if train_dataset.assignment_sha256 != validation_dataset.assignment_sha256:
            raise ValueError("Train and validation assignments differ")
        summary = run_multi_seed(
            base_config_paths=args.configs,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            output_dir=args.output_dir,
            project_root=PROJECT_ROOT,
            project_commit=_project_commit(),
            split_manifest_sha256=_sha256_file(args.split_manifest),
            assignment_sha256=train_dataset.assignment_sha256,
            device=torch.device(args.device),
            progress_callback=_progress,
            batch_callback=reporter.on_batch,
            data_protocol={"preprocessing_mode": train_dataset.preprocessing},
            seeds=tuple(args.seeds),
        )
    reporter.finish()
    print(
        json.dumps(
            {
                "event": "multi_seed_complete",
                "run_count": summary["run_count"],
                "test_accessed": summary["test_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
