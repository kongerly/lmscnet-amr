"""Generate the module-7 first-priority seed-13 ablation report."""

from __future__ import annotations

import argparse
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
from na_lmscnet.evaluation import (  # noqa: E402
    generate_core_ablation_report,
    validate_split_audit_pair,
)

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the module-7 seed-13 core ablations.")
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--reference-training-commit", required=True)
    parser.add_argument("--wo-multi-scale-config", type=Path, required=True)
    parser.add_argument("--wo-multi-scale-run-dir", type=Path, required=True)
    parser.add_argument("--wo-multi-scale-training-commit", required=True)
    parser.add_argument("--fixed-average-config", type=Path, required=True)
    parser.add_argument("--fixed-average-run-dir", type=Path, required=True)
    parser.add_argument("--fixed-average-training-commit", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args(argv)


def _project_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _assert_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("Report generation requires a clean Git worktree")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _assert_clean_worktree()
    validate_split_audit_pair(args.split_manifest, args.leakage_audit)
    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    with RadioML2016HDF5Dataset(split="validation", **common) as validation_dataset:
        manifest = generate_core_ablation_report(
            reference_config_path=args.reference_config,
            reference_run_dir=args.reference_run_dir,
            reference_training_commit=args.reference_training_commit,
            wo_multi_scale_config_path=args.wo_multi_scale_config,
            wo_multi_scale_run_dir=args.wo_multi_scale_run_dir,
            wo_multi_scale_training_commit=args.wo_multi_scale_training_commit,
            fixed_average_config_path=args.fixed_average_config,
            fixed_average_run_dir=args.fixed_average_run_dir,
            fixed_average_training_commit=args.fixed_average_training_commit,
            report_dir=args.report_dir,
            hdf5_path=args.hdf5,
            split_manifest_path=args.split_manifest,
            leakage_audit_path=args.leakage_audit,
            validation_dataset=validation_dataset,
            project_root=PROJECT_ROOT,
            report_generation_commit=_project_commit(),
            device=torch.device(args.device),
            warmup=args.warmup,
            iterations=args.iterations,
        )
    print(
        json.dumps(
            {"combined_decision": manifest["combined_decision"], "test_accessed": False},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
