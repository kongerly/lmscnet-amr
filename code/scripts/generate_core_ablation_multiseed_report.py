"""Generate the formal five-seed module-7 core-ablation report."""

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
    generate_core_ablation_multiseed_report,
    validate_split_audit_pair,
)

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report formal five-seed module-7 core ablations.")
    parser.add_argument("--reference-output-root", type=Path, required=True)
    parser.add_argument("--reference-training-commit", required=True)
    parser.add_argument("--wo-multi-scale-output-root", type=Path, required=True)
    parser.add_argument("--wo-multi-scale-training-commit", required=True)
    parser.add_argument("--fixed-average-output-root", type=Path, required=True)
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
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
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
        raise RuntimeError("Formal report generation requires a clean Git worktree")


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
        manifest = generate_core_ablation_multiseed_report(
            reference_output_root=args.reference_output_root,
            reference_training_commit=args.reference_training_commit,
            wo_multi_scale_output_root=args.wo_multi_scale_output_root,
            wo_multi_scale_training_commit=args.wo_multi_scale_training_commit,
            fixed_average_output_root=args.fixed_average_output_root,
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
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    print(
        json.dumps(
            {"mainline_decision": manifest["mainline_decision"], "test_accessed": False},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
