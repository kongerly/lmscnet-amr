"""Generate the formal five-seed w/o SNR auxiliary report."""

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
from na_lmscnet.evaluation.snr_auxiliary_ablation_report import (  # noqa: E402
    generate_snr_auxiliary_multiseed_report,
)

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the formal five-seed SNR auxiliary ablation."
    )
    parser.add_argument("--reference-output-root", type=Path, required=True)
    parser.add_argument("--reference-training-commit", required=True)
    parser.add_argument("--ablation-output-root", type=Path, required=True)
    parser.add_argument("--ablation-training-commit", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
        manifest = generate_snr_auxiliary_multiseed_report(
            reference_output_root=args.reference_output_root,
            reference_training_commit=args.reference_training_commit,
            ablation_output_root=args.ablation_output_root,
            ablation_training_commit=args.ablation_training_commit,
            report_dir=args.report_dir,
            hdf5_path=args.hdf5,
            split_manifest_path=args.split_manifest,
            leakage_audit_path=args.leakage_audit,
            validation_dataset=validation_dataset,
            project_root=PROJECT_ROOT,
            report_generation_commit=_project_commit(),
            device=torch.device(args.device),
        )
    print(json.dumps({"decision": manifest["decision"], "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
