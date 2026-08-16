"""Generate the external validation-only baseline report."""

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
from na_lmscnet.evaluation import generate_baseline_report  # noqa: E402

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a validation-only baseline report.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--training-project-commit", required=True)
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
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    split_manifest_sha256 = _sha256_file(args.split_manifest)
    report_generation_commit = _project_commit()
    with RadioML2016HDF5Dataset(split="validation", **common) as validation_dataset:
        manifest = generate_baseline_report(
            output_root=args.output_root,
            report_dir=args.report_dir,
            hdf5_path=args.hdf5,
            conversion_manifest_path=args.conversion_manifest,
            split_manifest_path=args.split_manifest,
            leakage_audit_path=args.leakage_audit,
            split_contract_path=args.split_contract,
            dataset_spec_path=args.dataset_spec,
            conversion_contract_path=args.conversion_contract,
            project_root=PROJECT_ROOT,
            training_project_commit=args.training_project_commit,
            report_generation_project_commit=report_generation_commit,
            validation_dataset=validation_dataset,
            device=torch.device(args.device),
            warmup=args.warmup,
            iterations=args.iterations,
        )
    print(json.dumps({"report_dir": str(args.report_dir.resolve()), "test_accessed": manifest["test_accessed"], "run_count": manifest["bindings"]["run_count"], "split_manifest_sha256": split_manifest_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
