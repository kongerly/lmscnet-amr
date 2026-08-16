"""Audit the downloaded RadioML 2018.01A source without test construction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import audit_radioml_2018_source  # noqa: E402


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2018_01a.yml",
    )
    args = parser.parse_args()
    result = audit_radioml_2018_source(
        archive_path=args.archive,
        hdf5_path=args.hdf5,
        classes_path=args.classes,
        license_path=args.license,
        dataset_spec_path=args.dataset_spec,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        project_commit=_commit(),
    )
    print(json.dumps({"samples": result["audit"]["samples"], "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
