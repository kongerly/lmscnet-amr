"""Generate the frozen compact RadioML 2018.01A split artifact."""

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

from na_lmscnet.data import generate_radioml_2018_split  # noqa: E402


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
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-exact-duplicate-audit", action="store_true")
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2018_01a.yml",
    )
    parser.add_argument(
        "--split-contract",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2018_01a_split.yml",
    )
    args = parser.parse_args()
    result = generate_radioml_2018_split(
        hdf5_path=args.hdf5,
        source_manifest_path=args.source_manifest,
        dataset_spec_path=args.dataset_spec,
        split_contract_path=args.split_contract,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        project_commit=_commit(),
        audit_exact_duplicates=not args.skip_exact_duplicate_audit,
    )
    print(
        json.dumps(
            {
                "assignment_sha256": result["assignment"]["sha256"],
                "counts": result["counts"],
                "test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
