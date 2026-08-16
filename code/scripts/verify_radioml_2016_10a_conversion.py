"""Independently verify RadioML HDF5, manifest, contract, and source archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.hdf5_conversion import verify_conversion  # noqa: E402

DEFAULT_SPEC = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a.yml"
DEFAULT_CONTRACT = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a_conversion.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a converted RadioML HDF5 artifact.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_conversion(
        args.hdf5,
        args.manifest,
        args.archive,
        args.spec,
        args.contract,
        PROJECT_ROOT,
    )
    summary = {
        "dataset_id": result["manifest"]["dataset_id"],
        "cell_count": result["inspection"]["cell_count"],
        "sample_count": result["inspection"]["sample_count"],
        "output_file_sha256": result["manifest"]["digests"]["output_file_sha256"],
        "output_logical_content_sha256": result["inspection"]["logical_content_sha256"],
        "verified": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
