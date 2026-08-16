"""Convert the verified RadioML archive to the contracted HDF5 layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.hdf5_conversion import convert_archive  # noqa: E402

DEFAULT_SPEC = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a.yml"
DEFAULT_CONTRACT = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a_conversion.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert statically validated RadioML buffers to atomic HDF5 artifacts."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = convert_archive(
        args.archive, args.spec, args.contract, args.output_dir, PROJECT_ROOT
    )
    summary = {
        "dataset_id": manifest["dataset_id"],
        "hdf5_filename": manifest["artifacts"]["hdf5_filename"],
        "manifest_filename": manifest["artifacts"]["manifest_filename"],
        "output_file_sha256": manifest["digests"]["output_file_sha256"],
        "output_logical_content_sha256": manifest["digests"]["output_logical_content_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
