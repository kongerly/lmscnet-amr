"""Validate the repository RadioML conversion contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.conversion_contract import (  # noqa: E402
    conversion_contract_sha256,
    load_conversion_contract,
)

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a.yml"
DEFAULT_CONTRACT = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a_conversion.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the RadioML HDF5 conversion contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_conversion_contract(args.contract, args.dataset_spec)
    summary = {
        "schema_version": contract["schema_version"],
        "contract_id": contract["contract_id"],
        "contract_sha256": conversion_contract_sha256(args.contract),
        "dataset_id": contract["dataset_id"],
        "format": contract["format"]["name"],
        "output_filename": contract["format"]["output_filename"],
        "manifest_filename": contract["manifest"]["filename"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
