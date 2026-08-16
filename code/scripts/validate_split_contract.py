"""Validate the repository RadioML split design contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.split_contract import (  # noqa: E402
    allocation_counts,
    load_split_contract,
    split_contract_sha256,
)

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the RadioML split design contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_split_contract(args.contract, args.dataset_spec, args.conversion_contract)
    per_stratum = allocation_counts(contract, contract["stratification"]["samples_per_stratum"])
    summary = {
        "contract_id": contract["contract_id"],
        "contract_sha256": split_contract_sha256(args.contract),
        "dataset_id": contract["dataset_id"],
        "generation_enabled": contract["generation_gate"]["split_generation_enabled"],
        "per_stratum": per_stratum,
        "seed": contract["assignment"]["seed"],
        "totals": contract["stratification"]["expected_totals"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
