"""Validate the repository RadioML near-duplicate audit design contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.near_duplicate_contract import (  # noqa: E402
    load_near_duplicate_contract,
    near_duplicate_contract_sha256,
)

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_near_duplicate.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the RadioML near-duplicate audit design contract."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_near_duplicate_contract(
        args.contract, args.dataset_spec, args.conversion_contract
    )
    summary = {
        "contract_id": contract["contract_id"],
        "contract_sha256": near_duplicate_contract_sha256(args.contract),
        "dataset_id": contract["dataset_id"],
        "candidate_generation": contract["candidate_generation"]["status"],
        "threshold_calibration": contract["threshold_calibration"]["status"],
        "review": contract["review"]["status"],
        "audit_generation_enabled": contract["generation_gate"]["audit_generation_enabled"],
        "split_generation_enabled": contract["generation_gate"]["split_generation_enabled"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
