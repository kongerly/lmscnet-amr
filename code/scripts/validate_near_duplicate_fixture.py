"""Validate deterministic near-duplicate calibration fixture evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.near_duplicate_fixture import (  # noqa: E402
    build_near_duplicate_fixture_evidence,
)

DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_NEAR_DUPLICATE_CONTRACT = (
    PROJECT_ROOT / "code/configs/data/radioml_2016_10a_near_duplicate.yml"
)
DEFAULT_FIXTURE_CONTRACT = (
    PROJECT_ROOT / "code/configs/data/radioml_2016_10a_near_duplicate_fixture.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate deterministic near-duplicate calibration fixture evidence."
    )
    parser.add_argument("--fixture-contract", type=Path, default=DEFAULT_FIXTURE_CONTRACT)
    parser.add_argument(
        "--near-duplicate-contract", type=Path, default=DEFAULT_NEAR_DUPLICATE_CONTRACT
    )
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = build_near_duplicate_fixture_evidence(
        args.fixture_contract,
        args.near_duplicate_contract,
        args.dataset_spec,
        args.conversion_contract,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
