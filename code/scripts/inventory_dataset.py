"""Create a safe metadata inventory for a locally acquired dataset archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.provenance import build_archive_inventory, write_inventory  # noqa: E402

DEFAULT_SPEC = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash and inspect a dataset archive without extracting it."
    )
    parser.add_argument("archive", type=Path, help="Path to RML2016.10a.tar.bz2")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.output.name.endswith(".dataset-inventory.json"):
        raise ValueError("Output filename must end with '.dataset-inventory.json'")
    inventory = build_archive_inventory(args.archive, args.spec)
    write_inventory(inventory, args.output)
    print(
        f"Recorded {inventory['archive']['filename']} with "
        f"SHA-256 {inventory['archive']['sha256']} in {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
