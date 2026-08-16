"""Scan the legacy RadioML pickle payload without deserializing it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.pickle_safety import inspect_pickle_archive  # noqa: E402
from na_lmscnet.data.provenance import write_inventory  # noqa: E402

DEFAULT_SPEC = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect RadioML Python 2 pickle opcodes without deserialization."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.output.name.endswith(".pickle-scan.json"):
        raise ValueError("Output filename must end with '.pickle-scan.json'")
    report = inspect_pickle_archive(args.archive, args.spec)
    write_inventory(report, args.output)
    print(f"Wrote no-execution pickle scan to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
