"""Audit numeric quality in the validated RadioML pickle payload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.numeric_quality import audit_numeric_quality_archive  # noqa: E402
from na_lmscnet.data.provenance import write_inventory  # noqa: E402

DEFAULT_SPEC = PROJECT_ROOT / "code" / "configs" / "data" / "radioml_2016_10a.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RadioML float32 buffers without pickle deserialization."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.output.name.endswith(".numeric-audit.json"):
        raise ValueError("Output filename must end with '.numeric-audit.json'")
    report = audit_numeric_quality_archive(args.archive, args.spec)
    write_inventory(report, args.output)
    print(f"Wrote numeric quality audit to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
