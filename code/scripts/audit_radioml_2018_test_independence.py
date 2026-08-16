"""Audit whether RadioML 2018.01A test remains eligible for confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation.radioml_2018_independence import (  # noqa: E402
    audit_radioml_2018_test_independence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        required=True,
        help="External artifact root to scan; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-date", required=True, help="Frozen audit date in YYYY-MM-DD form.")
    parser.add_argument("--max-text-mib", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_radioml_2018_test_independence(
        project_root=PROJECT_ROOT,
        artifact_roots=args.artifact_root,
        output_dir=args.output_dir,
        audit_date=args.audit_date,
        max_text_bytes=args.max_text_mib * 1024 * 1024,
    )
    print(
        json.dumps(
            {
                "conclusion": result["conclusion"],
                "decisive_evidence_count": result["decisive_evidence_count"],
                "output_dir": result["output_dir"],
                "test_sample_content_opened_by_this_audit": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result["conclusion"] in {"eligible", "ineligible"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
