"""Generate the external experiment-freeze manifest after all validation evidence is complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation import (  # noqa: E402
    build_freeze_manifest,
    sha256_file,
    write_manifest_atomic,
)


def _report(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("report must use NAME=PATH")
    return name, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--final-queue-root", type=Path, required=True)
    parser.add_argument("--extended-queue-root", type=Path, required=True)
    parser.add_argument("--report", type=_report, action="append", required=True)
    parser.add_argument("--consumption-marker", type=Path, required=True)
    parser.add_argument("--test-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_freeze_manifest(
        project_root=PROJECT_ROOT,
        hdf5_path=args.hdf5,
        conversion_manifest_path=args.conversion_manifest,
        split_manifest_path=args.split_manifest,
        leakage_audit_path=args.leakage_audit,
        final_queue_root=args.final_queue_root,
        extended_queue_root=args.extended_queue_root,
        reports=args.report,
        consumption_marker_path=args.consumption_marker,
        test_output_dir=args.test_output_dir,
    )
    output = write_manifest_atomic(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "selected_run_count": len(manifest["selection"]["runs"]),
                "report_count": len(manifest["reports"]),
                "test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
