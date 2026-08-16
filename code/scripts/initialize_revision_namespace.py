"""Initialize the external Major Revision Phase R0 artifact namespace."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation.revision_namespace import initialize_revision_namespace  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialization-date", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "code/configs/revision/phase_r0.yml",
    )
    parser.add_argument("--independence-report", type=Path, required=True)
    parser.add_argument("--test-consumed-marker", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = initialize_revision_namespace(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        config_path=args.config,
        independence_report_path=args.independence_report,
        test_consumed_marker_path=args.test_consumed_marker,
        initialization_date=args.initialization_date,
        project_commit=_git("rev-parse", "HEAD").strip(),
        worktree_status=_git("status", "--porcelain=v1", "--untracked-files=all"),
    )
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "status": result["status"],
                "formal_runs_authorized": result["guards"]["formal_runs_authorized"],
                "confirmatory_test_construction_allowed": result["guards"][
                    "confirmatory_test_construction_allowed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
