"""Audit whether frozen NA-LMSCNet checkpoints can be reused for formal ablations."""

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

from na_lmscnet.evaluation import source_equivalence_audit  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reusable NA-LMSCNet validation evidence.")
    parser.add_argument("--reference-training-commit", required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _clean_project_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Source-equivalence audit requires a clean Git worktree")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite source-equivalence audit: {output}")
    project_commit = _clean_project_commit()
    audit = source_equivalence_audit(
        project_root=PROJECT_ROOT,
        reference_training_commit=args.reference_training_commit,
        formal_training_commit=project_commit,
        reference_checkpoint_path=args.reference_checkpoint,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(output), "reuse_authorized": audit["reuse_authorized"]}))
    return 0 if audit["reuse_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
