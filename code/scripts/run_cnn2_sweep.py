"""Backward-compatible entry point for the CNN2 validation sweep."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "code" / "scripts"
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
for root in (SCRIPTS_ROOT, SOURCE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from run_validation_sweep import main  # noqa: E402

DEFAULT_SWEEP = PROJECT_ROOT / "code/configs/experiments/cnn2_radioml_2016_10a_sweep.yml"


def _entry() -> int:
    arguments = sys.argv[1:]
    if "--sweep" not in arguments:
        arguments.extend(["--sweep", str(DEFAULT_SWEEP)])
    return main(arguments)


if __name__ == "__main__":
    raise SystemExit(_entry())
