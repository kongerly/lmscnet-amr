"""Audit that no training, evaluation, bootstrap, or monitoring job is running."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATTERNS = (
    "train_baseline.py",
    "train_cnn2.py",
    "run_multi_seed.py",
    "run_validation_sweep.py",
    "run_final_validation_family.py",
    "run_extended_baseline_validation.py",
    "run_r2_prediction_replay.py",
    "run_revision_intervention_replay.py",
    "run_r2_primary_contrasts.py",
    "run_final_paired_bootstrap.py",
    "run_extended_paired_bootstrap.py",
    "monitor_sweep.py",
    "analyze_r2_gate_mechanism.py",
    "audit_r2_intervention_validity.py",
    "audit_r6_fixed_epoch_queue.py",
    "summarize_r6_fixed_epoch_validation.py",
    "run_r6_fixed_epoch_contrasts.py",
    "generate_r6_validation_freeze.py",
)


def _processes() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(result.stdout)
    return value if isinstance(value, list) else [value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite process audit: {output}")
    if PROJECT_ROOT.resolve() == output or PROJECT_ROOT.resolve() in output.parents:
        raise ValueError("Process audit artifact must remain outside the repository")

    current_pid = os.getpid()
    matches = []
    for process in _processes():
        pid = int(process.get("ProcessId") or -1)
        command_line = str(process.get("CommandLine") or "")
        if pid == current_pid:
            continue
        matched = [pattern for pattern in PATTERNS if pattern.lower() in command_line.lower()]
        if matched:
            matches.append(
                {
                    "process_id": pid,
                    "name": process.get("Name"),
                    "matched_patterns": matched,
                    "command_line": command_line,
                }
            )
    report = {
        "schema_version": 1,
        "purpose": "revision_freeze_process_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": not matches,
        "test_accessed": False,
        "running_research_jobs": matches,
        "patterns_checked": list(PATTERNS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": not matches, "test_accessed": False}))
    return 0 if not matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
