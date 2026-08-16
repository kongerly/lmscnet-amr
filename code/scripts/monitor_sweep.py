"""Print a compact live progress table for external sweep or training output dirs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _epoch_counts(log_path: Path) -> dict[str, int]:
    """Count completed epochs per run_id from JSON events or human progress lines."""

    counts: dict[str, int] = {}
    if not log_path.is_file():
        return counts
    try:
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("{"):
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") != "epoch_complete":
                        continue
                    run_id = event.get("run_id")
                    epoch = event.get("epoch")
                    if isinstance(run_id, str) and isinstance(epoch, int):
                        counts[run_id] = epoch
                    continue
                fields = {}
                for token in stripped.split("|"):
                    token = token.strip()
                    if token.startswith("epoch "):
                        fields["epoch"] = token[6:].split("/")[0]
                    elif "=" in token:
                        key, value = token.split("=", 1)
                        fields[key.strip()] = value.strip()
                run_id = fields.get("run_id")
                epoch_text = fields.get("epoch", "")
                if run_id and epoch_text.isdigit():
                    counts[run_id] = int(epoch_text)
    except OSError:
        return counts
    return counts


def _run_line(run_dir: Path, epoch_counts: dict[str, int]) -> str:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return (
                f"{run_dir.name:<32} done    epochs={metrics['epochs_completed']:>3} "
                f"best_epoch={metrics['best_epoch']:>3} "
                f"macro_f1={metrics['best_validation_macro_f1']:.4f}"
            )
        except (OSError, ValueError, KeyError, TypeError):
            return f"{run_dir.name:<32} done    (metrics unreadable)"
    epochs = epoch_counts.get(run_dir.name, 0)
    if (run_dir / "last.pt").is_file():
        return f"{run_dir.name:<32} running epoch={epochs:>3} (best pending)"
    return f"{run_dir.name:<32} queued"


def render(output_dirs: list[Path], log_path: Path | None) -> str:
    """Render a compact table of every sweep or training run found in the output dirs."""

    lines = [f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    counts = _epoch_counts(log_path) if log_path is not None else {}
    for output_dir in output_dirs:
        summary_path = output_dir / "sweep-summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                lines.append(
                    f"[COMPLETE] {output_dir.name}: selected={summary['selected_run_id']} "
                    f"test_accessed={summary['test_accessed']}"
                )
                continue
            except (OSError, ValueError, KeyError, TypeError):
                lines.append(f"[?] {output_dir.name}: summary unreadable")
                continue
        run_dirs = sorted(
            (item for item in output_dir.iterdir() if item.is_dir() and item.name != "configs"),
            key=lambda item: item.name,
        )
        if not run_dirs:
            lines.append(f"[WAITING] {output_dir.name}: no run started yet")
            continue
        lines.append(f"[RUNNING] {output_dir.name}: {len(run_dirs)} run(s)")
        for run_dir in run_dirs:
            lines.append(f"    {_run_line(run_dir, counts)}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live progress table for external sweep or training output dirs."
    )
    parser.add_argument("--dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None, help="Runner JSON-lines log path")
    parser.add_argument("--interval", type=float, default=0.0, help="Refresh seconds (0 = once)")
    parser.add_argument("--output", type=Path, default=None, help="Progress file to overwrite")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for output_dir in args.dirs:
        if not output_dir.is_dir():
            print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
            return 2
    while True:
        text = render(args.dirs, args.log)
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        if args.interval <= 0:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
