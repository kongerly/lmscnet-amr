"""Summarize the Phase R6 fixed-epoch five-seed validation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)
MODELS = (
    "lmscnet_s2",
    "lmscnet_s1_static",
    "lmscnet_s1_wide_static",
    "sknet_1d_adaptation",
    "afnet_adaptation",
)
SEEDS = (13, 37, 73, 101, 137)
CHECKPOINT_EPOCH = 100


class SummaryError(ValueError):
    """Raised when the R6 validation evidence is incomplete."""


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryError(f"{field} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_validation(metrics: dict[str, Any]) -> tuple[dict[str, Any], float]:
    epoch = metrics.get("selected_checkpoint_epoch")
    history = metrics.get("history")
    if metrics.get("selection_metric") != "fixed_epoch" or epoch != CHECKPOINT_EPOCH:
        raise SummaryError("Metrics does not use the frozen epoch-100 checkpoint")
    if not isinstance(history, list) or len(history) != CHECKPOINT_EPOCH:
        raise SummaryError("Metrics history is incomplete")
    record = history[epoch - 1]
    if not isinstance(record, dict) or not isinstance(record.get("validation"), dict):
        raise SummaryError("Selected validation record is invalid")
    return record["validation"], float(record["validation_loss"])


def _run_rows(queue_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = queue_root / "multi-seed-summary.json"
    summary = _load_json(summary_path, "multi-seed summary")
    if summary.get("test_accessed") is not False:
        raise SummaryError("Training summary reports test access")
    rows: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            raise SummaryError("Multi-seed run is invalid")
        model = str(run["model"])
        if model not in MODELS:
            raise SummaryError(f"Unexpected model in R6 queue: {model}")
        metrics = _load_json(queue_root / str(run["run_id"]) / "metrics.json", f"{run['run_id']} metrics")
        if metrics.get("test_accessed") is not False:
            raise SummaryError(f"{run['run_id']} reports test access")
        validation, validation_loss = _selected_validation(metrics)
        per_snr = validation.get("per_snr_accuracy")
        if not isinstance(per_snr, dict):
            raise SummaryError(f"{run['run_id']} lacks per-SNR accuracy")
        low_snr_accuracy = statistics.fmean(float(per_snr[f"{snr:+d}"]) for snr in LOW_SNR_VALUES)
        rows.append(
            {
                "model": model,
                "seed": int(run["seed"]),
                "run_id": str(run["run_id"]),
                "overall_accuracy": float(validation["accuracy"]),
                "macro_f1": float(validation["macro_f1"]),
                "low_snr_accuracy": low_snr_accuracy,
                "validation_loss": validation_loss,
                "checkpoint_epoch": CHECKPOINT_EPOCH,
                "checkpoint_sha256": str(run["checkpoint_sha256"]),
                "test_accessed": False,
            }
        )
    return rows, {
        "summary_sha256": _sha256_file(summary_path),
        "bindings": summary.get("bindings"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    if set(grouped) != set(MODELS):
        raise SummaryError("R6 model set is incomplete")
    output: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = grouped[model]
        if len(model_rows) != 5 or sorted(int(row["seed"]) for row in model_rows) != list(SEEDS):
            raise SummaryError(f"{model} does not contain the frozen five seeds")
        result: dict[str, Any] = {"model": model, "run_count": 5}
        for metric in ("overall_accuracy", "macro_f1", "low_snr_accuracy", "validation_loss"):
            values = [float(row[metric]) for row in model_rows]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_sample_std"] = statistics.stdev(values)
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_root = args.queue_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite summary directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise SummaryError("Summary output must remain outside the repository")
    rows, source = _run_rows(queue_root)
    if len(rows) != 25:
        raise SummaryError(f"Expected 25 R6 five-seed runs, got {len(rows)}")
    aggregate = _aggregate(rows)
    report = {
        "schema_version": 1,
        "purpose": "phase_r6_fixed_epoch_five_seed_validation_summary",
        "test_accessed": False,
        "selection_metric": "fixed_epoch",
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "validation_role": "post_training_assessment_only",
        "does_not_replace_phase_r2_gatekeeping": True,
        "low_snr_values_db": list(LOW_SNR_VALUES),
        "aggregation": "five-seed arithmetic mean and sample standard deviation",
        "source": source,
        "model_summary": aggregate,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "r6-fixed-epoch-five-seed-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "r6-fixed-epoch-five-seed-runs.csv", rows)
    _write_csv(output_dir / "r6-fixed-epoch-five-seed-models.csv", aggregate)
    print(json.dumps({"output_dir": str(output_dir), "run_count": len(rows), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
