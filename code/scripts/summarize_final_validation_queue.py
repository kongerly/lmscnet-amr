"""Summarize frozen five-seed validation results and freeze model selections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.engine import experiment_config_sha256  # noqa: E402

LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)
S0_ORDER = ("lmscnet_s0_k3", "lmscnet_s0_k7", "lmscnet_s0_k15")
BASELINE_MODELS = ("cnn2", "cldnn", "resnet1d")


class SummaryError(ValueError):
    """Raised when the audited validation evidence is incomplete."""


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


def _best_validation(metrics: dict[str, Any]) -> tuple[dict[str, Any], float]:
    best_epoch = metrics.get("best_epoch")
    history = metrics.get("history")
    if not isinstance(best_epoch, int) or not isinstance(history, list):
        raise SummaryError("Metrics lacks best epoch history")
    record = history[best_epoch - 1]
    if not isinstance(record, dict) or not isinstance(record.get("validation"), dict):
        raise SummaryError("Best validation record is invalid")
    return record["validation"], float(record["validation_loss"])


def _run_rows(group_dir: Path) -> list[dict[str, Any]]:
    summary = _load_json(group_dir / "multi-seed-summary.json", "multi-seed summary")
    rows = []
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            raise SummaryError("Multi-seed run is invalid")
        metrics_path = group_dir / str(run["run_id"]) / "metrics.json"
        metrics = _load_json(metrics_path, f"{run['run_id']} metrics")
        validation, validation_loss = _best_validation(metrics)
        per_snr = validation.get("per_snr_accuracy")
        if not isinstance(per_snr, dict):
            raise SummaryError(f"{run['run_id']} lacks per-SNR accuracy")
        low_snr_accuracy = statistics.fmean(float(per_snr[f"{snr:+d}"]) for snr in LOW_SNR_VALUES)
        rows.append(
            {
                "model": str(run["model"]),
                "seed": int(run["seed"]),
                "run_id": str(run["run_id"]),
                "overall_accuracy": float(validation["accuracy"]),
                "macro_f1": float(validation["macro_f1"]),
                "low_snr_accuracy": low_snr_accuracy,
                "validation_loss": validation_loss,
                "best_epoch": int(metrics["best_epoch"]),
                "checkpoint_sha256": str(run["checkpoint_sha256"]),
                "test_accessed": False,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    output = []
    for model, model_rows in grouped.items():
        if len(model_rows) != 5 or sorted(row["seed"] for row in model_rows) != [13, 37, 73, 101, 137]:
            raise SummaryError(f"{model} does not contain the frozen five seeds")
        result: dict[str, Any] = {"model": model, "run_count": 5}
        for metric in ("overall_accuracy", "macro_f1", "low_snr_accuracy", "validation_loss"):
            values = [float(row[metric]) for row in model_rows]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_sample_std"] = statistics.stdev(values)
        output.append(result)
    return sorted(output, key=lambda item: item["model"])


def _select_s0(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = {row["model"]: row for row in aggregate if row["model"] in S0_ORDER}
    if set(candidates) != set(S0_ORDER):
        raise SummaryError("S0 candidates are incomplete")
    return min(
        candidates.values(),
        key=lambda row: (
            -float(row["macro_f1_mean"]),
            float(row["validation_loss_mean"]),
            S0_ORDER.index(str(row["model"])),
        ),
    )


def _baseline_selection(queue_root: Path) -> list[dict[str, Any]]:
    selections = []
    for model in BASELINE_MODELS:
        sweep_dir = queue_root / "baseline-sweeps" / model
        summary = _load_json(sweep_dir / "sweep-summary.json", f"{model} sweep summary")
        selected_id = str(summary["selected_run_id"])
        selected_run = next(
            run for run in summary["runs"] if isinstance(run, dict) and run["run_id"] == selected_id
        )
        external_config = sweep_dir / "configs" / str(selected_run["config_filename"])
        repository_config = (
            PROJECT_ROOT / "code/configs/experiments" / f"{model}_radioml_2016_10a_selected.yml"
        )
        external = load_experiment_config(external_config)
        repository = load_experiment_config(repository_config)
        frozen_values = {
            "learning_rate": float(external.optimizer["learning_rate"]),
            "dropout": float(external.model["dropout"]),
        }
        if frozen_values != {
            "learning_rate": float(repository.optimizer["learning_rate"]),
            "dropout": float(repository.model["dropout"]),
        }:
            raise SummaryError(f"Repository selected config differs for {model}")
        selections.append(
            {
                "model": model,
                "selection_rule": summary["selection"],
                "selected_run_id": selected_id,
                **frozen_values,
                "external_config_sha256": experiment_config_sha256(external_config),
                "repository_config_filename": repository_config.name,
                "repository_config_sha256": experiment_config_sha256(repository_config),
                "selected_checkpoint_sha256": selected_run["checkpoint_sha256"],
                "test_accessed": False,
            }
        )
    return selections


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_root = args.queue_root.resolve(strict=True)
    audit_report = args.audit_report.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite summary directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise SummaryError("Summary output must remain outside the repository")
    audit = _load_json(audit_report, "queue audit")
    if audit.get("test_accessed") is not False or audit.get("counts", {}).get("total") != 57:
        raise SummaryError("Queue audit is incomplete")
    rows = _run_rows(queue_root / "final-family-multiseed")
    rows.extend(_run_rows(queue_root / "baseline-multiseed"))
    if len(rows) != 45:
        raise SummaryError("Expected 45 formal five-seed runs")
    aggregate = _aggregate(rows)
    strongest_s0 = _select_s0(aggregate)
    baseline_selection = _baseline_selection(queue_root)
    strongest_current_baseline = max(
        (row for row in aggregate if row["model"] in BASELINE_MODELS),
        key=lambda row: float(row["macro_f1_mean"]),
    )
    report = {
        "schema_version": 1,
        "purpose": "final_validation_five_seed_summary_and_selection_freeze",
        "test_accessed": False,
        "queue_audit_sha256": _sha256_file(audit_report),
        "low_snr_values_db": list(LOW_SNR_VALUES),
        "aggregation": "five-seed arithmetic mean and sample standard deviation",
        "s0_selection_rule": [
            "mean validation macro-F1 descending",
            "mean validation loss ascending",
            "kernel order 3,7,15",
        ],
        "strongest_s0": strongest_s0["model"],
        "strongest_current_baseline": strongest_current_baseline["model"],
        "model_summary": aggregate,
        "baseline_selection": baseline_selection,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "validation-five-seed-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "validation-five-seed-runs.csv", rows)
    _write_csv(output_dir / "validation-five-seed-models.csv", aggregate)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "strongest_s0": strongest_s0["model"],
                "strongest_current_baseline": strongest_current_baseline["model"],
                "test_accessed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
