"""Summarize audited extended-baseline validation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("resnet1d_macs", "mobilenetv2_1d", "mcldnn", "se_msfn_1d")
SEEDS = (13, 37, 73, 101, 137)
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    queue_root = args.queue_root.resolve(strict=True)
    audit_path = args.audit_report.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise ValueError("Summary output must remain outside repository")
    audit = _json(audit_path)
    if audit.get("test_accessed") is not False or audit.get("counts", {}).get("total") != 36:
        raise ValueError("Audit is incomplete")
    rows: list[dict[str, Any]] = []
    group = _json(queue_root / "multiseed" / "multi-seed-summary.json")
    for run in group["runs"]:
        model = str(run["model"])
        metrics = _json(queue_root / "multiseed" / str(run["run_id"]) / "metrics.json")
        best_epoch = int(metrics["best_epoch"])
        record = metrics["history"][best_epoch - 1]
        validation = record["validation"]
        low = statistics.fmean(float(validation["per_snr_accuracy"][f"{snr:+d}"]) for snr in LOW_SNR_VALUES)
        rows.append({
            "model": model,
            "seed": int(run["seed"]),
            "run_id": str(run["run_id"]),
            "overall_accuracy": float(validation["accuracy"]),
            "macro_f1": float(validation["macro_f1"]),
            "low_snr_accuracy": low,
            "validation_loss": float(record["validation_loss"]),
            "best_epoch": best_epoch,
            "checkpoint_sha256": str(run["checkpoint_sha256"]),
            "test_accessed": False,
        })
    if len(rows) != 20:
        raise ValueError("Expected 20 multiseed rows")
    aggregate: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    for model in BASELINES:
        model_rows = grouped[model]
        if sorted(row["seed"] for row in model_rows) != list(SEEDS):
            raise ValueError(f"Seed set differs for {model}")
        result: dict[str, Any] = {"model": model, "run_count": 5}
        for metric in ("overall_accuracy", "macro_f1", "low_snr_accuracy", "validation_loss"):
            values = [float(row[metric]) for row in model_rows]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_sample_std"] = statistics.stdev(values)
        aggregate.append(result)
    strongest = max(aggregate, key=lambda row: float(row["macro_f1_mean"]))
    report = {
        "schema_version": 1,
        "purpose": "extended_baseline_validation_five_seed_summary",
        "test_accessed": False,
        "queue_audit_sha256": _sha256(audit_path),
        "low_snr_values_db": list(LOW_SNR_VALUES),
        "aggregation": "five-seed arithmetic mean and sample standard deviation",
        "strongest_extended_baseline": strongest["model"],
        "model_summary": sorted(aggregate, key=lambda row: row["model"]),
        "rows": rows,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "extended-five-seed-summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "strongest_extended_baseline": strongest["model"], "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
