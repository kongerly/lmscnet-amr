"""Audit every artifact in the frozen validation queue without dataset access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.engine import experiment_config_sha256  # noqa: E402

EXPECTED_RUN_COUNT = 57
EXPECTED_FINAL_COUNT = 30
EXPECTED_SWEEP_COUNT = 12
EXPECTED_BASELINE_COUNT = 15


class QueueAuditError(ValueError):
    """Raised when a validation artifact differs from the frozen queue protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QueueAuditError(f"{field} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QueueAuditError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise QueueAuditError(f"{field} must contain a JSON object")
    return value


def _best_record(metrics: dict[str, Any]) -> dict[str, Any]:
    epoch = metrics.get("best_epoch")
    history = metrics.get("history")
    if not isinstance(epoch, int) or not isinstance(history, list) or not 1 <= epoch <= len(history):
        raise QueueAuditError("metrics best_epoch/history is invalid")
    record = history[epoch - 1]
    if not isinstance(record, dict) or record.get("epoch") != epoch:
        raise QueueAuditError("best epoch record is inconsistent")
    return record


def _float_equal(left: object, right: object, field: str) -> None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise QueueAuditError(f"{field} must be numeric")
    if abs(float(left) - float(right)) > 1e-12:
        raise QueueAuditError(f"{field} differs")


def _audit_run(
    *,
    category: str,
    summary_run: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    if (run_dir / "last.pt").exists():
        raise QueueAuditError(f"Completed run retains last.pt: {run_dir.name}")
    metrics = _load_json(metrics_path, f"{run_dir.name} metrics")
    config = load_experiment_config(config_path)
    config_sha256 = experiment_config_sha256(config_path)
    seed = int(summary_run["seed"])
    expected_bindings = {
        "experiment_config_sha256": config_sha256,
        "split_manifest_sha256": protocol["split_manifest_sha256"],
        "assignment_sha256": protocol["assignment_sha256"],
        "project_commit": protocol["project_commit"],
        "seed": seed,
        "data_protocol": {"preprocessing_mode": protocol["preprocessing_mode"]},
    }
    if metrics.get("bindings") != expected_bindings:
        raise QueueAuditError(f"Bindings differ: {run_dir.name}")
    if metrics.get("test_accessed") is not False or config.test_access != "forbidden":
        raise QueueAuditError(f"Test isolation differs: {run_dir.name}")
    if int(config.training["seed"]) != seed:
        raise QueueAuditError(f"Config seed differs: {run_dir.name}")
    if config.data.get("assignment_sha256") != protocol["assignment_sha256"]:
        raise QueueAuditError(f"Config assignment differs: {run_dir.name}")
    if summary_run.get("config_sha256") != config_sha256:
        raise QueueAuditError(f"Summary config hash differs: {run_dir.name}")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("checkpoint_sha256") != checkpoint_sha256:
        raise QueueAuditError(f"Metrics checkpoint hash differs: {run_dir.name}")
    if summary_run.get("checkpoint_sha256") != checkpoint_sha256:
        raise QueueAuditError(f"Summary checkpoint hash differs: {run_dir.name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise QueueAuditError(f"Checkpoint is not a mapping: {run_dir.name}")
    if checkpoint.get("bindings") != expected_bindings:
        raise QueueAuditError(f"Checkpoint bindings differ: {run_dir.name}")
    if checkpoint.get("epoch") != metrics.get("best_epoch"):
        raise QueueAuditError(f"Checkpoint epoch differs: {run_dir.name}")
    if checkpoint.get("model_name") != config.model["name"]:
        raise QueueAuditError(f"Checkpoint model differs: {run_dir.name}")
    best = _best_record(metrics)
    validation = best.get("validation")
    checkpoint_validation = checkpoint.get("validation")
    if not isinstance(validation, dict) or not isinstance(checkpoint_validation, dict):
        raise QueueAuditError(f"Validation record is invalid: {run_dir.name}")
    _float_equal(
        validation.get("macro_f1"), metrics.get("best_validation_macro_f1"), "best macro F1"
    )
    _float_equal(
        checkpoint_validation.get("macro_f1"),
        metrics.get("best_validation_macro_f1"),
        "checkpoint macro F1",
    )
    return {
        "category": category,
        "run_id": summary_run["run_id"],
        "model": summary_run.get("model", config.model["name"]),
        "seed": seed,
        "project_commit": protocol["project_commit"],
        "split_manifest_sha256": protocol["split_manifest_sha256"],
        "assignment_sha256": protocol["assignment_sha256"],
        "preprocessing_mode": protocol["preprocessing_mode"],
        "config_filename": config_path.name,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": metrics["best_epoch"],
        "epochs_completed": metrics["epochs_completed"],
        "best_validation_accuracy": summary_run["best_validation_accuracy"],
        "best_validation_macro_f1": metrics["best_validation_macro_f1"],
        "best_validation_loss": summary_run["best_validation_loss"],
        "test_accessed": False,
    }


def _audit_multiseed_group(
    *,
    category: str,
    group_dir: Path,
    summary: dict[str, Any],
    expected_count: int,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    if summary.get("run_count") != expected_count or summary.get("test_accessed") is not False:
        raise QueueAuditError(f"{category} summary count or test isolation differs")
    expected_summary_bindings = {
        "split_manifest_sha256": protocol["split_manifest_sha256"],
        "assignment_sha256": protocol["assignment_sha256"],
        "project_commit": protocol["project_commit"],
        "seeds": protocol["seeds"],
        "data_protocol": {"preprocessing_mode": protocol["preprocessing_mode"]},
    }
    if summary.get("bindings") != expected_summary_bindings:
        raise QueueAuditError(f"{category} summary bindings differ")
    rows = []
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            raise QueueAuditError(f"{category} summary run is invalid")
        rows.append(
            _audit_run(
                category=category,
                summary_run=run,
                config_path=group_dir / "configs" / str(run["config_filename"]),
                run_dir=group_dir / str(run["run_id"]),
                protocol=protocol,
            )
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-log", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_root = args.queue_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output}")
    if output == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output.parents:
        raise QueueAuditError("Audit output must remain outside the repository")
    error_log = (
        args.error_log.resolve()
        if args.error_log
        else queue_root.parent / f"{queue_root.name}.err.log"
    )
    if not error_log.is_file() or error_log.stat().st_size != 0:
        raise QueueAuditError("Queue error log is missing or non-empty")
    protocol = _load_json(queue_root / "queue-protocol.json", "queue protocol")
    queue_summary = _load_json(queue_root / "queue-summary.json", "queue summary")
    if queue_summary.get("status") != "complete" or queue_summary.get("test_accessed") is not False:
        raise QueueAuditError("Queue summary is incomplete or test isolation differs")
    if any(queue_summary.get(key) != protocol.get(key) for key in protocol):
        raise QueueAuditError("Queue summary differs from queue protocol")
    final_dir = queue_root / "final-family-multiseed"
    baseline_dir = queue_root / "baseline-multiseed"
    final_summary_path = final_dir / "multi-seed-summary.json"
    baseline_summary_path = baseline_dir / "multi-seed-summary.json"
    if queue_summary.get("final_family_summary_sha256") != _sha256_file(final_summary_path):
        raise QueueAuditError("Final family summary hash differs")
    if queue_summary.get("baseline_summary_sha256") != _sha256_file(baseline_summary_path):
        raise QueueAuditError("Baseline summary hash differs")
    final_summary = _load_json(final_summary_path, "final family summary")
    baseline_summary = _load_json(baseline_summary_path, "baseline summary")
    rows = _audit_multiseed_group(
        category="final-family",
        group_dir=final_dir,
        summary=final_summary,
        expected_count=EXPECTED_FINAL_COUNT,
        protocol=protocol,
    )
    selected_configs = {
        Path(str(item["path"])).resolve(): str(item["sha256"])
        for item in queue_summary.get("selected_baseline_configs", [])
        if isinstance(item, dict)
    }
    sweep_count = 0
    for baseline in protocol["baselines"]:
        sweep_dir = queue_root / "baseline-sweeps" / str(baseline)
        summary = _load_json(sweep_dir / "sweep-summary.json", f"{baseline} sweep summary")
        if summary.get("test_accessed") is not False:
            raise QueueAuditError(f"{baseline} sweep test isolation differs")
        bindings = summary.get("bindings")
        if not isinstance(bindings, dict):
            raise QueueAuditError(f"{baseline} sweep bindings are invalid")
        for field in ("split_manifest_sha256", "assignment_sha256", "project_commit"):
            if bindings.get(field) != protocol[field]:
                raise QueueAuditError(f"{baseline} sweep {field} differs")
        if bindings.get("seed") != 13 or bindings.get("data_protocol") != {
            "preprocessing_mode": protocol["preprocessing_mode"]
        }:
            raise QueueAuditError(f"{baseline} sweep seed or preprocessing differs")
        runs = summary.get("runs")
        if not isinstance(runs, list) or len(runs) != 4:
            raise QueueAuditError(f"{baseline} sweep must contain four runs")
        for run in runs:
            if not isinstance(run, dict):
                raise QueueAuditError(f"{baseline} sweep run is invalid")
            run_with_model = {**run, "model": baseline, "seed": 13}
            rows.append(
                _audit_run(
                    category="baseline-sweep",
                    summary_run=run_with_model,
                    config_path=sweep_dir / "configs" / str(run["config_filename"]),
                    run_dir=sweep_dir / str(run["output_directory"]),
                    protocol=protocol,
                )
            )
            sweep_count += 1
        selected_run_id = summary.get("selected_run_id")
        selected_path = sweep_dir / "configs" / f"{selected_run_id}.yml"
        if selected_configs.get(selected_path.resolve()) != _sha256_file(selected_path):
            raise QueueAuditError(f"{baseline} selected config hash differs")
    if sweep_count != EXPECTED_SWEEP_COUNT:
        raise QueueAuditError("Sweep run count differs")
    rows.extend(
        _audit_multiseed_group(
            category="baseline-multiseed",
            group_dir=baseline_dir,
            summary=baseline_summary,
            expected_count=EXPECTED_BASELINE_COUNT,
            protocol=protocol,
        )
    )
    if len(rows) != EXPECTED_RUN_COUNT or len({row["run_id"] for row in rows}) != 45:
        # Sweep run IDs repeat across three baseline directories; category/model disambiguates them.
        identities = {(row["category"], row["model"], row["run_id"]) for row in rows}
        if len(rows) != EXPECTED_RUN_COUNT or len(identities) != EXPECTED_RUN_COUNT:
            raise QueueAuditError("Audited run identities are incomplete or duplicated")
    report = {
        "schema_version": 1,
        "purpose": "final_validation_queue_recovery_audit",
        "queue_root_name": queue_root.name,
        "test_accessed": False,
        "queue_protocol_sha256": _sha256_file(queue_root / "queue-protocol.json"),
        "queue_summary_sha256": _sha256_file(queue_root / "queue-summary.json"),
        "error_log_empty": True,
        "counts": {
            "final_family": EXPECTED_FINAL_COUNT,
            "baseline_sweep": EXPECTED_SWEEP_COUNT,
            "baseline_multiseed": EXPECTED_BASELINE_COUNT,
            "total": len(rows),
        },
        "bindings": {
            "project_commit": protocol["project_commit"],
            "split_manifest_sha256": protocol["split_manifest_sha256"],
            "assignment_sha256": protocol["assignment_sha256"],
            "preprocessing_mode": protocol["preprocessing_mode"],
            "seeds": protocol["seeds"],
        },
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "run_count": len(rows), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
