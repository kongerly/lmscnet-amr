"""Audit the completed extended-baseline validation queue without test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("resnet1d_macs", "mobilenetv2_1d", "mcldnn", "se_msfn_1d")
SEEDS = (13, 37, 73, 101, 137)


class AuditError(ValueError):
    """Raised when a queue artifact violates the frozen protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"Expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected JSON object: {path}")
    return value


def _audit_run(
    run_dir: Path,
    *,
    summary_run: dict[str, Any],
    protocol: dict[str, Any],
    expected_model: str,
    expected_seed: int,
    config_path: Path,
    category: str,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise AuditError(f"Missing completed artifacts: {run_dir}")
    if (run_dir / "last.pt").exists():
        raise AuditError(f"Completed run retains last.pt: {run_dir}")
    metrics = _json(metrics_path)
    bindings = metrics.get("bindings")
    expected_bindings = {
        "assignment_sha256": protocol["assignment_sha256"],
        "data_protocol": {"preprocessing_mode": protocol["preprocessing_mode"]},
        "experiment_config_sha256": _sha256(config_path),
        "project_commit": protocol["project_commit"],
        "seed": expected_seed,
        "split_manifest_sha256": protocol["split_manifest_sha256"],
    }
    if bindings != expected_bindings:
        raise AuditError(f"Bindings differ: {run_dir}")
    if metrics.get("test_accessed") is not False:
        raise AuditError(f"Test isolation differs: {run_dir}")
    model = metrics.get("model")
    if not isinstance(model, dict) or model.get("name") != expected_model:
        raise AuditError(f"Model differs: {run_dir}")
    checkpoint_sha = _sha256(checkpoint_path)
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("checkpoint_sha256") != checkpoint_sha:
        raise AuditError(f"Checkpoint hash differs: {run_dir}")
    if summary_run.get("config_sha256") != expected_bindings["experiment_config_sha256"]:
        raise AuditError(f"Summary config hash differs: {run_dir}")
    if summary_run.get("checkpoint_sha256") != checkpoint_sha:
        raise AuditError(f"Summary checkpoint hash differs: {run_dir}")
    if summary_run.get("test_accessed") is not False:
        raise AuditError(f"Summary test isolation differs: {run_dir}")
    checkpoint = _json_checkpoint(checkpoint_path)
    if checkpoint.get("bindings") != expected_bindings:
        raise AuditError(f"Checkpoint bindings differ: {run_dir}")
    if checkpoint.get("model_name") != expected_model:
        raise AuditError(f"Checkpoint model differs: {run_dir}")
    return {
        "category": category,
        "model": expected_model,
        "run_id": run_dir.name,
        "seed": expected_seed,
        "config_filename": config_path.name,
        "config_sha256": expected_bindings["experiment_config_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "best_epoch": metrics.get("best_epoch"),
        "epochs_completed": metrics.get("epochs_completed"),
        "best_validation_accuracy": metrics.get("best_validation_accuracy"),
        "best_validation_macro_f1": metrics.get("best_validation_macro_f1"),
        "test_accessed": False,
    }


def _json_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise AuditError(f"Checkpoint is not a mapping: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-log", type=Path, required=True)
    args = parser.parse_args(argv)
    queue_root = args.queue_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output.parents:
        raise AuditError("Audit output must remain outside repository")
    if not args.error_log.is_file() or args.error_log.stat().st_size != 0:
        raise AuditError("Queue error log is missing or non-empty")
    protocol = _json(queue_root / "queue-protocol.json")
    summary = _json(queue_root / "queue-summary.json")
    if summary.get("status") != "complete" or summary.get("test_accessed") is not False:
        raise AuditError("Queue is incomplete or test isolation differs")
    for key in protocol:
        if summary.get(key) != protocol.get(key):
            raise AuditError(f"Queue summary differs from protocol: {key}")
    rows: list[dict[str, Any]] = []
    for model in BASELINES:
        sweep_dir = queue_root / "sweeps" / model
        sweep = _json(sweep_dir / "sweep-summary.json")
        if sweep.get("test_accessed") is not False or len(sweep.get("runs", [])) != 4:
            raise AuditError(f"Invalid sweep summary: {model}")
        bindings = sweep.get("bindings")
        if not isinstance(bindings, dict) or any(
            bindings.get(field) != protocol[field]
            for field in ("project_commit", "split_manifest_sha256", "assignment_sha256")
        ):
            raise AuditError(f"Sweep bindings differ: {model}")
        if bindings.get("seed") != 13 or bindings.get("data_protocol") != {
            "preprocessing_mode": protocol["preprocessing_mode"]
        }:
            raise AuditError(f"Sweep seed or preprocessing differs: {model}")
        for run in sweep["runs"]:
            config_path = sweep_dir / "configs" / str(run["config_filename"])
            run_dir = sweep_dir / str(run["output_directory"])
            rows.append(_audit_run(run_dir, summary_run=run, protocol=protocol, expected_model=model,
                                   expected_seed=13, config_path=config_path,
                                   category="sweep"))
        selected_id = str(sweep.get("selected_run_id"))
        selected_path = sweep_dir / "configs" / f"{selected_id}.yml"
        queue_selected = {
            Path(str(item["path"])).resolve(): str(item["sha256"])
            for item in summary.get("selected_configs", [])
            if isinstance(item, dict)
        }
        if queue_selected.get(selected_path.resolve()) != _sha256(selected_path):
            raise AuditError(f"Selected config binding differs: {model}")
    multiseed = _json(queue_root / "multiseed" / "multi-seed-summary.json")
    if multiseed.get("run_count") != 20 or multiseed.get("test_accessed") is not False:
        raise AuditError("Invalid multiseed summary")
    if summary.get("multiseed_summary_sha256") != _sha256(
        queue_root / "multiseed" / "multi-seed-summary.json"
    ):
        raise AuditError("Multiseed summary hash differs")
    expected_multiseed_bindings = {
        "assignment_sha256": protocol["assignment_sha256"],
        "data_protocol": {"preprocessing_mode": protocol["preprocessing_mode"]},
        "project_commit": protocol["project_commit"],
        "seeds": protocol["seeds"],
        "split_manifest_sha256": protocol["split_manifest_sha256"],
    }
    if multiseed.get("bindings") != expected_multiseed_bindings:
        raise AuditError("Multiseed bindings differ")
    for run in multiseed.get("runs", []):
        model = str(run["model"])
        if model not in BASELINES or int(run["seed"]) not in SEEDS:
            raise AuditError(f"Invalid multiseed identity: {run}")
        config_path = queue_root / "multiseed" / "configs" / str(run["config_filename"])
        run_dir = queue_root / "multiseed" / str(run["run_id"])
        rows.append(_audit_run(run_dir, summary_run=run, protocol=protocol, expected_model=model,
                               expected_seed=int(run["seed"]), config_path=config_path,
                               category="multiseed"))
    if len(rows) != 36 or len({(r["category"], r["model"], r["run_id"]) for r in rows}) != 36:
        raise AuditError(f"Expected 36 unique audited runs, got {len(rows)}")
    report = {
        "schema_version": 1,
        "purpose": "extended_baseline_validation_queue_audit",
        "queue_root_name": queue_root.name,
        "test_accessed": False,
        "error_log_empty": True,
        "counts": {"sweep": 16, "multiseed": 20, "total": 36},
        "bindings": {
            "project_commit": protocol["project_commit"],
            "split_manifest_sha256": protocol["split_manifest_sha256"],
            "assignment_sha256": protocol["assignment_sha256"],
            "preprocessing_mode": protocol["preprocessing_mode"],
            "seeds": protocol["seeds"],
        },
        "queue_protocol_sha256": _sha256(queue_root / "queue-protocol.json"),
        "queue_summary_sha256": _sha256(queue_root / "queue-summary.json"),
        "runs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "run_count": 36, "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
