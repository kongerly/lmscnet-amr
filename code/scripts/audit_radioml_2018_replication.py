"""Audit the completed RadioML 2018.01A validation-only replication."""

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

MODELS = ("lmscnet_s0_k15", "lmscnet_s1", "lmscnet_s2", "se_msfn_1d")
SEEDS = (13, 37, 73)


class AuditError(ValueError):
    """Raised when a replication artifact differs from the frozen protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"Expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected JSON object: {path}")
    return value


def audit_queue(
    queue_root: Path, error_log: Path, expected_models: tuple[str, ...] = MODELS
) -> dict[str, Any]:
    queue = queue_root.resolve(strict=True)
    if not error_log.is_file() or error_log.stat().st_size != 0:
        raise AuditError("Queue error log is missing or non-empty")
    protocol = _json(queue / "queue-protocol.json")
    summary = _json(queue / "queue-summary.json")
    expected_run_count = len(expected_models) * len(SEEDS)
    if protocol.get("models") != list(expected_models) or protocol.get("run_count") != expected_run_count:
        raise AuditError("Queue model shard or run count differs")
    if summary.get("status") != "complete" or summary.get("test_accessed") is not False:
        raise AuditError("Queue summary is incomplete or test isolation differs")
    if any(summary.get(key) != value for key, value in protocol.items()):
        raise AuditError("Queue summary differs from the frozen protocol")
    multiseed_path = queue / "multiseed/multi-seed-summary.json"
    multiseed = _json(multiseed_path)
    if (
        summary.get("multi_seed_summary_sha256") != _sha256(multiseed_path)
        or multiseed.get("run_count") != expected_run_count
        or multiseed.get("test_accessed") is not False
    ):
        raise AuditError("Multi-seed summary identity or count differs")
    data_protocol = {
        "dataset_id": "radioml_2018_01a",
        "preprocessing_mode": "per_sample_max_abs",
        "input_shape": [2, 1024],
        "train_samples": 1_789_008,
        "validation_samples": 255_840,
    }
    expected_summary_bindings = {
        "split_manifest_sha256": protocol["split_manifest_sha256"],
        "assignment_sha256": protocol["assignment_sha256"],
        "project_commit": protocol["project_commit"],
        "seeds": list(SEEDS),
        "data_protocol": data_protocol,
    }
    if multiseed.get("bindings") != expected_summary_bindings:
        raise AuditError("Multi-seed summary bindings differ")
    rows = []
    identities = set()
    for run in multiseed.get("runs", []):
        if not isinstance(run, dict):
            raise AuditError("Multi-seed run summary is invalid")
        model = str(run.get("model"))
        seed = int(run.get("seed", -1))
        identity = (model, seed)
        if model not in expected_models or seed not in SEEDS or identity in identities:
            raise AuditError(f"Unexpected or duplicate run identity: {identity}")
        identities.add(identity)
        config_path = queue / "multiseed/configs" / str(run["config_filename"])
        run_dir = queue / "multiseed" / str(run["run_id"])
        metrics = _json(run_dir / "metrics.json")
        checkpoint_path = run_dir / "best.pt"
        if (run_dir / "last.pt").exists() or not checkpoint_path.is_file():
            raise AuditError(f"Completed run artifacts are invalid: {run_dir.name}")
        config = load_experiment_config(config_path)
        config_sha256 = experiment_config_sha256(config_path)
        expected_bindings = {
            "experiment_config_sha256": config_sha256,
            "split_manifest_sha256": protocol["split_manifest_sha256"],
            "assignment_sha256": protocol["assignment_sha256"],
            "project_commit": protocol["project_commit"],
            "seed": seed,
            "data_protocol": data_protocol,
        }
        checkpoint_sha256 = _sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            config.data["dataset_id"] != "radioml_2018_01a"
            or config.data["assignment_sha256"] != protocol["assignment_sha256"]
            or config.test_access != "forbidden"
            or int(config.model["num_classes"]) != 24
            or int(config.training["seed"]) != seed
            or metrics.get("bindings") != expected_bindings
            or metrics.get("test_accessed") is not False
            or not isinstance(checkpoint, dict)
            or checkpoint.get("bindings") != expected_bindings
            or checkpoint.get("model_name") != model
            or run.get("config_sha256") != config_sha256
            or run.get("checkpoint_sha256") != checkpoint_sha256
            or metrics.get("artifacts", {}).get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise AuditError(f"Run bindings differ: {run_dir.name}")
        rows.append(
            {
                "model": model,
                "seed": seed,
                "run_id": run["run_id"],
                "config_sha256": config_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "best_epoch": metrics["best_epoch"],
                "epochs_completed": metrics["epochs_completed"],
                "best_validation_macro_f1": metrics["best_validation_macro_f1"],
                "test_accessed": False,
            }
        )
    if identities != {(model, seed) for model in expected_models for seed in SEEDS}:
        raise AuditError("Replication matrix is incomplete")
    return {
        "schema_version": 1,
        "purpose": "radioml_2018_01a_validation_replication_audit",
        "test_accessed": False,
        "error_log_empty": True,
        "run_count": expected_run_count,
        "bindings": expected_summary_bindings,
        "queue_provenance": {
            "source_manifest_sha256": protocol["source_manifest_sha256"],
            "split_manifest_sha256": protocol["split_manifest_sha256"],
            "split_artifact_sha256": protocol["split_artifact_sha256"],
            "assignment_sha256": protocol["assignment_sha256"],
            "project_commit": protocol["project_commit"],
            "preprocessing_mode": protocol["preprocessing_mode"],
            "input_shape": protocol["input_shape"],
        },
        "queue_protocol_sha256": _sha256(queue / "queue-protocol.json"),
        "queue_summary_sha256": _sha256(queue / "queue-summary.json"),
        "runs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-log", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_queue(args.queue_root, args.error_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_count": report["run_count"], "test_accessed": False, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
