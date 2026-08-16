"""Audit the Phase R6 fixed-epoch validation queue and its frozen bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS = (
    "lmscnet_s2",
    "lmscnet_s1_static",
    "lmscnet_s1_wide_static",
    "sknet_1d_adaptation",
    "afnet_adaptation",
)
SEEDS = (13, 37, 73, 101, 137)
EXPECTED_COMMIT = "b6c56ced7b6893a135554b4c8a5fb3c089f58744"
EXPECTED_SPLIT = "2be8545369dc76a0400d876e502708e6be95d1392c6d73ea1c2b6c5b1af50c71"
EXPECTED_ASSIGNMENT = "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941"
CHECKPOINT_EPOCH = 100


class R6AuditError(ValueError):
    """Raised when the R6 fixed-epoch queue audit fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R6AuditError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise R6AuditError(f"{field} must contain a JSON object")
    return value


def _audit_metrics(metrics: dict[str, Any], *, run_id: str, model: str, seed: int) -> None:
    if metrics.get("test_accessed") is not False:
        raise R6AuditError(f"{run_id} metrics report test access")
    if metrics.get("purpose") != "publication_candidate":
        raise R6AuditError(f"{run_id} purpose differs")
    if metrics.get("selection_metric") != "fixed_epoch":
        raise R6AuditError(f"{run_id} selection metric differs")
    if metrics.get("selected_checkpoint_epoch") != CHECKPOINT_EPOCH:
        raise R6AuditError(f"{run_id} selected checkpoint epoch differs")
    if metrics.get("best_epoch") != CHECKPOINT_EPOCH:
        raise R6AuditError(f"{run_id} recorded checkpoint epoch differs")
    history = metrics.get("history")
    if not isinstance(history, list) or len(history) != CHECKPOINT_EPOCH:
        raise R6AuditError(f"{run_id} history is incomplete")
    if history[-1].get("epoch") != CHECKPOINT_EPOCH:
        raise R6AuditError(f"{run_id} final history epoch differs")
    model_record = metrics.get("model")
    if not isinstance(model_record, dict) or model_record.get("name") != model:
        raise R6AuditError(f"{run_id} model binding differs")
    bindings = metrics.get("bindings")
    expected = {
        "project_commit": EXPECTED_COMMIT,
        "split_manifest_sha256": EXPECTED_SPLIT,
        "assignment_sha256": EXPECTED_ASSIGNMENT,
        "seed": seed,
    }
    if not isinstance(bindings, dict) or any(bindings.get(key) != value for key, value in expected.items()):
        raise R6AuditError(f"{run_id} artifact bindings differ")


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
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise R6AuditError("Output must remain outside the repository")

    summary_path = queue_root / "multi-seed-summary.json"
    summary = _load_json(summary_path, "multi-seed summary")
    if summary.get("test_accessed") is not False or summary.get("run_count") != 25:
        raise R6AuditError("Multi-seed summary count or test isolation differs")
    expected_bindings = {
        "split_manifest_sha256": EXPECTED_SPLIT,
        "assignment_sha256": EXPECTED_ASSIGNMENT,
        "project_commit": EXPECTED_COMMIT,
        "seeds": list(SEEDS),
        "data_protocol": {"preprocessing_mode": "per_sample_max_abs"},
    }
    if summary.get("bindings") != expected_bindings:
        raise R6AuditError("Multi-seed summary bindings differ")

    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != 25:
        raise R6AuditError("R6 queue must contain exactly 25 runs")
    expected_pairs = {(model, seed) for model in MODELS for seed in SEEDS}
    observed_pairs: set[tuple[str, int]] = set()
    audited: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise R6AuditError("Multi-seed run entry is invalid")
        run_id = str(run["run_id"])
        model = str(run["model"])
        seed = int(run["seed"])
        pair = (model, seed)
        if pair not in expected_pairs or pair in observed_pairs:
            raise R6AuditError(f"Unexpected or duplicate run: {run_id}")
        observed_pairs.add(pair)
        if run.get("test_accessed") is not False:
            raise R6AuditError(f"{run_id} reports test access")
        if run.get("best_epoch") != CHECKPOINT_EPOCH or run.get("epochs_completed") != CHECKPOINT_EPOCH:
            raise R6AuditError(f"{run_id} did not complete the fixed epoch")

        run_dir = queue_root / run_id
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "best.pt"
        config_path = queue_root / "configs" / str(run["config_filename"])
        if not all(path.is_file() for path in (metrics_path, checkpoint_path, config_path)):
            raise R6AuditError(f"Incomplete run artifacts: {run_id}")
        if (run_dir / "last.pt").exists():
            raise R6AuditError(f"Unexpected resumable checkpoint: {run_id}")
        metrics = _load_json(metrics_path, f"{run_id} metrics")
        _audit_metrics(metrics, run_id=run_id, model=model, seed=seed)
        checkpoint_sha = _sha256_file(checkpoint_path)
        config_sha = _sha256_file(config_path)
        if checkpoint_sha != run.get("checkpoint_sha256"):
            raise R6AuditError(f"Checkpoint digest differs: {run_id}")
        if config_sha != run.get("config_sha256"):
            raise R6AuditError(f"Config digest differs: {run_id}")
        audited.append(
            {
                "run_id": run_id,
                "model": model,
                "seed": seed,
                "checkpoint_epoch": CHECKPOINT_EPOCH,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "test_accessed": False,
            }
        )
    if observed_pairs != expected_pairs:
        raise R6AuditError("R6 model-seed matrix is incomplete")

    report = {
        "schema_version": 1,
        "purpose": "phase_r6_fixed_epoch_queue_audit",
        "passed": True,
        "test_accessed": False,
        "queue_root": str(queue_root),
        "queue_summary_sha256": _sha256_file(summary_path),
        "selection_metric": "fixed_epoch",
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "run_count": len(audited),
        "last_pt_count": 0,
        "bindings": expected_bindings,
        "runs": sorted(audited, key=lambda item: (item["model"], item["seed"])),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "r6-fixed-epoch-queue-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "run_count": len(audited), "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
