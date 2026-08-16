"""Audit the Phase R2 validation queue for binding and test-isolation integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


R2_MODELS = ("lmscnet_s1_static", "lmscnet_s1_wide_static", "sknet_1d_adaptation", "afnet_adaptation")
SEEDS = (13, 37, 73, 101, 137)
EXPECTED_COMMIT = "b0310ec63956c092a2327d68b45226494ea52a5a"
EXPECTED_SPLIT = "7c1d93c15bc24656f5857638bbccfd59932cc2f21b4c9f7ea36f47b3a5850dae"
EXPECTED_ASSIGNMENT = "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941"


class R2AuditError(ValueError):
    """Raised when the R2 queue audit fails."""


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
        raise R2AuditError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise R2AuditError(f"{field} must contain a JSON object")
    return value


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
        raise R2AuditError("Output must remain outside the repository")
    summary = _load_json(queue_root / "multi-seed-summary.json", "multi-seed summary")
    if summary.get("test_accessed") is not False:
        raise R2AuditError("Multi-seed summary reports test access")
    bindings = summary.get("bindings", {})
    expected_bindings = {
        "split_manifest_sha256": EXPECTED_SPLIT,
        "assignment_sha256": EXPECTED_ASSIGNMENT,
        "project_commit": EXPECTED_COMMIT,
        "seeds": list(SEEDS),
        "data_protocol": {"preprocessing_mode": "per_sample_max_abs"},
    }
    if bindings != expected_bindings:
        raise R2AuditError(f"Multi-seed bindings differ: {bindings}")
    runs = summary.get("runs", [])
    if len(runs) != 20:
        raise R2AuditError(f"Expected 20 runs, got {len(runs)}")
    checkpoints: list[dict[str, Any]] = []
    last_pt_found: list[Path] = []
    for run in runs:
        run_id = str(run["run_id"])
        model = str(run["model"])
        seed = int(run["seed"])
        if model not in R2_MODELS or seed not in SEEDS:
            raise R2AuditError(f"Unexpected run: {run_id}")
        if run.get("test_accessed") is not False:
            raise R2AuditError(f"Run {run_id} reports test access")
        run_dir = queue_root / run_id
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "best.pt"
        if not metrics_path.is_file() or not checkpoint_path.is_file():
            raise R2AuditError(f"Incomplete run: {run_id}")
        metrics = _load_json(metrics_path, f"{run_id} metrics")
        if metrics.get("test_accessed") is not False:
            raise R2AuditError(f"{run_id} metrics report test access")
        run_bindings = metrics.get("bindings", {})
        if run_bindings.get("project_commit") != EXPECTED_COMMIT:
            raise R2AuditError(f"{run_id} commit binding differs")
        if run_bindings.get("split_manifest_sha256") != EXPECTED_SPLIT:
            raise R2AuditError(f"{run_id} split binding differs")
        if run_bindings.get("assignment_sha256") != EXPECTED_ASSIGNMENT:
            raise R2AuditError(f"{run_id} assignment binding differs")
        if run_bindings.get("seed") != seed:
            raise R2AuditError(f"{run_id} seed binding differs")
        last_pt = run_dir / "last.pt"
        if last_pt.is_file():
            last_pt_found.append(last_pt)
        checkpoints.append(
            {
                "run_id": run_id,
                "model": model,
                "seed": seed,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "recorded_checkpoint_sha256": str(run["checkpoint_sha256"]),
                "config_sha256": _sha256_file(queue_root / "configs" / str(run["config_filename"])),
                "recorded_config_sha256": str(run["config_sha256"]),
            }
        )
    if last_pt_found:
        raise R2AuditError(f"Unexpected last.pt files: {last_pt_found}")
    for item in checkpoints:
        if item["checkpoint_sha256"] != item["recorded_checkpoint_sha256"]:
            raise R2AuditError(f"Checkpoint digest differs for {item['run_id']}")
        if item["config_sha256"] != item["recorded_config_sha256"]:
            raise R2AuditError(f"Config digest differs for {item['run_id']}")
    report = {
        "schema_version": 1,
        "purpose": "phase_r2_queue_recovery_audit",
        "test_accessed": False,
        "queue_root": str(queue_root),
        "expected_commit": EXPECTED_COMMIT,
        "expected_split_manifest_sha256": EXPECTED_SPLIT,
        "expected_assignment_sha256": EXPECTED_ASSIGNMENT,
        "run_count": len(checkpoints),
        "last_pt_count": 0,
        "runs": checkpoints,
        "passed": True,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "r2-queue-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "run_count": len(checkpoints), "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
