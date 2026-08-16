"""Merge-audit four completed RadioML 2018.01A single-model GPU shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from audit_radioml_2018_replication import (  # noqa: E402
    MODELS,
    SEEDS,
    AuditError,
    audit_queue,
)


def audit_shards(shard_root: Path, logs_root: Path) -> dict[str, Any]:
    shards = []
    identities: set[tuple[str, int]] = set()
    common: dict[str, object] | None = None
    for model in MODELS:
        report = audit_queue(
            shard_root / f"shard-{model}",
            logs_root / f"shard-{model}.err.log",
            (model,),
        )
        current_common = report["queue_provenance"]
        if common is None:
            common = current_common
        elif current_common != common:
            raise AuditError("Shard commit, split, assignment, or preprocessing differs")
        for row in report["runs"]:
            identity = (str(row["model"]), int(row["seed"]))
            if identity in identities:
                raise AuditError(f"Duplicate shard run identity: {identity}")
            identities.add(identity)
        shards.append(
            {
                "model": model,
                "queue_protocol_sha256": report["queue_protocol_sha256"],
                "queue_summary_sha256": report["queue_summary_sha256"],
                "run_count": report["run_count"],
                "runs": report["runs"],
            }
        )
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if identities != expected:
        raise AuditError("Merged replication matrix is incomplete or duplicated")
    return {
        "schema_version": 1,
        "purpose": "radioml_2018_01a_validation_replication_shard_audit",
        "test_accessed": False,
        "error_logs_empty": True,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "run_count": len(expected),
        "bindings": common,
        "shards": shards,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit_shards(args.shard_root.resolve(strict=True), args.logs_root.resolve(strict=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_count": report["run_count"], "test_accessed": False, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
