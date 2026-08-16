"""Run or resume the frozen RadioML 2018.01A validation-only replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2018HDF5Dataset  # noqa: E402
from na_lmscnet.training import load_experiment_config, run_multi_seed  # noqa: E402
from na_lmscnet.training.progress import ProgressReporter  # noqa: E402

CONFIG_NAMES = (
    "lmscnet_s0_k15_radioml_2018_01a_selected.yml",
    "lmscnet_s1_radioml_2018_01a_selected.yml",
    "lmscnet_s2_radioml_2018_01a_selected.yml",
    "se_msfn_1d_radioml_2018_01a_selected.yml",
)
MODELS = ("lmscnet_s0_k15", "lmscnet_s1", "lmscnet_s2", "se_msfn_1d")
CONFIG_BY_MODEL = dict(zip(MODELS, CONFIG_NAMES, strict=True))
SEEDS = (13, 37, 73)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _project_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Replication requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_or_validate_json(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        if path.is_symlink() or _json(path) != value:
            raise ValueError(f"Existing protocol differs: {path}")
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_models(models: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if models is None:
        return MODELS
    requested = tuple(models)
    if not requested:
        raise ValueError("models must be non-empty")
    if len(requested) != len(set(requested)):
        raise ValueError("models must be unique")
    unknown = set(requested).difference(MODELS)
    if unknown:
        raise ValueError(f"models must be a frozen subset; unknown: {sorted(unknown)}")
    return tuple(model for model in MODELS if model in requested)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-artifact", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        help="Frozen model subset for a task-level GPU shard; defaults to all four models.",
    )
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2018_01a.yml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_models = _validate_models(args.models)
    commit = _project_commit()
    output = args.output_root.resolve()
    root = PROJECT_ROOT.resolve()
    if output == root or root in output.parents:
        raise ValueError("Replication artifacts must remain outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    config_paths = [
        PROJECT_ROOT / "code/configs/experiments" / CONFIG_BY_MODEL[model]
        for model in selected_models
    ]
    configs = [load_experiment_config(path) for path in config_paths]
    if tuple(str(config.model["name"]) for config in configs) != selected_models:
        raise ValueError("Replication configs differ from the frozen model order")
    split_manifest = _json(args.split_manifest)
    assignment = str(split_manifest.get("assignment", {}).get("sha256"))
    protocol = {
        "schema_version": 1,
        "purpose": "radioml_2018_01a_validation_replication",
        "project_commit": commit,
        "dataset_id": "radioml_2018_01a",
        "source_manifest_sha256": _sha256(args.source_manifest),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "split_artifact_sha256": _sha256(args.split_artifact),
        "assignment_sha256": assignment,
        "preprocessing_mode": "per_sample_max_abs",
        "input_shape": [2, 1024],
        "models": list(selected_models),
        "seeds": list(SEEDS),
        "run_count": len(selected_models) * len(SEEDS),
        "config_sha256": {path.name: _sha256(path) for path in config_paths},
        "test_accessed": False,
    }
    _write_or_validate_json(output / "queue-protocol.json", protocol)
    multiseed_output = output / "multiseed"
    multiseed_output.mkdir(exist_ok=True)
    common = {
        "hdf5_path": args.hdf5,
        "source_manifest_path": args.source_manifest,
        "split_artifact_path": args.split_artifact,
        "split_manifest_path": args.split_manifest,
        "dataset_spec_path": args.dataset_spec,
    }
    reporter = ProgressReporter()

    def progress(event: dict[str, object]) -> None:
        if event.get("event") == "epoch_complete":
            reporter.on_epoch(event, run_id=str(event.get("run_id")))

    with (
        RadioML2018HDF5Dataset(split="train", **common) as train_dataset,
        RadioML2018HDF5Dataset(split="validation", **common) as validation_dataset,
    ):
        if (
            train_dataset.assignment_sha256 != assignment
            or validation_dataset.assignment_sha256 != assignment
        ):
            raise ValueError("Loaded assignments differ from the queue protocol")
        data_protocol = {
            "dataset_id": "radioml_2018_01a",
            "preprocessing_mode": "per_sample_max_abs",
            "input_shape": [2, 1024],
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
        }
        summary = run_multi_seed(
            base_config_paths=config_paths,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            output_dir=multiseed_output,
            project_root=PROJECT_ROOT,
            project_commit=commit,
            split_manifest_sha256=protocol["split_manifest_sha256"],
            assignment_sha256=assignment,
            device=torch.device(args.device),
            progress_callback=progress,
            batch_callback=reporter.on_batch,
            data_protocol=data_protocol,
            seeds=SEEDS,
        )
    reporter.finish()
    queue_summary = {
        **protocol,
        "status": "complete",
        "multi_seed_summary_sha256": _sha256(output / "multiseed/multi-seed-summary.json"),
        "completed_run_count": summary["run_count"],
    }
    _write_or_validate_json(output / "queue-summary.json", queue_summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "models": list(selected_models),
                "run_count": len(selected_models) * len(SEEDS),
                "test_accessed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
