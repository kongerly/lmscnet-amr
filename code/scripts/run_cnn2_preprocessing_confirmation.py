"""Confirm the two close CNN2 preprocessing candidates on seeds 37 and 73."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import GlobalZScoreStatistics, RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.training import load_experiment_config, run_training  # noqa: E402
from na_lmscnet.training.progress import ProgressReporter  # noqa: E402

MODES = ("global_zscore", "per_sample_max_abs")
SEEDS = (37, 73)
DEFAULT_CONFIG = PROJECT_ROOT / "code/configs/experiments/cnn2_radioml_2016_10a_selected.yml"
DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _project_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _load_statistics(path: Path) -> tuple[GlobalZScoreStatistics, dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document["statistics"]
    statistics = GlobalZScoreStatistics(
        channel_mean=tuple(raw["channel_mean"]),
        channel_std=tuple(raw["channel_std"]),
        scalar_count_per_channel=int(raw["scalar_count_per_channel"]),
        split=str(raw["split"]),
        estimator=str(raw["estimator"]),
    )
    if statistics.split != "train" or statistics.estimator != "population":
        raise ValueError("Confirmation requires train-only population statistics")
    if document.get("statistics_sha256") != statistics.sha256():
        raise ValueError("Global z-score statistics hash is inconsistent")
    if document.get("test_accessed") is not False:
        raise ValueError("Statistics artifact does not attest test isolation")
    return statistics, document


def main() -> int:
    args = parse_args()
    project_commit = _project_commit()
    diagnosis_dir = args.diagnosis_dir.resolve(strict=True)
    if diagnosis_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in diagnosis_dir.parents:
        raise ValueError("Confirmation artifacts must remain outside the repository")
    output = diagnosis_dir / "preprocessing-confirmation"
    output.mkdir(exist_ok=True)
    configs_dir = output / "configs"
    configs_dir.mkdir(exist_ok=True)

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    base_config = load_experiment_config(args.base_config)
    if base_config.model["name"] != "cnn2" or int(base_config.training["seed"]) != 13:
        raise ValueError("Confirmation base config must be the selected seed-13 CNN2 config")

    diagnostic_protocol_path = diagnosis_dir / "diagnostic-protocol.json"
    diagnostic_protocol = json.loads(diagnostic_protocol_path.read_text(encoding="utf-8"))
    if diagnostic_protocol.get("test_accessed") is not False:
        raise ValueError("Seed-13 diagnostic protocol does not attest test isolation")
    statistics_path = diagnosis_dir / "global-zscore-statistics.json"
    statistics, statistics_document = _load_statistics(statistics_path)

    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    with RadioML2016HDF5Dataset(split="train", preprocessing="raw", **common) as train:
        assignment_sha256 = train.assignment_sha256
    if statistics_document["bindings"]["assignment_sha256"] != assignment_sha256:
        raise ValueError("Statistics artifact is bound to another split assignment")

    protocol = {
        "schema_version": 1,
        "purpose": "cnn2_preprocessing_three_seed_confirmation",
        "test_accessed": False,
        "project_commit": project_commit,
        "selection_reason": (
            "seed 13 global_zscore and per_sample_max_abs were close and crossed metrics"
        ),
        "candidate_modes": list(MODES),
        "additional_seeds": list(SEEDS),
        "seed_13_results_reused_from": str(diagnosis_dir.name),
        "controlled_fields": diagnostic_protocol["controlled_fields"],
        "global_zscore_statistics_sha256": statistics.sha256(),
        "bindings": {
            "diagnostic_protocol_file_sha256": _sha256_file(diagnostic_protocol_path),
            "diagnostic_protocol_sha256": diagnostic_protocol["protocol_sha256"],
            "statistics_file_sha256": _sha256_file(statistics_path),
            "assignment_sha256": assignment_sha256,
            "split_manifest_sha256": _sha256_file(args.split_manifest),
            "conversion_manifest_sha256": _sha256_file(args.conversion_manifest),
            "leakage_audit_sha256": _sha256_file(args.leakage_audit),
            "hdf5_sha256": _sha256_file(args.hdf5),
        },
    }
    protocol_sha256 = _json_sha256(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    protocol_path = output / "confirmation-protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing confirmation protocol differs from this invocation")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    reporter = ProgressReporter()
    completed = []
    for seed in SEEDS:
        for mode in MODES:
            run_id = f"{mode}-seed-{seed}"
            derived = json.loads(json.dumps(base))
            derived["experiment_id"] = f"cnn2_preprocessing_{mode}_seed{seed}_confirmation_v1"
            derived["training"]["seed"] = seed
            config_path = configs_dir / f"{run_id}.yml"
            serialized = yaml.safe_dump(derived, sort_keys=False)
            if config_path.exists() and config_path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError(f"Existing derived config differs: {config_path}")
            config_path.write_text(serialized, encoding="utf-8")
            config = load_experiment_config(config_path)
            run_dir = output / run_id
            run_dir.mkdir(exist_ok=True)
            if (run_dir / "metrics.json").is_file():
                completed.append(run_id)
                continue
            preprocessing = {
                "preprocessing": mode,
                "global_zscore": statistics if mode == "global_zscore" else None,
            }
            with (
                RadioML2016HDF5Dataset(split="train", **preprocessing, **common) as train,
                RadioML2016HDF5Dataset(
                    split="validation", **preprocessing, **common
                ) as validation,
            ):
                run_training(
                    config=config,
                    config_path=config_path,
                    train_dataset=train,
                    validation_dataset=validation,
                    output_dir=run_dir,
                    project_root=PROJECT_ROOT,
                    project_commit=project_commit,
                    split_manifest_sha256=_sha256_file(args.split_manifest),
                    device=torch.device(args.device),
                    epoch_callback=lambda event, name=run_id: reporter.on_epoch(
                        event, run_id=name
                    ),
                    batch_callback=reporter.on_batch,
                    resume=(run_dir / "last.pt").is_file(),
                    data_protocol={
                        "confirmation_protocol_sha256": protocol_sha256,
                        "preprocessing_mode": mode,
                        "global_zscore_statistics_sha256": (
                            statistics.sha256() if mode == "global_zscore" else None
                        ),
                    },
                )
            completed.append(run_id)
    reporter.finish()
    print(json.dumps({"completed_runs": completed, "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
