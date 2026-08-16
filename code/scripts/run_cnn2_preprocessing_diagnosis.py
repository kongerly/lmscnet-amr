"""Run the controlled seed-13 CNN2 preprocessing diagnosis without test access."""

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

from na_lmscnet.data import (  # noqa: E402
    PREPROCESSING_MODES,
    RadioML2016HDF5Dataset,
    compute_global_zscore_statistics,
)
from na_lmscnet.training import load_experiment_config, run_training  # noqa: E402
from na_lmscnet.training.progress import ProgressReporter  # noqa: E402

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
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_commit = _project_commit()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config = load_experiment_config(args.base_config)
    if config.model["name"] != "cnn2" or int(config.training["seed"]) != 13:
        raise ValueError("The diagnosis base config must be CNN2 with seed 13")
    output = args.output_dir.resolve()
    if PROJECT_ROOT.resolve() == output or PROJECT_ROOT.resolve() in output.parents:
        raise ValueError("Diagnosis artifacts must remain outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    configs_dir = output / "configs"
    configs_dir.mkdir(exist_ok=True)

    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    with RadioML2016HDF5Dataset(split="train", preprocessing="raw", **common) as raw_train:
        statistics = compute_global_zscore_statistics(raw_train, batch_size=4096)
        assignment_sha256 = raw_train.assignment_sha256

    statistics_document = {
        "schema_version": 1,
        "purpose": "cnn2_train_only_global_zscore_statistics",
        "test_accessed": False,
        "statistics": statistics.to_dict(),
        "statistics_sha256": statistics.sha256(),
        "bindings": {
            "project_commit": project_commit,
            "assignment_sha256": assignment_sha256,
            "split_manifest_sha256": _sha256_file(args.split_manifest),
            "hdf5_sha256": _sha256_file(args.hdf5),
        },
    }
    stats_path = output / "global-zscore-statistics.json"
    stats_path.write_text(json.dumps(statistics_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "purpose": "cnn2_preprocessing_diagnosis_seed13",
        "test_accessed": False,
        "project_commit": project_commit,
        "base_config_sha256": _sha256_file(args.base_config),
        "controlled_fields": [
            "model",
            "train_augmentation",
            "optimizer",
            "scheduler",
            "training_budget",
            "early_stopping",
            "checkpoint_selection",
        ],
        "varied_field": "preprocessing_mode",
        "modes": list(PREPROCESSING_MODES),
        "global_zscore_statistics_sha256": statistics.sha256(),
        "global_zscore_definition": "per-channel population mean/std over every scalar in frozen train split only",
        "low_snr_definition_db": [-10, -8, -6, -4, -2, 0],
        "bindings": {
            "assignment_sha256": assignment_sha256,
            "split_manifest_sha256": _sha256_file(args.split_manifest),
            "conversion_manifest_sha256": _sha256_file(args.conversion_manifest),
            "leakage_audit_sha256": _sha256_file(args.leakage_audit),
            "hdf5_sha256": _sha256_file(args.hdf5),
        },
    }
    protocol_sha256 = _json_sha256(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    (output / "diagnostic-protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    reporter = ProgressReporter()
    completed = []
    for mode in PREPROCESSING_MODES:
        derived = json.loads(json.dumps(base))
        derived["experiment_id"] = f"cnn2_preprocessing_{mode}_seed13_v1"
        config_path = configs_dir / f"{mode}.yml"
        config_path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
        run_config = load_experiment_config(config_path)
        run_dir = output / mode
        run_dir.mkdir(exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            completed.append(mode)
            continue
        preprocessing_kwargs = {
            "preprocessing": mode,
            "global_zscore": statistics if mode == "global_zscore" else None,
        }
        with (
            RadioML2016HDF5Dataset(split="train", **preprocessing_kwargs, **common) as train_dataset,
            RadioML2016HDF5Dataset(
                split="validation", **preprocessing_kwargs, **common
            ) as validation_dataset,
        ):
            data_protocol = {
                "diagnostic_protocol_sha256": protocol_sha256,
                "preprocessing_mode": mode,
                "global_zscore_statistics_sha256": (
                    statistics.sha256() if mode == "global_zscore" else None
                ),
            }
            run_training(
                config=run_config,
                config_path=config_path,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                output_dir=run_dir,
                project_root=PROJECT_ROOT,
                project_commit=project_commit,
                split_manifest_sha256=_sha256_file(args.split_manifest),
                device=torch.device(args.device),
                epoch_callback=lambda event, run_id=mode: reporter.on_epoch(event, run_id=run_id),
                batch_callback=reporter.on_batch,
                resume=(run_dir / "last.pt").is_file(),
                data_protocol=data_protocol,
            )
        completed.append(mode)
    reporter.finish()
    print(json.dumps({"completed_modes": completed, "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
