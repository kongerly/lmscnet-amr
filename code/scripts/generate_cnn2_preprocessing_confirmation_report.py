"""Generate the three-seed CNN2 preprocessing confirmation report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from generate_cnn2_preprocessing_report import FOCUS, _replay  # noqa: E402
from na_lmscnet.data import GlobalZScoreStatistics, RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.models import build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402

MODES = ("global_zscore", "per_sample_max_abs")
SEEDS = (13, 37, 73)
DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    parser.add_argument("--seed13-report-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def _statistics(path: Path) -> GlobalZScoreStatistics:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document["statistics"]
    value = GlobalZScoreStatistics(
        channel_mean=tuple(raw["channel_mean"]),
        channel_std=tuple(raw["channel_std"]),
        scalar_count_per_channel=int(raw["scalar_count_per_channel"]),
        split=str(raw["split"]),
        estimator=str(raw["estimator"]),
    )
    if document.get("statistics_sha256") != value.sha256() or value.split != "train":
        raise ValueError("Global z-score statistics artifact is invalid")
    return value


def main() -> int:
    args = parse_args()
    project_commit = _project_commit()
    diagnosis = args.diagnosis_dir.resolve(strict=True)
    seed13_report = args.seed13_report_dir.resolve(strict=True)
    report = args.report_dir.resolve()
    if report.exists():
        raise RuntimeError("Report directory already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.", dir=report.parent))
    confirmation = diagnosis / "preprocessing-confirmation"
    confirmation_protocol = json.loads(
        (confirmation / "confirmation-protocol.json").read_text(encoding="utf-8")
    )
    if confirmation_protocol.get("test_accessed") is not False:
        raise ValueError("Confirmation protocol does not attest test isolation")
    statistics = _statistics(diagnosis / "global-zscore-statistics.json")
    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
    }
    results: dict[tuple[str, int], dict[str, object]] = {}
    try:
        for mode in MODES:
            with RadioML2016HDF5Dataset(
                split="validation", preprocessing=mode,
                global_zscore=statistics if mode == "global_zscore" else None, **common,
            ) as dataset:
                for seed in SEEDS:
                    if seed == 13:
                        config_path = diagnosis / "configs" / f"{mode}.yml"
                        checkpoint = diagnosis / mode / "best.pt"
                    else:
                        run_id = f"{mode}-seed-{seed}"
                        config_path = confirmation / "configs" / f"{run_id}.yml"
                        checkpoint = confirmation / run_id / "best.pt"
                    config = load_experiment_config(config_path)
                    if int(config.training["seed"]) != seed:
                        raise ValueError(f"Seed mismatch in {config_path}")
                    model = build_model(
                        "cnn2", num_classes=11, dropout=float(config.model["dropout"])
                    )
                    results[(mode, seed)] = _replay(
                        model, checkpoint, dataset, torch.device(args.device),
                        int(config.data["batch_size"]),
                        amp=bool(config.training["amp"]),
                    )

        run_rows: list[dict[str, object]] = []
        focus_rows: list[dict[str, object]] = []
        snr_rows: list[dict[str, object]] = []
        for (mode, seed), result in results.items():
            metrics = result["metrics"]
            run_rows.append({
                "preprocessing": mode, "seed": seed, "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "low_snr_accuracy": result["low_snr_accuracy"],
            })
            for label in FOCUS:
                focus_rows.append({
                    "preprocessing": mode, "seed": seed, "class": label,
                    "accuracy": result["class_accuracy"][label],
                    "low_snr_accuracy": result["class_low_snr_accuracy"][label],
                })
            for snr, value in metrics.per_snr_accuracy.items():
                snr_rows.append({
                    "preprocessing": mode, "seed": seed,
                    "snr_db": int(snr), "accuracy": value,
                })

        aggregate_rows: list[dict[str, object]] = []
        for mode in MODES:
            selected = [row for row in run_rows if row["preprocessing"] == mode]
            accuracy_mean, accuracy_std = _mean_std([float(row["accuracy"]) for row in selected])
            f1_mean, f1_std = _mean_std([float(row["macro_f1"]) for row in selected])
            low_mean, low_std = _mean_std(
                [float(row["low_snr_accuracy"]) for row in selected]
            )
            aggregate_rows.append({
                "preprocessing": mode, "seed_count": len(selected),
                "accuracy_mean": accuracy_mean, "accuracy_std": accuracy_std,
                "macro_f1_mean": f1_mean, "macro_f1_std": f1_std,
                "low_snr_accuracy_mean": low_mean, "low_snr_accuracy_std": low_std,
            })

        difference_rows: list[dict[str, object]] = []
        for seed in SEEDS:
            global_row = next(
                row for row in run_rows
                if row["preprocessing"] == "global_zscore" and row["seed"] == seed
            )
            max_row = next(
                row for row in run_rows
                if row["preprocessing"] == "per_sample_max_abs" and row["seed"] == seed
            )
            difference_rows.append({
                "seed": seed,
                "accuracy_global_minus_max_abs": float(global_row["accuracy"]) - float(max_row["accuracy"]),
                "macro_f1_global_minus_max_abs": float(global_row["macro_f1"]) - float(max_row["macro_f1"]),
                "low_snr_global_minus_max_abs": float(global_row["low_snr_accuracy"]) - float(max_row["low_snr_accuracy"]),
            })

        focus_aggregate_rows: list[dict[str, object]] = []
        for mode in MODES:
            for label in FOCUS:
                selected = [
                    row for row in focus_rows
                    if row["preprocessing"] == mode and row["class"] == label
                ]
                accuracy_mean, accuracy_std = _mean_std(
                    [float(row["accuracy"]) for row in selected]
                )
                low_mean, low_std = _mean_std(
                    [float(row["low_snr_accuracy"]) for row in selected]
                )
                focus_aggregate_rows.append({
                    "preprocessing": mode, "class": label,
                    "accuracy_mean": accuracy_mean, "accuracy_std": accuracy_std,
                    "low_snr_accuracy_mean": low_mean, "low_snr_accuracy_std": low_std,
                })

        snr_difference_rows: list[dict[str, object]] = []
        for snr in sorted({int(row["snr_db"]) for row in snr_rows}):
            differences = []
            for seed in SEEDS:
                global_value = next(
                    float(row["accuracy"]) for row in snr_rows
                    if row["preprocessing"] == "global_zscore"
                    and row["seed"] == seed and row["snr_db"] == snr
                )
                max_value = next(
                    float(row["accuracy"]) for row in snr_rows
                    if row["preprocessing"] == "per_sample_max_abs"
                    and row["seed"] == seed and row["snr_db"] == snr
                )
                differences.append(global_value - max_value)
            mean, std = _mean_std(differences)
            snr_difference_rows.append({
                "snr_db": snr, "global_minus_max_abs_mean": mean,
                "global_minus_max_abs_std": std,
                "positive_seed_count": sum(value > 0 for value in differences),
            })

        _write_csv(staging / "confirmation_runs.csv", run_rows)
        _write_csv(staging / "confirmation_aggregate.csv", aggregate_rows)
        _write_csv(staging / "paired_differences.csv", difference_rows)
        _write_csv(staging / "focus_class_runs.csv", focus_rows)
        _write_csv(staging / "focus_class_aggregate.csv", focus_aggregate_rows)
        _write_csv(staging / "per_snr_runs.csv", snr_rows)
        _write_csv(staging / "per_snr_paired_differences.csv", snr_difference_rows)

        source_summary = next(
            row for row in csv.DictReader(
                (seed13_report / "summary.csv").open(encoding="utf-8")
            ) if row["experiment"] == "source-aligned"
        )
        lines = [
            "# CNN2 Three-Seed Preprocessing Confirmation and Final-Protocol Diagnosis", "",
            "This report reads only the frozen validation data and the completed source-evaluation half; it neither constructs nor reads the project test dataset. The bound seed-13 report contains the four controlled comparisons and confusion matrices. Because global z-score and per-sample maximum-amplitude normalization were close and produced conflicting class-level directions, only these two candidates were extended to seeds 37 and 73 under the pre-specified rule.",
            "", "## Three-Seed Candidate Summary", "",
            "| Preprocessing | Overall accuracy | Macro-F1 | Low-SNR accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in aggregate_rows:
            lines.append(
                f"| {row['preprocessing']} | {row['accuracy_mean']:.6f} +/- {row['accuracy_std']:.6f} | "
                f"{row['macro_f1_mean']:.6f} +/- {row['macro_f1_std']:.6f} | "
                f"{row['low_snr_accuracy_mean']:.6f} +/- {row['low_snr_accuracy_std']:.6f} |"
            )
        lines.extend([
            "", "Differences are defined as `global_zscore - per_sample_max_abs`.", "",
            "| Seed | Overall delta | Macro-F1 delta | Low-SNR delta |",
            "| ---: | ---: | ---: | ---: |",
        ])
        for row in difference_rows:
            lines.append(
                f"| {row['seed']} | {row['accuracy_global_minus_max_abs']:+.6f} | "
                f"{row['macro_f1_global_minus_max_abs']:+.6f} | "
                f"{row['low_snr_global_minus_max_abs']:+.6f} |"
            )
        lines.extend(["", "## Three-Seed Means for Focus Classes", ""])
        for low_only, heading in ((False, "Overall"), (True, "Low SNR")):
            lines.extend([
                f"### {heading}", "",
                "| Preprocessing | AM-DSB | AM-SSB | WBFM | QAM16 | QAM64 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ])
            for mode in MODES:
                values = {
                    row["class"]: row["low_snr_accuracy_mean" if low_only else "accuracy_mean"]
                    for row in focus_aggregate_rows if row["preprocessing"] == mode
                }
                lines.append(
                    f"| {mode} | {values['AM-DSB']:.4f} | {values['AM-SSB']:.4f} | "
                    f"{values['WBFM']:.4f} | {values['QAM16']:.4f} | {values['QAM64']:.4f} |"
                )
            lines.append("")
        lines.extend([
            "## Source-Aligned Reproduction (Reported Separately)", "",
            "This group is a controlled PyTorch adaptation of the original raw/50-50/VT-CNN2/Adam/validation-loss protocol. To preserve the project test lock, the 50/50 division is reconstructed only within the train-plus-validation pool. It is not part of the controlled comparison above.",
            "", "| Accuracy | Macro-F1 | Low-SNR accuracy | Samples |",
            "| ---: | ---: | ---: | ---: |",
            f"| {float(source_summary['accuracy']):.6f} | {float(source_summary['macro_f1']):.6f} | {float(source_summary['low_snr_accuracy']):.6f} | {source_summary['sample_count']} |",
            "", "## Decision Boundary", "",
            "The final freeze must consider paired seed directions, per-SNR differences, and focus classes together. It must not rely on a small single-seed difference or an absolute accuracy threshold outside the protocol. The final decision and affected rerun scope must be recorded in the governing plan.", "",
        ])
        (staging / "confirmation-report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "purpose": "cnn2_preprocessing_three_seed_confirmation_report",
            "test_accessed": False,
            "report_generation_project_commit": project_commit,
            "bindings": {
                "confirmation_protocol_sha256": _sha256_file(
                    confirmation / "confirmation-protocol.json"
                ),
                "seed13_report_manifest_sha256": _sha256_file(
                    seed13_report / "report-manifest.json"
                ),
                "split_manifest_sha256": _sha256_file(args.split_manifest),
            },
        }
        (staging / "report-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(report))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"report_dir": str(report), "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
