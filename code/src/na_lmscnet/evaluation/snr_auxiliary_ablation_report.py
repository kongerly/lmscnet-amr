"""Validation-only report for the w/o SNR auxiliary ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from na_lmscnet.data.contracts import ModulationSample
from na_lmscnet.evaluation.efficiency import count_macs, count_parameters
from na_lmscnet.evaluation.na_lmscnet_report import ALL_SNRS, LOW_SNR_VALUES, _font, _line_plot
from na_lmscnet.models import build_model
from na_lmscnet.training.engine import (
    ExperimentConfig,
    experiment_config_sha256,
    load_experiment_config,
)
from na_lmscnet.training.metrics import classification_metrics
from na_lmscnet.training.multiseed import _best_epoch_record

REFERENCE_MODEL = "na_lmscnet"
ABLATION_MODEL = "na_lmscnet_wo_snr_auxiliary"
CLEAR_LOW_SNR_DROP = 0.010
CLEAR_OVERALL_DROP = 0.005
BASICALLY_UNCHANGED = 0.005


class SNRAuxiliaryAblationReportError(ValueError):
    """Raised when ablation report inputs violate the frozen protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SNRAuxiliaryAblationReportError(f"{field} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SNRAuxiliaryAblationReportError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise SNRAuxiliaryAblationReportError(f"{field} must be an object")
    return value


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SNRAuxiliaryAblationReportError(f"{field} must be a string-keyed mapping")
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_loader(
    dataset: Dataset[ModulationSample], config: ExperimentConfig
) -> DataLoader[ModulationSample]:
    workers = int(config.data["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config.data["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config.data["pin_memory"]),
        persistent_workers=workers > 0,
    )


def _validate_run(
    *,
    config_path: Path,
    run_dir: Path,
    expected_model: str,
    expected_commit: str,
    split_sha256: str,
    assignment_sha256: str,
    expected_seed: int = 13,
) -> tuple[ExperimentConfig, dict[str, Any], Path]:
    config = load_experiment_config(config_path)
    metrics = _load_json(run_dir / "metrics.json", f"{expected_model} metrics")
    checkpoint_path = run_dir / "best.pt"
    if config.model["name"] != expected_model or int(config.training["seed"]) != expected_seed:
        raise SNRAuxiliaryAblationReportError(
            f"{expected_model} must use the frozen seed {expected_seed} config"
        )
    bindings = _mapping(metrics.get("bindings"), f"{expected_model} metrics bindings")
    expected_bindings = {
        "experiment_config_sha256": experiment_config_sha256(config_path),
        "split_manifest_sha256": split_sha256,
        "assignment_sha256": assignment_sha256,
        "project_commit": expected_commit,
        "seed": expected_seed,
    }
    if bindings != expected_bindings:
        raise SNRAuxiliaryAblationReportError(f"{expected_model} metrics bindings differ")
    artifacts = _mapping(metrics.get("artifacts"), f"{expected_model} artifacts")
    if (
        metrics.get("test_accessed") is not False
        or artifacts.get("checkpoint_filename") != "best.pt"
        or artifacts.get("checkpoint_sha256") != _sha256_file(checkpoint_path)
    ):
        raise SNRAuxiliaryAblationReportError(f"{expected_model} artifact/test binding is invalid")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("model_name") != expected_model
        or checkpoint.get("bindings") != expected_bindings
        or not isinstance(checkpoint.get("model_state_dict"), dict)
    ):
        raise SNRAuxiliaryAblationReportError(f"{expected_model} checkpoint binding is invalid")
    return config, metrics, checkpoint_path


def _replay_run(
    *,
    config: ExperimentConfig,
    checkpoint_path: Path,
    dataset: Dataset[ModulationSample],
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    modulations: list[torch.Tensor] = []
    snrs: list[torch.Tensor] = []
    snr_hats: list[torch.Tensor] = []
    sample_ids: list[str] = []
    amp = bool(config.training["amp"]) and device.type == "cuda"
    with torch.inference_mode():
        for batch in _build_loader(dataset, config):
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(batch["iq"].to(device, non_blocking=True))
            logits = outputs["logits"]
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(batch["modulation"].cpu())
            modulations.append(batch["modulation"].cpu())
            snrs.append(batch["snr"].cpu())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            if "snr_hat" in outputs:
                snr_hats.append(outputs["snr_hat"].cpu())
    if not predictions:
        raise SNRAuxiliaryAblationReportError("Validation replay consumed zero samples")
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    modulation_tensor = torch.cat(modulations)
    snr_tensor = torch.cat(snrs)
    snr_hat_tensor = torch.cat(snr_hats) if snr_hats else None
    metrics = classification_metrics(
        prediction_tensor,
        target_tensor,
        snr_tensor,
        num_classes=int(config.model["num_classes"]),
        snr_prediction_db=snr_hat_tensor,
    )
    low_mask = torch.zeros_like(snr_tensor, dtype=torch.bool)
    for value in LOW_SNR_VALUES:
        low_mask |= snr_tensor == value
    low_snr_accuracy = float(
        (prediction_tensor[low_mask] == target_tensor[low_mask]).double().mean()
    )
    return {
        "model": model.cpu(),
        "metrics": metrics,
        "low_snr_accuracy": low_snr_accuracy,
        "sample_ids": sample_ids,
        "predictions": prediction_tensor.numpy(),
        "targets": target_tensor.numpy(),
        "modulation": modulation_tensor.numpy(),
        "snr_db": snr_tensor.numpy(),
        "snr_hat_db": snr_hat_tensor.numpy() if snr_hat_tensor is not None else None,
    }


def _assert_replay_matches_metrics(replay: dict[str, Any], metrics: dict[str, Any]) -> None:
    best = _best_epoch_record(metrics)
    recorded = _mapping(best.get("validation"), "best validation")
    actual = replay["metrics"]
    for field in ("accuracy", "macro_f1"):
        if not math.isclose(
            float(getattr(actual, field)), float(recorded[field]), rel_tol=3e-5, abs_tol=3e-5
        ):
            raise SNRAuxiliaryAblationReportError(f"Replay differs from recorded {field}")
    recorded_per_snr = _mapping(recorded.get("per_snr_accuracy"), "recorded per-SNR accuracy")
    for snr, accuracy in actual.per_snr_accuracy.items():
        if not math.isclose(
            accuracy, float(recorded_per_snr[f"{int(snr):+d}"]), rel_tol=3e-5, abs_tol=3e-5
        ):
            raise SNRAuxiliaryAblationReportError(f"Replay differs at SNR {snr}")
    if actual.snr_mae_db is None:
        if recorded.get("snr_mae_db") is not None:
            raise SNRAuxiliaryAblationReportError("Ablation unexpectedly records SNR MAE")
    elif not math.isclose(
        actual.snr_mae_db, float(recorded["snr_mae_db"]), rel_tol=3e-5, abs_tol=3e-5
    ):
        raise SNRAuxiliaryAblationReportError("Replay differs from recorded SNR MAE")


def _snr_hat_distribution(true_snr: np.ndarray, snr_hat: np.ndarray) -> list[dict[str, object]]:
    if true_snr.shape != snr_hat.shape or true_snr.ndim != 1:
        raise SNRAuxiliaryAblationReportError("SNR distribution arrays must be aligned vectors")
    rows = []
    for value in ALL_SNRS:
        selected = snr_hat[true_snr == value].astype(np.float64)
        if selected.size == 0:
            raise SNRAuxiliaryAblationReportError(f"Missing validation predictions at {value} dB")
        quantiles = np.quantile(selected, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            {
                "true_snr_db": value,
                "sample_count": int(selected.size),
                "mean_snr_hat_db": float(selected.mean()),
                "std_snr_hat_db": float(selected.std(ddof=1)),
                "min_snr_hat_db": float(selected.min()),
                "q05_snr_hat_db": float(quantiles[0]),
                "q25_snr_hat_db": float(quantiles[1]),
                "median_snr_hat_db": float(quantiles[2]),
                "q75_snr_hat_db": float(quantiles[3]),
                "q95_snr_hat_db": float(quantiles[4]),
                "max_snr_hat_db": float(selected.max()),
                "mae_db": float(np.abs(selected - value).mean()),
            }
        )
    return rows


def screening_decision(
    *,
    reference_accuracy: float,
    reference_macro_f1: float,
    reference_low_snr: float,
    ablation_accuracy: float,
    ablation_macro_f1: float,
    ablation_low_snr: float,
) -> dict[str, object]:
    deltas = {
        "accuracy": reference_accuracy - ablation_accuracy,
        "macro_f1": reference_macro_f1 - ablation_macro_f1,
        "low_snr_accuracy": reference_low_snr - ablation_low_snr,
    }
    clear_drop = deltas["low_snr_accuracy"] >= CLEAR_LOW_SNR_DROP and (
        deltas["accuracy"] >= CLEAR_OVERALL_DROP or deltas["macro_f1"] >= CLEAR_OVERALL_DROP
    )
    unchanged = all(abs(value) <= BASICALLY_UNCHANGED for value in deltas.values())
    if clear_drop:
        action = "run_five_seed_formal_validation"
        reason = "clear_seed13_drop"
    elif unchanged:
        action = "narrow_to_lightweight_multiscale_dynamic_fusion"
        reason = "seed13_basically_unchanged"
    elif any(value < -BASICALLY_UNCHANGED for value in deltas.values()):
        action = "narrow_to_lightweight_multiscale_dynamic_fusion"
        reason = "ablation_not_worse"
    else:
        action = "run_five_seed_formal_validation"
        reason = "seed13_borderline"
    return {"action": action, "reason": reason, "reference_minus_ablation": deltas}


def _plot_accuracy(path: Path, rows: list[dict[str, object]]) -> None:
    reference = [(float(row["snr_db"]), float(row["reference_accuracy"])) for row in rows]
    ablation = [(float(row["snr_db"]), float(row["ablation_accuracy"])) for row in rows]
    _line_plot(
        path,
        title="w/o SNR auxiliary seed-13 validation accuracy",
        panels=[
            (
                "Frozen split, seed 13",
                [
                    ("NA-LMSCNet", "#146c94", reference),
                    ("w/o SNR aux", "#c5542d", ablation),
                ],
            )
        ],
        y_label="Accuracy",
    )


def _plot_snr_hat(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 1500, 760
    left, right, top, bottom = 110, width - 70, 100, height - 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, label_font = _font(28), _font(18), _font(15)
    draw.text(
        (width / 2, 24),
        "NA-LMSCNet seed-13 SNR prediction distribution",
        font=title_font,
        fill="#17212b",
        anchor="ma",
    )
    draw.rectangle((left, top, right, bottom), outline="#7a8793", width=2)
    y_min, y_max = -20.0, 18.0

    def x_coord(value: int) -> float:
        return left + (right - left) * ALL_SNRS.index(value) / (len(ALL_SNRS) - 1)

    def y_coord(value: float) -> float:
        return bottom - (bottom - top) * (value - y_min) / (y_max - y_min)

    for value in range(-20, 20, 2):
        x = x_coord(value)
        draw.line((x, bottom, x, bottom + 6), fill="#7a8793", width=1)
        draw.text((x, bottom + 11), str(value), font=label_font, fill="#52606d", anchor="ma")
    for value in range(-20, 19, 4):
        y = y_coord(float(value))
        draw.line((left, y, right, y), fill="#d9dee3", width=1)
        draw.text((left - 12, y), str(value), font=label_font, fill="#52606d", anchor="rm")
    identity = [(x_coord(value), y_coord(float(value))) for value in ALL_SNRS]
    draw.line(identity, fill="#7a8793", width=2)
    means = []
    for row in rows:
        x = x_coord(int(row["true_snr_db"]))
        q05, q25 = y_coord(float(row["q05_snr_hat_db"])), y_coord(float(row["q25_snr_hat_db"]))
        median = y_coord(float(row["median_snr_hat_db"]))
        q75, q95 = y_coord(float(row["q75_snr_hat_db"])), y_coord(float(row["q95_snr_hat_db"]))
        draw.line((x, q05, x, q95), fill="#4d6a78", width=3)
        draw.rectangle((x - 10, q75, x + 10, q25), fill="#9ec5d3", outline="#146c94", width=2)
        draw.line((x - 10, median, x + 10, median), fill="#17212b", width=3)
        means.append((x, y_coord(float(row["mean_snr_hat_db"]))))
    draw.line(means, fill="#c5542d", width=4)
    for x, y in means:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#c5542d")
    draw.text(
        (width / 2, height - 38), "True SNR (dB)", font=axis_font, fill="#17212b", anchor="ma"
    )
    draw.text((28, height / 2), "Predicted SNR (dB)", font=axis_font, fill="#17212b", anchor="mm")
    image.save(path, format="PNG")


def generate_snr_auxiliary_ablation_report(
    *,
    reference_config_path: Path,
    reference_run_dir: Path,
    reference_training_commit: str,
    ablation_config_path: Path,
    ablation_run_dir: Path,
    ablation_training_commit: str,
    report_dir: Path,
    hdf5_path: Path,
    split_manifest_path: Path,
    leakage_audit_path: Path,
    validation_dataset: Dataset[ModulationSample],
    project_root: Path,
    report_generation_commit: str,
    device: torch.device,
) -> dict[str, object]:
    project_root = project_root.resolve(strict=True)
    report_dir = report_dir.resolve()
    if report_dir == project_root or project_root in report_dir.parents:
        raise SNRAuxiliaryAblationReportError("Report output must be outside repository")
    if report_dir.exists():
        raise SNRAuxiliaryAblationReportError(f"Refusing to overwrite report: {report_dir}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SNRAuxiliaryAblationReportError("CUDA requested but unavailable")
    split_sha256 = _sha256_file(split_manifest_path)
    assignment_sha256 = validation_dataset.assignment_sha256
    audit = _mapping(_load_json(leakage_audit_path, "leakage audit"), "leakage audit")
    if audit.get("split_manifest_sha256") != validation_dataset.split_manifest_sha256:
        raise SNRAuxiliaryAblationReportError("Validation split differs from leakage audit")
    reference_config, reference_metrics, reference_checkpoint = _validate_run(
        config_path=reference_config_path,
        run_dir=reference_run_dir,
        expected_model=REFERENCE_MODEL,
        expected_commit=reference_training_commit,
        split_sha256=split_sha256,
        assignment_sha256=assignment_sha256,
    )
    ablation_config, ablation_metrics, ablation_checkpoint = _validate_run(
        config_path=ablation_config_path,
        run_dir=ablation_run_dir,
        expected_model=ABLATION_MODEL,
        expected_commit=ablation_training_commit,
        split_sha256=split_sha256,
        assignment_sha256=assignment_sha256,
    )
    reference = _replay_run(
        config=reference_config,
        checkpoint_path=reference_checkpoint,
        dataset=validation_dataset,
        device=device,
    )
    ablation = _replay_run(
        config=ablation_config,
        checkpoint_path=ablation_checkpoint,
        dataset=validation_dataset,
        device=device,
    )
    _assert_replay_matches_metrics(reference, reference_metrics)
    _assert_replay_matches_metrics(ablation, ablation_metrics)
    if ablation["snr_hat_db"] is not None or reference["snr_hat_db"] is None:
        raise SNRAuxiliaryAblationReportError(
            "SNR prediction presence differs from ablation design"
        )
    reference_metrics_value = reference["metrics"]
    ablation_metrics_value = ablation["metrics"]
    decision = screening_decision(
        reference_accuracy=reference_metrics_value.accuracy,
        reference_macro_f1=reference_metrics_value.macro_f1,
        reference_low_snr=reference["low_snr_accuracy"],
        ablation_accuracy=ablation_metrics_value.accuracy,
        ablation_macro_f1=ablation_metrics_value.macro_f1,
        ablation_low_snr=ablation["low_snr_accuracy"],
    )
    reference_model = reference["model"]
    ablation_model = ablation["model"]
    comparison_rows = [
        {
            "model": REFERENCE_MODEL,
            "accuracy": reference_metrics_value.accuracy,
            "macro_f1": reference_metrics_value.macro_f1,
            "low_snr_accuracy": reference["low_snr_accuracy"],
            "snr_mae_db": reference_metrics_value.snr_mae_db,
            "parameter_count": count_parameters(reference_model),
            "macs": count_macs(reference_model, (1, 2, 128), device),
        },
        {
            "model": ABLATION_MODEL,
            "accuracy": ablation_metrics_value.accuracy,
            "macro_f1": ablation_metrics_value.macro_f1,
            "low_snr_accuracy": ablation["low_snr_accuracy"],
            "snr_mae_db": None,
            "parameter_count": count_parameters(ablation_model),
            "macs": count_macs(ablation_model, (1, 2, 128), device),
        },
    ]
    per_snr_rows = [
        {
            "snr_db": snr,
            "reference_accuracy": reference_metrics_value.per_snr_accuracy[f"{snr:+d}"],
            "ablation_accuracy": ablation_metrics_value.per_snr_accuracy[f"{snr:+d}"],
            "reference_minus_ablation": reference_metrics_value.per_snr_accuracy[f"{snr:+d}"]
            - ablation_metrics_value.per_snr_accuracy[f"{snr:+d}"],
        }
        for snr in ALL_SNRS
    ]
    distribution_rows = _snr_hat_distribution(reference["snr_db"], reference["snr_hat_db"])
    prediction_rows = [
        {"sample_id": sample_id, "true_snr_db": int(snr), "snr_hat_db": float(snr_hat)}
        for sample_id, snr, snr_hat in zip(
            reference["sample_ids"], reference["snr_db"], reference["snr_hat_db"], strict=True
        )
    ]
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=report_dir.parent))
    try:
        (staging / "figures").mkdir()
        _write_csv(staging / "comparison.csv", list(comparison_rows[0]), comparison_rows)
        _write_csv(staging / "per_snr_accuracy.csv", list(per_snr_rows[0]), per_snr_rows)
        _write_csv(
            staging / "snr_hat_distribution_by_true_snr.csv",
            list(distribution_rows[0]),
            distribution_rows,
        )
        _write_csv(staging / "snr_hat_predictions.csv", list(prediction_rows[0]), prediction_rows)
        _plot_accuracy(staging / "figures" / "per_snr_accuracy.png", per_snr_rows)
        _plot_snr_hat(staging / "figures" / "snr_hat_distribution.png", distribution_rows)
        summary = {
            "schema_version": 1,
            "purpose": "snr_auxiliary_seed13_ablation_screen",
            "test_accessed": False,
            "test_dataset_constructed": False,
            "bindings": {
                "reference_training_commit": reference_training_commit,
                "ablation_training_commit": ablation_training_commit,
                "report_generation_commit": report_generation_commit,
                "reference_config_sha256": experiment_config_sha256(reference_config_path),
                "ablation_config_sha256": experiment_config_sha256(ablation_config_path),
                "reference_checkpoint_sha256": _sha256_file(reference_checkpoint),
                "ablation_checkpoint_sha256": _sha256_file(ablation_checkpoint),
                "split_manifest_sha256": split_sha256,
                "assignment_sha256": assignment_sha256,
                "hdf5_file_sha256": _sha256_file(hdf5_path),
                "leakage_audit_sha256": _sha256_file(leakage_audit_path),
                "seed": 13,
                "validation_sample_count": len(validation_dataset),
            },
            "screening_rule": {
                "clear_low_snr_drop": CLEAR_LOW_SNR_DROP,
                "clear_accuracy_or_macro_f1_drop": CLEAR_OVERALL_DROP,
                "basically_unchanged_absolute_delta": BASICALLY_UNCHANGED,
            },
            "comparison": comparison_rows,
            "decision": decision,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        lines = [
            "# w/o SNR Auxiliary Seed-13 Validation Screen",
            "",
            "Only the frozen validation split was used; no test dataset was constructed or accessed.",
            "",
            f"- decision: `{decision['action']}`",
            f"- reason: `{decision['reason']}`",
            f"- Accuracy delta (reference - ablation): `{decision['reference_minus_ablation']['accuracy']:.6f}`",
            f"- Macro-F1 delta: `{decision['reference_minus_ablation']['macro_f1']:.6f}`",
            f"- Low-SNR accuracy delta: `{decision['reference_minus_ablation']['low_snr_accuracy']:.6f}`",
            "",
            "`snr_hat_predictions.csv` and `snr_hat_distribution_by_true_snr.csv` contain only the frozen seed-13 SNR predictions from the complete NA-LMSCNet. The ablation has no SNR head, so no synthetic SNR predictions are generated for it.",
        ]
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest = {**summary, "files": files}
        (staging / "report-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        shutil.move(str(staging), str(report_dir))
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_multiseed_inputs(
    *,
    output_root: Path,
    expected_model: str,
    expected_commit: str,
    split_sha256: str,
    assignment_sha256: str,
) -> list[tuple[int, ExperimentConfig, dict[str, Any], Path]]:
    summary = _load_json(output_root / "multi-seed-summary.json", f"{expected_model} summary")
    bindings = _mapping(summary.get("bindings"), f"{expected_model} summary bindings")
    if (
        summary.get("run_count") != 5
        or summary.get("test_accessed") is not False
        or bindings
        != {
            "project_commit": expected_commit,
            "split_manifest_sha256": split_sha256,
            "assignment_sha256": assignment_sha256,
            "seeds": [13, 37, 73, 101, 137],
        }
    ):
        raise SNRAuxiliaryAblationReportError(f"{expected_model} multi-seed bindings differ")
    result = []
    for seed in (13, 37, 73, 101, 137):
        run_id = f"{expected_model}-seed-{seed}"
        record = next((row for row in summary["runs"] if row.get("run_id") == run_id), None)
        if not isinstance(record, dict):
            raise SNRAuxiliaryAblationReportError(f"Missing {expected_model} run {run_id}")
        config_path = output_root / "configs" / str(record["config_filename"])
        config, metrics, checkpoint = _validate_run(
            config_path=config_path,
            run_dir=output_root / run_id,
            expected_model=expected_model,
            expected_commit=expected_commit,
            split_sha256=split_sha256,
            assignment_sha256=assignment_sha256,
            expected_seed=seed,
        )
        if record.get("checkpoint_sha256") != _sha256_file(checkpoint):
            raise SNRAuxiliaryAblationReportError(f"Summary checkpoint digest differs for {run_id}")
        result.append((seed, config, metrics, checkpoint))
    return result


def generate_snr_auxiliary_multiseed_report(
    *,
    reference_output_root: Path,
    reference_training_commit: str,
    ablation_output_root: Path,
    ablation_training_commit: str,
    report_dir: Path,
    hdf5_path: Path,
    split_manifest_path: Path,
    leakage_audit_path: Path,
    validation_dataset: Dataset[ModulationSample],
    project_root: Path,
    report_generation_commit: str,
    device: torch.device,
) -> dict[str, object]:
    project_root = project_root.resolve(strict=True)
    reference_output_root = reference_output_root.resolve(strict=True)
    ablation_output_root = ablation_output_root.resolve(strict=True)
    report_dir = report_dir.resolve()
    if report_dir == project_root or project_root in report_dir.parents:
        raise SNRAuxiliaryAblationReportError("Report output must be outside repository")
    if report_dir.exists():
        raise SNRAuxiliaryAblationReportError(f"Refusing to overwrite report: {report_dir}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SNRAuxiliaryAblationReportError("CUDA requested but unavailable")
    split_sha256 = _sha256_file(split_manifest_path)
    assignment_sha256 = validation_dataset.assignment_sha256
    audit = _mapping(_load_json(leakage_audit_path, "leakage audit"), "leakage audit")
    if audit.get("split_manifest_sha256") != validation_dataset.split_manifest_sha256:
        raise SNRAuxiliaryAblationReportError("Validation split differs from leakage audit")
    reference_inputs = _load_multiseed_inputs(
        output_root=reference_output_root,
        expected_model=REFERENCE_MODEL,
        expected_commit=reference_training_commit,
        split_sha256=split_sha256,
        assignment_sha256=assignment_sha256,
    )
    ablation_inputs = _load_multiseed_inputs(
        output_root=ablation_output_root,
        expected_model=ABLATION_MODEL,
        expected_commit=ablation_training_commit,
        split_sha256=split_sha256,
        assignment_sha256=assignment_sha256,
    )
    reference_by_seed = {
        seed: (config, metrics, checkpoint)
        for seed, config, metrics, checkpoint in reference_inputs
    }
    ablation_by_seed = {
        seed: (config, metrics, checkpoint) for seed, config, metrics, checkpoint in ablation_inputs
    }
    seeds = (13, 37, 73, 101, 137)
    run_rows: list[dict[str, object]] = []
    per_snr_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    distribution_seed_rows: list[dict[str, object]] = []
    all_reference_snr: list[np.ndarray] = []
    all_reference_hat: list[np.ndarray] = []
    model_rows: list[dict[str, object]] = []
    for seed in seeds:
        reference_config, reference_metrics, reference_checkpoint = reference_by_seed[seed]
        ablation_config, ablation_metrics, ablation_checkpoint = ablation_by_seed[seed]
        reference = _replay_run(
            config=reference_config,
            checkpoint_path=reference_checkpoint,
            dataset=validation_dataset,
            device=device,
        )
        ablation = _replay_run(
            config=ablation_config,
            checkpoint_path=ablation_checkpoint,
            dataset=validation_dataset,
            device=device,
        )
        _assert_replay_matches_metrics(reference, reference_metrics)
        _assert_replay_matches_metrics(ablation, ablation_metrics)
        if ablation["snr_hat_db"] is not None or reference["snr_hat_db"] is None:
            raise SNRAuxiliaryAblationReportError(
                "SNR prediction presence differs from ablation design"
            )
        reference_metric = reference["metrics"]
        ablation_metric = ablation["metrics"]
        run_rows.extend(
            [
                {
                    "seed": seed,
                    "model": REFERENCE_MODEL,
                    "accuracy": reference_metric.accuracy,
                    "macro_f1": reference_metric.macro_f1,
                    "low_snr_accuracy": reference["low_snr_accuracy"],
                    "snr_mae_db": reference_metric.snr_mae_db,
                },
                {
                    "seed": seed,
                    "model": ABLATION_MODEL,
                    "accuracy": ablation_metric.accuracy,
                    "macro_f1": ablation_metric.macro_f1,
                    "low_snr_accuracy": ablation["low_snr_accuracy"],
                    "snr_mae_db": None,
                },
            ]
        )
        per_seed_distribution = _snr_hat_distribution(reference["snr_db"], reference["snr_hat_db"])
        distribution_seed_rows.extend({"seed": seed, **row} for row in per_seed_distribution)
        all_reference_snr.append(reference["snr_db"])
        all_reference_hat.append(reference["snr_hat_db"])
        prediction_rows.extend(
            {
                "seed": seed,
                "sample_id": sample_id,
                "true_snr_db": int(snr),
                "snr_hat_db": float(snr_hat),
            }
            for sample_id, snr, snr_hat in zip(
                reference["sample_ids"], reference["snr_db"], reference["snr_hat_db"], strict=True
            )
        )
        for snr in ALL_SNRS:
            per_snr_rows.extend(
                [
                    {
                        "seed": seed,
                        "snr_db": snr,
                        "model": REFERENCE_MODEL,
                        "accuracy": reference_metric.per_snr_accuracy[f"{snr:+d}"],
                    },
                    {
                        "seed": seed,
                        "snr_db": snr,
                        "model": ABLATION_MODEL,
                        "accuracy": ablation_metric.per_snr_accuracy[f"{snr:+d}"],
                    },
                ]
            )
        if not model_rows:
            model_rows = [
                {
                    "model": REFERENCE_MODEL,
                    "parameter_count": count_parameters(reference["model"]),
                    "macs": count_macs(reference["model"], (1, 2, 128), device),
                },
                {
                    "model": ABLATION_MODEL,
                    "parameter_count": count_parameters(ablation["model"]),
                    "macs": count_macs(ablation["model"], (1, 2, 128), device),
                },
            ]
    summary_rows = []
    for model_name in (REFERENCE_MODEL, ABLATION_MODEL):
        selected = [row for row in run_rows if row["model"] == model_name]
        efficiency = next(row for row in model_rows if row["model"] == model_name)
        summary_rows.append(
            {
                "model": model_name,
                "seed_count": len(selected),
                "accuracy_mean": float(np.mean([row["accuracy"] for row in selected])),
                "accuracy_std": float(np.std([row["accuracy"] for row in selected], ddof=1)),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in selected])),
                "macro_f1_std": float(np.std([row["macro_f1"] for row in selected], ddof=1)),
                "low_snr_accuracy_mean": float(
                    np.mean([row["low_snr_accuracy"] for row in selected])
                ),
                "low_snr_accuracy_std": float(
                    np.std([row["low_snr_accuracy"] for row in selected], ddof=1)
                ),
                "snr_mae_db_mean": None
                if model_name == ABLATION_MODEL
                else float(np.mean([row["snr_mae_db"] for row in selected])),
                "snr_mae_db_std": None
                if model_name == ABLATION_MODEL
                else float(np.std([row["snr_mae_db"] for row in selected], ddof=1)),
                **efficiency,
            }
        )
    reference_summary = next(row for row in summary_rows if row["model"] == REFERENCE_MODEL)
    ablation_summary = next(row for row in summary_rows if row["model"] == ABLATION_MODEL)
    decision = screening_decision(
        reference_accuracy=reference_summary["accuracy_mean"],
        reference_macro_f1=reference_summary["macro_f1_mean"],
        reference_low_snr=reference_summary["low_snr_accuracy_mean"],
        ablation_accuracy=ablation_summary["accuracy_mean"],
        ablation_macro_f1=ablation_summary["macro_f1_mean"],
        ablation_low_snr=ablation_summary["low_snr_accuracy_mean"],
    )
    if decision["reason"] == "clear_seed13_drop":
        decision = {
            **decision,
            "action": "retain_noise_aware_core_after_five_seed_validation",
            "reason": "clear_five_seed_drop",
        }
    elif decision["reason"] == "seed13_basically_unchanged":
        decision = {
            **decision,
            "action": "narrow_to_lightweight_multiscale_dynamic_fusion",
            "reason": "five_seed_basically_unchanged",
        }
    else:
        decision = {
            **decision,
            "action": "do_not_overclaim_noise_aware_mechanism",
            "reason": "five_seed_not_clear_drop",
        }
    per_snr_summary = []
    for snr in ALL_SNRS:
        for model_name in (REFERENCE_MODEL, ABLATION_MODEL):
            values = [
                row["accuracy"]
                for row in per_snr_rows
                if row["snr_db"] == snr and row["model"] == model_name
            ]
            per_snr_summary.append(
                {
                    "snr_db": snr,
                    "model": model_name,
                    "accuracy_mean": float(np.mean(values)),
                    "accuracy_std": float(np.std(values, ddof=1)),
                    "seed_count": len(values),
                }
            )
    distribution_rows = _snr_hat_distribution(
        np.concatenate(all_reference_snr), np.concatenate(all_reference_hat)
    )
    plot_rows = [
        {
            "snr_db": snr,
            "reference_accuracy": next(
                row["accuracy_mean"]
                for row in per_snr_summary
                if row["snr_db"] == snr and row["model"] == REFERENCE_MODEL
            ),
            "ablation_accuracy": next(
                row["accuracy_mean"]
                for row in per_snr_summary
                if row["snr_db"] == snr and row["model"] == ABLATION_MODEL
            ),
        }
        for snr in ALL_SNRS
    ]
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=report_dir.parent))
    try:
        (staging / "figures").mkdir()
        _write_csv(staging / "summary_runs.csv", list(run_rows[0]), run_rows)
        _write_csv(staging / "summary_models.csv", list(summary_rows[0]), summary_rows)
        _write_csv(staging / "efficiency.csv", list(model_rows[0]), model_rows)
        _write_csv(staging / "per_snr_accuracy.csv", list(per_snr_rows[0]), per_snr_rows)
        _write_csv(
            staging / "per_snr_accuracy_summary.csv", list(per_snr_summary[0]), per_snr_summary
        )
        _write_csv(staging / "snr_hat_predictions.csv", list(prediction_rows[0]), prediction_rows)
        _write_csv(
            staging / "snr_hat_distribution_by_true_snr_seed.csv",
            list(distribution_seed_rows[0]),
            distribution_seed_rows,
        )
        _write_csv(
            staging / "snr_hat_distribution_by_true_snr.csv",
            list(distribution_rows[0]),
            distribution_rows,
        )
        _plot_accuracy(staging / "figures" / "per_snr_accuracy.png", plot_rows)
        _plot_snr_hat(staging / "figures" / "snr_hat_distribution.png", distribution_rows)
        summary = {
            "schema_version": 1,
            "purpose": "snr_auxiliary_five_seed_ablation_report",
            "test_accessed": False,
            "test_dataset_constructed": False,
            "bindings": {
                "reference_training_commit": reference_training_commit,
                "ablation_training_commit": ablation_training_commit,
                "report_generation_commit": report_generation_commit,
                "reference_summary_sha256": _sha256_file(
                    reference_output_root / "multi-seed-summary.json"
                ),
                "ablation_summary_sha256": _sha256_file(
                    ablation_output_root / "multi-seed-summary.json"
                ),
                "split_manifest_sha256": split_sha256,
                "assignment_sha256": assignment_sha256,
                "hdf5_file_sha256": _sha256_file(hdf5_path),
                "leakage_audit_sha256": _sha256_file(leakage_audit_path),
                "seeds": list(seeds),
                "validation_sample_count": len(validation_dataset),
            },
            "screening_rule": {
                "clear_low_snr_drop": CLEAR_LOW_SNR_DROP,
                "clear_accuracy_or_macro_f1_drop": CLEAR_OVERALL_DROP,
                "basically_unchanged_absolute_delta": BASICALLY_UNCHANGED,
            },
            "comparison": summary_rows,
            "decision": decision,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        lines = [
            "# w/o SNR Auxiliary Five-Seed Validation Report",
            "",
            "Only the frozen validation split was used; no test dataset was constructed or accessed.",
            "",
            f"- decision: `{decision['action']}`",
            f"- reason: `{decision['reason']}`",
            f"- Accuracy delta (reference - ablation): `{decision['reference_minus_ablation']['accuracy']:.6f}`",
            f"- Macro-F1 delta: `{decision['reference_minus_ablation']['macro_f1']:.6f}`",
            f"- Low-SNR accuracy delta: `{decision['reference_minus_ablation']['low_snr_accuracy']:.6f}`",
            "",
            "`snr_hat_predictions.csv` and the quantile table contain only the five frozen checkpoints from the complete NA-LMSCNet; the ablation has no SNR head.",
        ]
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest = {**summary, "files": files}
        (staging / "report-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        shutil.move(str(staging), str(report_dir))
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "SNRAuxiliaryAblationReportError",
    "generate_snr_auxiliary_ablation_report",
    "generate_snr_auxiliary_multiseed_report",
    "screening_decision",
]
