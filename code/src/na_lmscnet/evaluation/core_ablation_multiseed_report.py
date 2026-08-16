"""Formal five-seed replay and paired bootstrap report for module 7 core ablations."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from na_lmscnet.data.contracts import ModulationSample
from na_lmscnet.evaluation.core_ablation_report import (
    FIXED_AVERAGE_MODEL,
    REFERENCE_MODEL,
    WO_MULTI_SCALE_MODEL,
    CoreAblationReportError,
    validate_split_audit_pair,
)
from na_lmscnet.evaluation.efficiency import count_macs, count_parameters
from na_lmscnet.evaluation.na_lmscnet_report import ALL_SNRS, LOW_SNR_VALUES, _line_plot
from na_lmscnet.evaluation.snr_auxiliary_ablation_report import (
    _assert_replay_matches_metrics,
    _load_json,
    _load_multiseed_inputs,
    _mapping,
    _replay_run,
    _sha256_file,
    _write_csv,
)
from na_lmscnet.models import build_model

EXPECTED_SEEDS = (13, 37, 73, 101, 137)
BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 10_000
LOW_SNR_BOOTSTRAP_VALUES = tuple(LOW_SNR_VALUES)
FORMAL_METRICS = ("accuracy", "macro_f1", "low_snr_accuracy")


class CoreAblationMultiseedReportError(CoreAblationReportError):
    """Raised when formal module-7 evidence violates the frozen protocol."""


def _git_blob(project_root: Path, commit: str, relative_path: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise CoreAblationMultiseedReportError(
            f"Could not read {relative_path} from commit {commit}"
        ) from error
    return result.stdout


def _symbol_fingerprint(source: bytes, name: str) -> str:
    tree = ast.parse(source.decode("utf-8"), type_comments=True)
    node = next(
        (
            candidate
            for candidate in tree.body
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and candidate.name == name
        ),
        None,
    )
    if node is None:
        raise CoreAblationMultiseedReportError(f"Missing source symbol {name}")
    canonical = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _legacy_branch_fingerprint(source: bytes, function_name: str) -> str:
    """Fingerprint the NA-LMSCNet branch retained inside an extended training loop."""

    tree = ast.parse(source.decode("utf-8"), type_comments=True)
    function = next(
        (
            candidate
            for candidate in tree.body
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            and candidate.name == function_name
        ),
        None,
    )
    if function is None:
        raise CoreAblationMultiseedReportError(f"Missing training symbol {function_name}")
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        body = ast.dump(
            ast.Module(body=node.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        )
        if "NoiseAwareJointLoss" in body:
            return hashlib.sha256(body.encode("utf-8")).hexdigest()
    raise CoreAblationMultiseedReportError(f"Missing NA-LMSCNet branch in {function_name}")


def source_equivalence_audit(
    *,
    project_root: Path,
    reference_training_commit: str,
    formal_training_commit: str,
    reference_checkpoint_path: Path,
) -> dict[str, object]:
    """Prove that the reusable full-model reference is semantically unchanged.

    The ablation implementation necessarily extends model/engine registries.  The audit therefore
    compares the full-model AST symbols and the legacy NA-LMSCNet training branches instead of
    requiring byte-identical files, while keeping exact fingerprints for shared protocol code.
    """

    project_root = project_root.resolve(strict=True)
    reference_checkpoint_path = reference_checkpoint_path.resolve(strict=True)
    model_path = "code/src/na_lmscnet/models/na_lmscnet.py"
    engine_path = "code/src/na_lmscnet/training/engine.py"
    current_model = _git_blob(project_root, formal_training_commit, model_path)
    current_engine = _git_blob(project_root, formal_training_commit, engine_path)
    old_model = _git_blob(project_root, reference_training_commit, model_path)
    old_engine = _git_blob(project_root, reference_training_commit, engine_path)
    exact_protocol_files = {
        "joint_loss": "code/src/na_lmscnet/training/losses.py",
        "data_preprocessing": "code/src/na_lmscnet/data/radioml_hdf5.py",
        "validation_metrics": "code/src/na_lmscnet/training/metrics.py",
    }
    checks: dict[str, object] = {}
    for label, relative_path in exact_protocol_files.items():
        old_source = _git_blob(project_root, reference_training_commit, relative_path)
        current_source = _git_blob(project_root, formal_training_commit, relative_path)
        symbols = {
            "joint_loss": ("NoiseAwareJointLoss",),
            "data_preprocessing": ("preprocess_iq",),
            "validation_metrics": ("classification_metrics",),
        }[label]
        old_fingerprints = {name: _symbol_fingerprint(old_source, name) for name in symbols}
        current_fingerprints = {name: _symbol_fingerprint(current_source, name) for name in symbols}
        checks[label] = {
            "equivalent": old_fingerprints == current_fingerprints,
            "reference_fingerprints": old_fingerprints,
            "current_fingerprints": current_fingerprints,
        }
    forward_symbols = ("_DynamicMultiScaleBlock", "NALMSCNet")
    old_forward = {name: _symbol_fingerprint(old_model, name) for name in forward_symbols}
    current_forward = {name: _symbol_fingerprint(current_model, name) for name in forward_symbols}
    checks["forward_structure"] = {
        "equivalent": old_forward == current_forward,
        "reference_fingerprints": old_forward,
        "current_fingerprints": current_forward,
    }
    old_train_branch = _legacy_branch_fingerprint(old_engine, "_train_epoch")
    current_train_branch = _legacy_branch_fingerprint(current_engine, "_train_epoch")
    old_validation_branch = _legacy_branch_fingerprint(old_engine, "_validate_epoch")
    current_validation_branch = _legacy_branch_fingerprint(current_engine, "_validate_epoch")
    checks["training_loop"] = {
        "equivalent": old_train_branch == current_train_branch
        and old_validation_branch == current_validation_branch,
        "reference_train_branch": old_train_branch,
        "current_train_branch": current_train_branch,
        "reference_validation_branch": old_validation_branch,
        "current_validation_branch": current_validation_branch,
    }
    checks["augmentation"] = checks["training_loop"] | {
        "equivalent": _symbol_fingerprint(old_engine, "augment_iq_batch")
        == _symbol_fingerprint(current_engine, "augment_iq_batch"),
        "reference_fingerprint": _symbol_fingerprint(old_engine, "augment_iq_batch"),
        "current_fingerprint": _symbol_fingerprint(current_engine, "augment_iq_batch"),
    }
    checks["checkpoint_selection"] = {
        "equivalent": _symbol_fingerprint(old_engine, "run_training")
        == _symbol_fingerprint(current_engine, "run_training"),
        "reference_fingerprint": _symbol_fingerprint(old_engine, "run_training"),
        "current_fingerprint": _symbol_fingerprint(current_engine, "run_training"),
    }
    checkpoint = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise CoreAblationMultiseedReportError("Reference checkpoint has no model_state_dict")
    reference_shapes = {
        str(key): list(value.shape)
        for key, value in checkpoint["model_state_dict"].items()
        if isinstance(value, torch.Tensor)
    }
    current_model_instance = build_model("na_lmscnet", num_classes=11, dropout=0.2)
    current_shapes = {
        str(key): list(value.shape) for key, value in current_model_instance.state_dict().items()
    }
    checks["parameter_state_dict"] = {
        "equivalent": reference_shapes == current_shapes,
        "reference_key_count": len(reference_shapes),
        "current_key_count": len(current_shapes),
        "reference_shapes_sha256": hashlib.sha256(
            json.dumps(reference_shapes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "current_shapes_sha256": hashlib.sha256(
            json.dumps(current_shapes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    equivalent = all(
        bool(_mapping(value, label).get("equivalent")) for label, value in checks.items()
    )
    return {
        "schema_version": 1,
        "reference_training_commit": reference_training_commit,
        "formal_training_commit": formal_training_commit,
        "method": "AST semantic fingerprints plus reference checkpoint key/shape compatibility",
        "checks": checks,
        "equivalent": equivalent,
        "reuse_authorized": equivalent,
    }


def _macro_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    confusion = np.bincount(
        targets.astype(np.int64) * num_classes + predictions.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = (
        2.0 * true_positive
        + confusion.sum(axis=0)
        - true_positive
        + confusion.sum(axis=1)
        - true_positive
    )
    scores = np.divide(
        2.0 * true_positive, denominator, out=np.zeros(num_classes), where=denominator > 0
    )
    return float(scores.mean())


def paired_hierarchical_bootstrap(
    *,
    reference_replays: Mapping[int, Mapping[str, object]],
    variant_replays: Mapping[int, Mapping[str, object]],
    metric: str,
    snr_values: Iterable[int] | None = None,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    num_classes: int = 11,
) -> dict[str, object]:
    """Run the frozen seed-level and (modulation, SNR)-level paired bootstrap."""

    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError("metric must be accuracy or macro_f1")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    seeds = tuple(EXPECTED_SEEDS)
    if set(reference_replays) != set(seeds) or set(variant_replays) != set(seeds):
        raise ValueError("reference and variant replays must contain the five formal seeds")
    allowed_snr = None if snr_values is None else frozenset(int(value) for value in snr_values)
    strata: list[tuple[int, int]] = []
    values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for seed_value in seeds:
        reference = reference_replays[seed_value]
        variant = variant_replays[seed_value]
        ref_pred = np.asarray(reference["predictions"], dtype=np.int64)
        var_pred = np.asarray(variant["predictions"], dtype=np.int64)
        targets = np.asarray(reference["targets"], dtype=np.int64)
        modulation = np.asarray(reference["modulation"], dtype=np.int64)
        snr_db = np.asarray(reference["snr_db"], dtype=np.int64)
        if not (
            np.array_equal(
                np.asarray(reference["sample_ids"], dtype=object),
                np.asarray(variant["sample_ids"], dtype=object),
            )
            and np.array_equal(targets, np.asarray(variant["targets"], dtype=np.int64))
            and np.array_equal(modulation, np.asarray(variant["modulation"], dtype=np.int64))
            and np.array_equal(snr_db, np.asarray(variant["snr_db"], dtype=np.int64))
        ):
            raise ValueError(f"paired replay alignment differs for seed {seed_value}")
        if allowed_snr is not None:
            mask = np.isin(snr_db, list(allowed_snr))
        else:
            mask = np.ones(len(snr_db), dtype=bool)
        current_strata = sorted(
            set(zip(modulation[mask].tolist(), snr_db[mask].tolist(), strict=True))
        )
        if seed_value == seeds[0]:
            strata = current_strata
        elif current_strata != strata:
            raise ValueError("paired replay strata differ between seeds")
        values.append((ref_pred[mask], var_pred[mask], targets[mask]))
    if not strata:
        raise ValueError("bootstrap selection contains no strata")
    observed_differences = []
    for seed_value in seeds:
        reference = reference_replays[seed_value]
        variant = variant_replays[seed_value]
        ref_pred = np.asarray(reference["predictions"], dtype=np.int64)
        var_pred = np.asarray(variant["predictions"], dtype=np.int64)
        targets = np.asarray(reference["targets"], dtype=np.int64)
        snr_db = np.asarray(reference["snr_db"], dtype=np.int64)
        if allowed_snr is not None:
            selection = np.isin(snr_db, list(allowed_snr))
            ref_pred, var_pred, targets = (
                ref_pred[selection],
                var_pred[selection],
                targets[selection],
            )
        if metric == "accuracy":
            observed_differences.append(
                float(
                    np.mean(
                        (ref_pred == targets).astype(np.int8)
                        - (var_pred == targets).astype(np.int8)
                    )
                )
            )
        else:
            observed_differences.append(
                _macro_f1(ref_pred, targets, num_classes)
                - _macro_f1(var_pred, targets, num_classes)
            )
    per_seed_strata: list[dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for (ref_pred, var_pred, targets), seed_value in zip(values, seeds, strict=True):
        modulation = np.asarray(reference_replays[seed_value]["modulation"], dtype=np.int64)
        snr_db = np.asarray(reference_replays[seed_value]["snr_db"], dtype=np.int64)
        if allowed_snr is not None:
            selection = np.isin(snr_db, list(allowed_snr))
            modulation, snr_db = modulation[selection], snr_db[selection]
        by_stratum = {}
        for key in strata:
            indices = np.flatnonzero((modulation == key[0]) & (snr_db == key[1]))
            by_stratum[key] = (ref_pred[indices], var_pred[indices], targets[indices])
        per_seed_strata.append(by_stratum)
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
    bootstrap_values = np.empty(resamples, dtype=np.float64)
    chunk_size = 64
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        chunk = stop - start
        if metric == "accuracy":
            numerators = np.zeros(chunk, dtype=np.float64)
            denominators = np.zeros(chunk, dtype=np.float64)
        else:
            ref_confusions = np.zeros((chunk, num_classes, num_classes), dtype=np.float64)
            var_confusions = np.zeros((chunk, num_classes, num_classes), dtype=np.float64)
        for key in strata:
            for source_seed_index in range(len(seeds)):
                ref_values, var_values, targets = per_seed_strata[source_seed_index][key]
                if len(ref_values) == 0:
                    continue
                sample_indices = rng.integers(
                    0, len(ref_values), size=(chunk, len(seeds), len(ref_values))
                )
                for source_position in range(len(seeds)):
                    positions = np.flatnonzero(
                        seed_draws[start:stop, source_position] == source_seed_index
                    )
                    if len(positions) == 0:
                        continue
                    picks = sample_indices[positions, source_position]
                    ref_sample = ref_values[picks]
                    var_sample = var_values[picks]
                    target_sample = targets[picks]
                    if metric == "accuracy":
                        numerators[positions] += (ref_sample == target_sample).sum(axis=1)
                        numerators[positions] -= (var_sample == target_sample).sum(axis=1)
                        denominators[positions] += len(ref_values)
                    else:
                        ref_flat = target_sample * num_classes + ref_sample
                        var_flat = target_sample * num_classes + var_sample
                        for local_index, position in enumerate(positions):
                            ref_confusions[position] += np.bincount(
                                ref_flat[local_index], minlength=num_classes * num_classes
                            ).reshape(num_classes, num_classes)
                            var_confusions[position] += np.bincount(
                                var_flat[local_index], minlength=num_classes * num_classes
                            ).reshape(num_classes, num_classes)
        if metric == "accuracy":
            bootstrap_values[start:stop] = numerators / denominators
        else:
            bootstrap_values[start:stop] = np.asarray(
                [
                    _macro_f1_from_confusions(ref_confusions[index], num_classes)
                    - _macro_f1_from_confusions(var_confusions[index], num_classes)
                    for index in range(chunk)
                ],
                dtype=np.float64,
            )
    return {
        "metric": metric,
        "snr_values": None if allowed_snr is None else sorted(allowed_snr),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "mean_difference": float(np.mean(observed_differences)),
        "bootstrap_mean": float(bootstrap_values.mean()),
        "ci_lower": float(np.percentile(bootstrap_values, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_values, 97.5)),
    }


def _paired_accuracy_bootstrap_suite(
    *,
    reference_replays: Mapping[int, Mapping[str, object]],
    variant_replays: Mapping[int, Mapping[str, object]],
    seed: int,
    resamples: int,
) -> dict[str, dict[str, object]]:
    """Compute overall, low-SNR, and all per-SNR accuracy intervals in one replay pass."""

    seeds = tuple(EXPECTED_SEEDS)
    if set(reference_replays) != set(seeds) or set(variant_replays) != set(seeds):
        raise ValueError("reference and variant replays must contain the five formal seeds")
    strata = []
    per_seed_strata = []
    for seed_value in seeds:
        reference = reference_replays[seed_value]
        variant = variant_replays[seed_value]
        sample_ids = np.asarray(reference["sample_ids"], dtype=object)
        targets = np.asarray(reference["targets"], dtype=np.int64)
        modulation = np.asarray(reference["modulation"], dtype=np.int64)
        snr_db = np.asarray(reference["snr_db"], dtype=np.int64)
        ref_pred = np.asarray(reference["predictions"], dtype=np.int64)
        var_pred = np.asarray(variant["predictions"], dtype=np.int64)
        if not (
            np.array_equal(sample_ids, np.asarray(variant["sample_ids"], dtype=object))
            and np.array_equal(targets, np.asarray(variant["targets"], dtype=np.int64))
            and np.array_equal(modulation, np.asarray(variant["modulation"], dtype=np.int64))
            and np.array_equal(snr_db, np.asarray(variant["snr_db"], dtype=np.int64))
        ):
            raise ValueError(f"paired replay alignment differs for seed {seed_value}")
        current_strata = sorted(set(zip(modulation.tolist(), snr_db.tolist(), strict=True)))
        if not strata:
            strata = current_strata
        elif strata != current_strata:
            raise ValueError("paired replay strata differ between seeds")
        by_stratum = {}
        for key in strata:
            indices = np.flatnonzero((modulation == key[0]) & (snr_db == key[1]))
            by_stratum[key] = (ref_pred[indices] == targets[indices]).astype(np.int8) - (
                var_pred[indices] == targets[indices]
            ).astype(np.int8)
        per_seed_strata.append(by_stratum)
    observed_by_snr = {snr: 0.0 for snr in ALL_SNRS}
    observed_by_snr_count = {snr: 0 for snr in ALL_SNRS}
    for by_stratum in per_seed_strata:
        for (_modulation, snr), correct_difference in by_stratum.items():
            observed_by_snr[snr] += float(correct_difference.sum())
            observed_by_snr_count[snr] += len(correct_difference)
    observed_total = sum(observed_by_snr.values()) / sum(observed_by_snr_count.values())
    observed_low = sum(observed_by_snr[snr] for snr in LOW_SNR_BOOTSTRAP_VALUES) / sum(
        observed_by_snr_count[snr] for snr in LOW_SNR_BOOTSTRAP_VALUES
    )
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
    snr_columns = {snr: index + 1 for index, snr in enumerate(ALL_SNRS)}
    low_snr_column = 1 + len(ALL_SNRS)
    values = np.empty((resamples, 2 + len(ALL_SNRS)), dtype=np.float64)
    chunk_size = 64
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        chunk = stop - start
        numerators = np.zeros((chunk, 2 + len(ALL_SNRS)), dtype=np.float64)
        denominators = np.zeros((chunk, 2 + len(ALL_SNRS)), dtype=np.float64)
        for key in strata:
            snr_column = snr_columns[key[1]]
            for source_seed_index in range(len(seeds)):
                correct_difference = per_seed_strata[source_seed_index][key]
                sample_indices = rng.integers(
                    0,
                    len(correct_difference),
                    size=(chunk, len(seeds), len(correct_difference)),
                )
                for source_position in range(len(seeds)):
                    positions = np.flatnonzero(
                        seed_draws[start:stop, source_position] == source_seed_index
                    )
                    if len(positions) == 0:
                        continue
                    delta_sum = correct_difference[sample_indices[positions, source_position]].sum(
                        axis=1
                    )
                    numerators[positions, 0] += delta_sum
                    numerators[positions, snr_column] += delta_sum
                    denominators[positions, 0] += len(correct_difference)
                    denominators[positions, snr_column] += len(correct_difference)
                    if key[1] in LOW_SNR_BOOTSTRAP_VALUES:
                        numerators[positions, low_snr_column] += delta_sum
                        denominators[positions, low_snr_column] += len(correct_difference)
        values[start:stop] = numerators / denominators
    result = {
        "accuracy": _bootstrap_summary(
            values[:, 0], "accuracy", None, seed, resamples, observed_total
        )
    }
    for snr in ALL_SNRS:
        result[f"snr_{snr:+d}"] = _bootstrap_summary(
            values[:, snr_columns[snr]],
            "accuracy",
            [snr],
            seed,
            resamples,
            observed_by_snr[snr] / observed_by_snr_count[snr],
        )
    result["low_snr_accuracy"] = _bootstrap_summary(
        values[:, low_snr_column],
        "low_snr_accuracy",
        list(LOW_SNR_BOOTSTRAP_VALUES),
        seed,
        resamples,
        observed_low,
    )
    return result


def _bootstrap_summary(
    values: np.ndarray,
    metric: str,
    snr_values: list[int] | None,
    seed: int,
    resamples: int,
    observed_difference: float,
) -> dict[str, object]:
    return {
        "metric": metric,
        "snr_values": snr_values,
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "mean_difference": observed_difference,
        "bootstrap_mean": float(values.mean()),
        "ci_lower": float(np.percentile(values, 2.5)),
        "ci_upper": float(np.percentile(values, 97.5)),
    }


def _macro_f1_from_confusions(confusion: np.ndarray, num_classes: int) -> float:
    true_positive = np.diag(confusion)
    denominator = (
        2.0 * true_positive
        + confusion.sum(axis=0)
        - true_positive
        + confusion.sum(axis=1)
        - true_positive
    )
    scores = np.divide(
        2.0 * true_positive, denominator, out=np.zeros(num_classes), where=denominator > 0
    )
    return float(scores.mean())


def formal_contribution_decision(
    *,
    low_snr_ci: Mapping[str, object],
    accuracy_ci: Mapping[str, object],
    macro_f1_ci: Mapping[str, object],
    positive_low_snr_seed_count: int,
) -> dict[str, object]:
    """Apply the frozen independent-contribution criterion."""

    low_snr_clear = float(low_snr_ci["ci_lower"]) > 0.0
    overall_clear = float(accuracy_ci["ci_lower"]) > 0.0 or float(macro_f1_ci["ci_lower"]) > 0.0
    direction_clear = positive_low_snr_seed_count >= 4
    qualifies = low_snr_clear and overall_clear and direction_clear
    if qualifies:
        status = "stable_independent_contribution"
    elif (
        float(low_snr_ci["mean_difference"]) > 0.0
        or float(accuracy_ci["mean_difference"]) > 0.0
        or float(macro_f1_ci["mean_difference"]) > 0.0
    ):
        status = "evidence_insufficient"
    else:
        status = "independent_contribution_not_supported"
    return {
        "qualifies": qualifies,
        "status": status,
        "low_snr_ci_lower_gt_zero": low_snr_clear,
        "accuracy_or_macro_f1_ci_lower_gt_zero": overall_clear,
        "positive_low_snr_seed_count": positive_low_snr_seed_count,
        "positive_low_snr_seed_fraction": positive_low_snr_seed_count / 5.0,
    }


def _formal_mainline(decisions: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    multi_scale = bool(decisions[WO_MULTI_SCALE_MODEL]["qualifies"])
    dynamic = bool(decisions[FIXED_AVERAGE_MODEL]["qualifies"])
    if multi_scale and dynamic:
        scope = "lightweight_multi_scale_dynamic_fusion"
    elif multi_scale:
        scope = "lightweight_multi_scale_fusion"
    else:
        scope = "lightweight_performance_complexity_tradeoff_only"
    return {
        "scope": scope,
        "multi_scale_contribution_supported": multi_scale,
        "dynamic_fusion_contribution_supported": dynamic,
        "noise_aware_core_restored": False,
    }


def _measure_latency(
    model: torch.nn.Module, device: torch.device, warmup: int, iterations: int
) -> float:
    model.eval().to(device)
    inputs = torch.zeros((1, 2, 128), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def _efficiency(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    parameters = count_parameters(model)
    macs = count_macs(model, (1, 2, 128), torch.device("cpu"))
    gpu_latency = (
        _measure_latency(model, device, warmup, iterations)
        if device.type == "cuda"
        else float("nan")
    )
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        cpu_latency = _measure_latency(model, torch.device("cpu"), warmup, iterations)
    finally:
        torch.set_num_threads(old_threads)
    return {
        "parameter_count": parameters,
        "macs": macs,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "gpu_latency_ms": gpu_latency,
        "gpu_throughput_samples_per_s": 1000.0 / gpu_latency
        if math.isfinite(gpu_latency)
        else float("nan"),
        "cpu_latency_ms": cpu_latency,
        "gpu_warmup": warmup,
        "gpu_iterations": iterations,
        "cpu_threads": 1,
    }


def _replay_alignment(replays: Mapping[str, Mapping[int, Mapping[str, object]]]) -> None:
    baseline = replays[REFERENCE_MODEL][EXPECTED_SEEDS[0]]
    baseline_ids = np.asarray(baseline["sample_ids"], dtype=object)
    baseline_targets = np.asarray(baseline["targets"], dtype=np.int64)
    baseline_modulation = np.asarray(baseline["modulation"], dtype=np.int64)
    baseline_snr = np.asarray(baseline["snr_db"], dtype=np.int64)
    for model_replays in replays.values():
        for seed in EXPECTED_SEEDS:
            replay = model_replays[seed]
            if not (
                np.array_equal(baseline_ids, np.asarray(replay["sample_ids"], dtype=object))
                and np.array_equal(baseline_targets, np.asarray(replay["targets"], dtype=np.int64))
                and np.array_equal(
                    baseline_modulation, np.asarray(replay["modulation"], dtype=np.int64)
                )
                and np.array_equal(baseline_snr, np.asarray(replay["snr_db"], dtype=np.int64))
            ):
                raise CoreAblationMultiseedReportError("Formal replays are not sample-aligned")


def _metric_row(replay: Mapping[str, object], seed: int, model: str) -> dict[str, object]:
    metrics = replay["metrics"]
    return {
        "run_id": f"{model}-seed-{seed}",
        "model": model,
        "seed": seed,
        "accuracy": float(metrics.accuracy),
        "macro_f1": float(metrics.macro_f1),
        "low_snr_accuracy": float(replay["low_snr_accuracy"]),
        "snr_mae_db": metrics.snr_mae_db,
        "sample_count": int(metrics.sample_count),
    }


def _plot_formal_accuracy(path: Path, rows: list[dict[str, object]]) -> None:
    # Keep the figure compact: one panel with the five-seed mean curves.
    mean_rows = []
    for snr in ALL_SNRS:
        row = {"snr_db": snr}
        for model in (REFERENCE_MODEL, WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
            values = [
                float(item["accuracy"])
                for item in rows
                if item["model"] == model and item["snr_db"] == snr
            ]
            row[model] = float(np.mean(values))
        mean_rows.append(row)
    _line_plot(
        path,
        title="Module 7 five-seed validation accuracy",
        panels=[
            (
                "Mean over seeds",
                [
                    (
                        "NA-LMSCNet",
                        "#146c94",
                        [(row["snr_db"], row[REFERENCE_MODEL]) for row in mean_rows],
                    ),
                    (
                        "w/o multi-scale",
                        "#c5542d",
                        [(row["snr_db"], row[WO_MULTI_SCALE_MODEL]) for row in mean_rows],
                    ),
                    (
                        "fixed-average",
                        "#397a4f",
                        [(row["snr_db"], row[FIXED_AVERAGE_MODEL]) for row in mean_rows],
                    ),
                ],
            )
        ],
        y_label="Accuracy",
    )


def generate_core_ablation_multiseed_report(
    *,
    reference_output_root: Path,
    reference_training_commit: str,
    wo_multi_scale_output_root: Path,
    wo_multi_scale_training_commit: str,
    fixed_average_output_root: Path,
    fixed_average_training_commit: str,
    report_dir: Path,
    hdf5_path: Path,
    split_manifest_path: Path,
    leakage_audit_path: Path,
    validation_dataset: Dataset[ModulationSample],
    project_root: Path,
    report_generation_commit: str,
    device: torch.device,
    warmup: int = 100,
    iterations: int = 1000,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Generate the complete formal module-7 evidence package."""

    project_root = project_root.resolve(strict=True)
    report_dir = report_dir.resolve()
    if report_dir == project_root or project_root in report_dir.parents:
        raise CoreAblationMultiseedReportError("Report output must be outside repository")
    if report_dir.exists():
        raise CoreAblationMultiseedReportError(f"Refusing to overwrite report: {report_dir}")
    if warmup < 1 or iterations < 1:
        raise CoreAblationMultiseedReportError("Latency warmup and iterations must be positive")
    if wo_multi_scale_training_commit != fixed_average_training_commit:
        raise CoreAblationMultiseedReportError(
            "Both formal ablations must share one training commit"
        )
    validate_split_audit_pair(split_manifest_path, leakage_audit_path)
    split_sha256 = _sha256_file(split_manifest_path)
    assignment_sha256 = validation_dataset.assignment_sha256
    audit = _mapping(_load_json(leakage_audit_path, "leakage audit"), "leakage audit")
    if audit.get("split_manifest_sha256") != validation_dataset.split_manifest_sha256:
        raise CoreAblationMultiseedReportError("Validation split differs from leakage audit")
    reference_inputs = _load_multiseed_inputs(
        output_root=reference_output_root.resolve(strict=True),
        expected_model=REFERENCE_MODEL,
        expected_commit=reference_training_commit,
        split_sha256=split_sha256,
        assignment_sha256=assignment_sha256,
    )
    wo_inputs = _load_multiseed_inputs(
        output_root=wo_multi_scale_output_root.resolve(strict=True),
        expected_model=WO_MULTI_SCALE_MODEL,
        expected_commit=wo_multi_scale_training_commit,
        split_sha256=split_sha256,
        assignment_sha256=assignment_sha256,
    )
    fixed_inputs = _load_multiseed_inputs(
        output_root=fixed_average_output_root.resolve(strict=True),
        expected_model=FIXED_AVERAGE_MODEL,
        expected_commit=fixed_average_training_commit,
        split_sha256=split_sha256,
        assignment_sha256=assignment_sha256,
    )
    audit_result = source_equivalence_audit(
        project_root=project_root,
        reference_training_commit=reference_training_commit,
        formal_training_commit=wo_multi_scale_training_commit,
        reference_checkpoint_path=reference_inputs[0][3],
    )
    if not audit_result["equivalent"]:
        raise CoreAblationMultiseedReportError(
            "Source-equivalence audit failed; full NA-LMSCNet five-seed rerun is required"
        )
    inputs_by_model = {
        REFERENCE_MODEL: reference_inputs,
        WO_MULTI_SCALE_MODEL: wo_inputs,
        FIXED_AVERAGE_MODEL: fixed_inputs,
    }
    replay_by_model: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    run_rows: list[dict[str, object]] = []
    efficiency_rows: list[dict[str, object]] = []
    for model_name, inputs in inputs_by_model.items():
        for seed, config, metrics, checkpoint in inputs:
            replay = _replay_run(
                config=config, checkpoint_path=checkpoint, dataset=validation_dataset, device=device
            )
            _assert_replay_matches_metrics(replay, metrics)
            replay_by_model[model_name][seed] = replay
            run_rows.append(_metric_row(replay, seed, model_name))
        first_replay = replay_by_model[model_name][EXPECTED_SEEDS[0]]
        efficiency = _efficiency(first_replay["model"], inputs[0][3], device, warmup, iterations)
        for seed, _, _, checkpoint in inputs:
            efficiency_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "run_id": f"{model_name}-seed-{seed}",
                    **efficiency,
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                }
            )
    _replay_alignment(replay_by_model)
    summary_rows = []
    for model_name in (REFERENCE_MODEL, WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
        selected = [row for row in run_rows if row["model"] == model_name]
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
                "snr_mae_db_mean": float(
                    np.mean(
                        [row["snr_mae_db"] for row in selected if row["snr_mae_db"] is not None]
                    )
                )
                if any(row["snr_mae_db"] is not None for row in selected)
                else None,
                "snr_mae_db_std": float(
                    np.std(
                        [row["snr_mae_db"] for row in selected if row["snr_mae_db"] is not None],
                        ddof=1,
                    )
                )
                if sum(row["snr_mae_db"] is not None for row in selected) > 1
                else None,
            }
        )
    paired_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    decisions: dict[str, dict[str, object]] = {}
    accuracy_suites: dict[str, dict[str, dict[str, object]]] = {}
    for variant in (WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
        for seed in EXPECTED_SEEDS:
            reference_row = next(
                row for row in run_rows if row["model"] == REFERENCE_MODEL and row["seed"] == seed
            )
            variant_row = next(
                row for row in run_rows if row["model"] == variant and row["seed"] == seed
            )
            paired_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "accuracy_difference": float(reference_row["accuracy"])
                    - float(variant_row["accuracy"]),
                    "macro_f1_difference": float(reference_row["macro_f1"])
                    - float(variant_row["macro_f1"]),
                    "low_snr_accuracy_difference": float(reference_row["low_snr_accuracy"])
                    - float(variant_row["low_snr_accuracy"]),
                }
            )
        accuracy_suites[variant] = _paired_accuracy_bootstrap_suite(
            reference_replays=replay_by_model[REFERENCE_MODEL],
            variant_replays=replay_by_model[variant],
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
        metric_bootstraps = {
            "accuracy": accuracy_suites[variant]["accuracy"],
            "macro_f1": paired_hierarchical_bootstrap(
                reference_replays=replay_by_model[REFERENCE_MODEL],
                variant_replays=replay_by_model[variant],
                metric="macro_f1",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
        }
        bootstrap_rows.extend(
            {"variant": variant, **metric_bootstraps[metric]} for metric in ("accuracy", "macro_f1")
        )
        low_bootstrap = accuracy_suites[variant]["low_snr_accuracy"]
        bootstrap_rows.append({"variant": variant, "metric": "low_snr_accuracy", **low_bootstrap})
        variant_seed_rows = [row for row in paired_rows if row["variant"] == variant]
        decisions[variant] = formal_contribution_decision(
            low_snr_ci=low_bootstrap,
            accuracy_ci=metric_bootstraps["accuracy"],
            macro_f1_ci=metric_bootstraps["macro_f1"],
            positive_low_snr_seed_count=sum(
                float(row["low_snr_accuracy_difference"]) > 0 for row in variant_seed_rows
            ),
        )
    per_snr_rows = []
    per_snr_paired_rows = []
    per_snr_bootstrap_rows = []
    for snr in ALL_SNRS:
        for seed in EXPECTED_SEEDS:
            for model_name in (REFERENCE_MODEL, WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
                metric = replay_by_model[model_name][seed]["metrics"]
                per_snr_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "snr_db": snr,
                        "accuracy": metric.per_snr_accuracy[f"{snr:+d}"],
                    }
                )
            ref = next(
                row
                for row in per_snr_rows
                if row["model"] == REFERENCE_MODEL and row["seed"] == seed and row["snr_db"] == snr
            )
            for variant in (WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
                var = next(
                    row
                    for row in per_snr_rows
                    if row["model"] == variant and row["seed"] == seed and row["snr_db"] == snr
                )
                per_snr_paired_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "snr_db": snr,
                        "accuracy_difference": float(ref["accuracy"]) - float(var["accuracy"]),
                    }
                )
        for variant in (WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
            ci = accuracy_suites[variant][f"snr_{snr:+d}"]
            per_snr_bootstrap_rows.append({"variant": variant, "snr_db": snr, **ci})
    mainline = _formal_mainline(decisions)
    efficiency_models = []
    for model_name in (REFERENCE_MODEL, WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
        selected = [row for row in efficiency_rows if row["model"] == model_name]
        first = selected[0]
        efficiency_models.append(
            {
                "model": model_name,
                "run_count": len(selected),
                "parameter_count": first["parameter_count"],
                "macs": first["macs"],
                "checkpoint_size_bytes_mean": float(
                    np.mean([row["checkpoint_size_bytes"] for row in selected])
                ),
                "gpu_latency_ms_mean": float(
                    np.nanmean([row["gpu_latency_ms"] for row in selected])
                ),
                "cpu_latency_ms_mean": float(
                    np.nanmean([row["cpu_latency_ms"] for row in selected])
                ),
            }
        )
    replay_rows = []
    baseline = replay_by_model[REFERENCE_MODEL][EXPECTED_SEEDS[0]]
    for seed in EXPECTED_SEEDS:
        for index, sample_id in enumerate(baseline["sample_ids"]):
            row = {
                "seed": seed,
                "sample_id": sample_id,
                "modulation": int(baseline["modulation"][index]),
                "snr_db": int(baseline["snr_db"][index]),
                "target": int(baseline["targets"][index]),
            }
            for model_name in (REFERENCE_MODEL, WO_MULTI_SCALE_MODEL, FIXED_AVERAGE_MODEL):
                row[f"{model_name}_prediction"] = int(
                    replay_by_model[model_name][seed]["predictions"][index]
                )
                row[f"{model_name}_correct"] = int(row[f"{model_name}_prediction"] == row["target"])
            replay_rows.append(row)
    summary = {
        "schema_version": 2,
        "purpose": "core_multiscale_dynamic_fusion_five_seed_formal_report",
        "test_accessed": False,
        "test_dataset_constructed": False,
        "bindings": {
            "reference_training_commit": reference_training_commit,
            "formal_training_commit": wo_multi_scale_training_commit,
            "report_generation_commit": report_generation_commit,
            "reference_summary_sha256": _sha256_file(
                reference_output_root / "multi-seed-summary.json"
            ),
            "wo_multi_scale_summary_sha256": _sha256_file(
                wo_multi_scale_output_root / "multi-seed-summary.json"
            ),
            "fixed_average_summary_sha256": _sha256_file(
                fixed_average_output_root / "multi-seed-summary.json"
            ),
            "split_manifest_sha256": split_sha256,
            "assignment_sha256": assignment_sha256,
            "hdf5_file_sha256": _sha256_file(hdf5_path),
            "leakage_audit_sha256": _sha256_file(leakage_audit_path),
            "seeds": list(EXPECTED_SEEDS),
            "validation_sample_count": len(validation_dataset),
        },
        "bootstrap_protocol": {
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
            "strata": "(modulation, SNR)",
            "low_snr_values_db": list(LOW_SNR_BOOTSTRAP_VALUES),
            "ci": "percentile_95",
        },
        "source_equivalence_audit": audit_result,
        "comparison": summary_rows,
        "paired_differences": paired_rows,
        "bootstrap_ci": bootstrap_rows,
        "formal_decisions": decisions,
        "mainline_decision": mainline,
    }
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=report_dir.parent))
    try:
        (staging / "figures").mkdir()
        _write_csv(staging / "summary_runs.csv", list(run_rows[0]), run_rows)
        _write_csv(staging / "summary_models.csv", list(summary_rows[0]), summary_rows)
        _write_csv(staging / "replay_validation.csv", list(replay_rows[0]), replay_rows)
        _write_csv(staging / "paired_differences.csv", list(paired_rows[0]), paired_rows)
        _write_csv(staging / "bootstrap_ci.csv", list(bootstrap_rows[0]), bootstrap_rows)
        _write_csv(staging / "per_snr_accuracy.csv", list(per_snr_rows[0]), per_snr_rows)
        _write_csv(
            staging / "per_snr_paired_differences.csv",
            list(per_snr_paired_rows[0]),
            per_snr_paired_rows,
        )
        _write_csv(
            staging / "per_snr_bootstrap_ci.csv",
            list(per_snr_bootstrap_rows[0]),
            per_snr_bootstrap_rows,
        )
        _write_csv(staging / "efficiency.csv", list(efficiency_rows[0]), efficiency_rows)
        _write_csv(staging / "efficiency_models.csv", list(efficiency_models[0]), efficiency_models)
        _plot_formal_accuracy(staging / "figures" / "per_snr_accuracy.png", per_snr_rows)
        (staging / "source-equivalence-audit.json").write_text(
            json.dumps(audit_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        lines = [
            "# Module 7 Core Ablations: Five-Seed Formal Validation",
            "",
            "Only the frozen validation split was used; per-sample replay neither constructed nor accessed a test dataset.",
            "",
            f"- mainline decision: `{mainline['scope']}`",
            f"- bootstrap: seed `{bootstrap_seed}`, `{bootstrap_resamples}` resamples, paired stratification by `(modulation, SNR)`",
        ]
        for variant, decision in decisions.items():
            lines.extend(
                [
                    "",
                    f"## {variant}",
                    f"- formal status: `{decision['status']}`",
                    f"- positive low-SNR seed count: `{decision['positive_low_snr_seed_count']}/5`",
                ]
            )
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
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CoreAblationMultiseedReportError",
    "EXPECTED_SEEDS",
    "formal_contribution_decision",
    "generate_core_ablation_multiseed_report",
    "paired_hierarchical_bootstrap",
    "source_equivalence_audit",
]
