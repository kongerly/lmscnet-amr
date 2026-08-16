"""Fail-closed experiment freeze manifest and one-shot test authorization."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from na_lmscnet.models import build_model
from na_lmscnet.training import load_experiment_config
from na_lmscnet.training.engine import experiment_config_sha256

SEEDS = (13, 37, 73, 101, 137)
TEST_MODELS = ("lmscnet_s2", "se_msfn_1d")
LOW_SNR_VALUES = (-10, -8, -6, -4, -2, 0)
BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 10_000
PREPROCESSING_MODE = "per_sample_max_abs"
ASSIGNMENT_SHA256 = "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941"
DISALLOWED_PROCESS_MARKERS = (
    "run_final_validation_family",
    "run_multi_seed",
    "run_validation_sweep",
    "train_baseline",
    "run_final_paired_bootstrap",
    "run_extended_paired_bootstrap",
    "run_radioml_2018_replication",
    "monitor_",
)


class ExperimentFreezeError(ValueError):
    """Raised when the experiment cannot be frozen or test access cannot be authorized."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExperimentFreezeError(f"Could not read {field}: {error}") from error
    if not isinstance(value, dict):
        raise ExperimentFreezeError(f"{field} must contain a JSON object")
    return value


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise ExperimentFreezeError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def require_clean_commit(project_root: Path, expected_commit: str | None = None) -> str:
    root = project_root.resolve(strict=True)
    status = _git(root, "status", "--porcelain")
    if status:
        raise ExperimentFreezeError("Experiment freeze requires a clean Git worktree")
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise ExperimentFreezeError(
            "Current Git commit differs from the frozen implementation commit"
        )
    if len(commit) != 40:
        raise ExperimentFreezeError("Git did not return a full commit identifier")
    return commit


def _regular_file(path: Path, field: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ExperimentFreezeError(f"{field} must be a regular file")
    return resolved


def file_binding(path: Path, field: str) -> dict[str, Any]:
    resolved = _regular_file(path, field)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _model_from_config(config: Any) -> torch.nn.Module:
    return build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model.get("expansion", 1.25)),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
    )


def _validate_run(
    *, model: str, seed: int, config_path: Path, metrics_path: Path, checkpoint_path: Path
) -> dict[str, Any]:
    config_path = _regular_file(config_path, "selected config")
    metrics_path = _regular_file(metrics_path, "selected metrics")
    checkpoint_path = _regular_file(checkpoint_path, "selected checkpoint")
    config = load_experiment_config(config_path)
    metrics = load_json(metrics_path, "selected metrics")
    if config.model["name"] != model or int(config.training["seed"]) != seed:
        raise ExperimentFreezeError(f"Selected config identity differs for {model} seed {seed}")
    if config.data["assignment_sha256"] != ASSIGNMENT_SHA256:
        raise ExperimentFreezeError("Selected config assignment differs from the frozen split")
    if metrics.get("test_accessed") is not False:
        raise ExperimentFreezeError("Selected metrics do not prove test_accessed=false")
    bindings = metrics.get("bindings")
    artifacts = metrics.get("artifacts")
    if not isinstance(bindings, dict) or not isinstance(artifacts, dict):
        raise ExperimentFreezeError("Selected metrics bindings are incomplete")
    config_sha256 = experiment_config_sha256(config_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        bindings.get("assignment_sha256") != ASSIGNMENT_SHA256
        or bindings.get("seed") != seed
        or bindings.get("experiment_config_sha256") != config_sha256
        or bindings.get("data_protocol") != {"preprocessing_mode": PREPROCESSING_MODE}
        or artifacts.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ExperimentFreezeError(f"Selected run bindings differ for {model} seed {seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ExperimentFreezeError("Selected checkpoint schema is invalid")
    if checkpoint.get("model_name") != model or checkpoint.get("bindings") != bindings:
        raise ExperimentFreezeError("Selected checkpoint internal bindings differ from metrics")
    candidate = _model_from_config(config)
    candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    candidate.eval()
    with torch.inference_mode():
        output = candidate(torch.zeros((1, 2, 128), dtype=torch.float32))
    logits = output["logits"] if isinstance(output, dict) else output
    if logits.shape != (1, 11) or not bool(torch.isfinite(logits).all()):
        raise ExperimentFreezeError("Selected checkpoint failed the synthetic forward preflight")
    return {
        "model": model,
        "seed": seed,
        "training_commit": str(bindings.get("project_commit")),
        "training_split_manifest_sha256": str(bindings.get("split_manifest_sha256")),
        "config": file_binding(config_path, "selected config"),
        "metrics": file_binding(metrics_path, "selected metrics"),
        "checkpoint": file_binding(checkpoint_path, "selected checkpoint"),
    }


def collect_selected_runs(
    final_queue_root: Path, extended_queue_root: Path
) -> list[dict[str, Any]]:
    roots = {
        "lmscnet_s2": final_queue_root.resolve(strict=True) / "final-family-multiseed",
        "se_msfn_1d": extended_queue_root.resolve(strict=True) / "multiseed",
    }
    runs: list[dict[str, Any]] = []
    for model in TEST_MODELS:
        root = roots[model]
        summary = load_json(root / "multi-seed-summary.json", f"{model} multi-seed summary")
        if summary.get("test_accessed") is not False:
            raise ExperimentFreezeError(f"{model} summary does not prove test_accessed=false")
        entries = {
            int(entry["seed"]): entry
            for entry in summary.get("runs", [])
            if isinstance(entry, dict) and entry.get("model") == model
        }
        if set(entries) != set(SEEDS):
            raise ExperimentFreezeError(f"Selected runs are incomplete for {model}")
        for seed in SEEDS:
            entry = entries[seed]
            run_id = str(entry["run_id"])
            runs.append(
                _validate_run(
                    model=model,
                    seed=seed,
                    config_path=root / "configs" / str(entry["config_filename"]),
                    metrics_path=root / run_id / "metrics.json",
                    checkpoint_path=root / run_id / "best.pt",
                )
            )
    return runs


def _validate_report(path: Path, name: str) -> dict[str, Any]:
    report = load_json(path, name)
    is_validation_report = report.get("test_accessed") is False
    is_complete_recovery_manifest = (
        report.get("purpose") == "radioml_2018_validation_replay_recovery"
        and report.get("all_sha256_match") is True
        and report.get("complete_evidence_bundle") is True
        and isinstance(report.get("files"), dict)
        and bool(report["files"])
    )
    if not (is_validation_report or is_complete_recovery_manifest):
        raise ExperimentFreezeError(f"{name} does not prove test_accessed=false")
    return {"name": name, **file_binding(path, name)}


def build_freeze_manifest(
    *,
    project_root: Path,
    hdf5_path: Path,
    conversion_manifest_path: Path,
    split_manifest_path: Path,
    leakage_audit_path: Path,
    final_queue_root: Path,
    extended_queue_root: Path,
    reports: Iterable[tuple[str, Path]],
    consumption_marker_path: Path,
    test_output_dir: Path,
) -> dict[str, Any]:
    commit = require_clean_commit(project_root)
    split = load_json(split_manifest_path, "2016.10A split manifest")
    leakage = load_json(leakage_audit_path, "2016.10A leakage audit")
    if (
        split.get("digests", {}).get("assignment_sha256") != ASSIGNMENT_SHA256
        or leakage.get("split_manifest_sha256") != canonical_json_sha256(split)
        or split.get("test_isolation", {}).get("test_metrics_before_freeze") != "forbidden"
    ):
        raise ExperimentFreezeError("2016.10A split or isolation binding differs")
    report_bindings = [_validate_report(path, name) for name, path in reports]
    names = [item["name"] for item in report_bindings]
    if len(names) != len(set(names)):
        raise ExperimentFreezeError("Report binding names must be unique")
    marker = consumption_marker_path.resolve()
    output = test_output_dir.resolve()
    root = project_root.resolve(strict=True)
    for path, field in ((marker, "consumption marker"), (output, "test output")):
        if path == root or root in path.parents:
            raise ExperimentFreezeError(f"{field} must remain outside the repository")
        if path.exists():
            raise ExperimentFreezeError(f"{field} already exists")
        if not path.parent.is_dir():
            raise ExperimentFreezeError(f"{field} parent directory must already exist")
    return {
        "schema_version": 1,
        "purpose": "experiment_freeze_manifest_v1",
        "implementation_commit": commit,
        "dataset": {
            "dataset_id": "radioml_2016_10a",
            "preprocessing_mode": PREPROCESSING_MODE,
            "assignment_sha256": ASSIGNMENT_SHA256,
            "hdf5": file_binding(hdf5_path, "2016.10A HDF5"),
            "conversion_manifest": file_binding(
                conversion_manifest_path, "2016.10A conversion manifest"
            ),
            "split_manifest": file_binding(split_manifest_path, "2016.10A split manifest"),
            "leakage_audit": file_binding(leakage_audit_path, "2016.10A leakage audit"),
        },
        "selection": {
            "final_model": "lmscnet_s2",
            "strongest_fair_baseline": "se_msfn_1d",
            "seeds": list(SEEDS),
            "runs": collect_selected_runs(final_queue_root, extended_queue_root),
        },
        "reports": report_bindings,
        "test_protocol": {
            "models": list(TEST_MODELS),
            "seeds": list(SEEDS),
            "metrics": [
                "overall_accuracy",
                "macro_f1",
                "low_snr_accuracy",
                "per_snr_accuracy",
                "ece_15_bin",
                "nll",
                "brier_score",
            ],
            "low_snr_values_db": list(LOW_SNR_VALUES),
            "comparison": "lmscnet_s2_minus_se_msfn_1d",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_stratification": ["modulation", "snr_db"],
            "no_tuning": True,
            "no_design_changes_after_test": True,
            "one_shot": True,
        },
        "test_authorization": {
            "consumption_marker_path": str(marker),
            "output_dir": str(output),
            "marker_written_before_test_dataset_construction": True,
            "retry_after_failure_or_interruption": "forbidden",
        },
        "pretest_evidence": {
            "bound_reports_test_accessed_false": True,
            "dataset_adapter_rejected_test_before_freeze": True,
            "claim_scope": "artifact-and-code-evidence; not an omniscient host-history proof",
        },
    }


def write_manifest_atomic(manifest: dict[str, Any], output_path: Path) -> Path:
    output = output_path.resolve()
    if output.exists():
        raise ExperimentFreezeError("Refusing to overwrite an experiment freeze manifest")
    if not output.parent.is_dir():
        raise ExperimentFreezeError("Manifest parent directory must already exist")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False, suffix=".tmp"
    ) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return output


def _current_process_commands() -> list[str]:
    if os.name == "nt":
        script = (
            "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | "
            "ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise ExperimentFreezeError("Could not audit running processes")
        value = json.loads(result.stdout or "[]")
        rows = value if isinstance(value, list) else [value]
        return [str(row.get("CommandLine") or "") for row in rows if isinstance(row, dict)]
    result = subprocess.run(
        ["ps", "-eo", "args="], check=False, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise ExperimentFreezeError("Could not audit running processes")
    return result.stdout.splitlines()


def audit_freeze_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    require_unconsumed: bool,
    process_commands: Iterable[str] | None = None,
) -> dict[str, Any]:
    path = _regular_file(manifest_path, "experiment freeze manifest")
    manifest = load_json(path, "experiment freeze manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("purpose") != "experiment_freeze_manifest_v1"
    ):
        raise ExperimentFreezeError("Experiment freeze manifest identity is invalid")
    require_clean_commit(project_root, str(manifest.get("implementation_commit")))
    dataset = manifest.get("dataset")
    selection = manifest.get("selection")
    protocol = manifest.get("test_protocol")
    authorization = manifest.get("test_authorization")
    if not all(isinstance(item, dict) for item in (dataset, selection, protocol, authorization)):
        raise ExperimentFreezeError("Experiment freeze manifest sections are incomplete")
    if (
        dataset.get("assignment_sha256") != ASSIGNMENT_SHA256
        or dataset.get("preprocessing_mode") != PREPROCESSING_MODE
        or selection.get("final_model") != "lmscnet_s2"
        or selection.get("strongest_fair_baseline") != "se_msfn_1d"
        or selection.get("seeds") != list(SEEDS)
        or protocol.get("models") != list(TEST_MODELS)
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("bootstrap_seed") != BOOTSTRAP_SEED
        or protocol.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES
        or protocol.get("one_shot") is not True
    ):
        raise ExperimentFreezeError("Frozen dataset, selection, or test protocol differs")
    for field in ("hdf5", "conversion_manifest", "split_manifest", "leakage_audit"):
        binding = dataset.get(field)
        if not isinstance(binding, dict):
            raise ExperimentFreezeError(f"Dataset binding is missing: {field}")
        current = file_binding(Path(str(binding.get("path"))), field)
        if current != binding:
            raise ExperimentFreezeError(f"Dataset binding changed: {field}")
    runs = selection.get("runs")
    if not isinstance(runs, list) or len(runs) != len(TEST_MODELS) * len(SEEDS):
        raise ExperimentFreezeError("Selected run matrix is incomplete")
    identities = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ExperimentFreezeError("Selected run entry is invalid")
        identity = (run.get("model"), run.get("seed"))
        identities.add(identity)
        validated = _validate_run(
            model=str(run.get("model")),
            seed=int(run.get("seed")),
            config_path=Path(str(run.get("config", {}).get("path"))),
            metrics_path=Path(str(run.get("metrics", {}).get("path"))),
            checkpoint_path=Path(str(run.get("checkpoint", {}).get("path"))),
        )
        if validated != run:
            raise ExperimentFreezeError(f"Selected run binding changed: {identity}")
    if identities != {(model, seed) for model in TEST_MODELS for seed in SEEDS}:
        raise ExperimentFreezeError("Selected run identities differ from the frozen matrix")
    reports = manifest.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ExperimentFreezeError("Report bindings are missing")
    for report in reports:
        if not isinstance(report, dict):
            raise ExperimentFreezeError("Report binding is invalid")
        current = _validate_report(Path(str(report.get("path"))), str(report.get("name")))
        if current != report:
            raise ExperimentFreezeError(f"Report binding changed: {report.get('name')}")
    marker = Path(str(authorization.get("consumption_marker_path")))
    output = Path(str(authorization.get("output_dir")))
    if require_unconsumed and (marker.exists() or output.exists()):
        raise ExperimentFreezeError(
            "Test authorization was already consumed or output already exists"
        )
    commands = (
        list(process_commands) if process_commands is not None else _current_process_commands()
    )
    conflicts = sorted(
        command
        for command in commands
        if any(marker_text in command.lower() for marker_text in DISALLOWED_PROCESS_MARKERS)
        and "audit_experiment_freeze.py" not in command.lower()
    )
    if conflicts:
        raise ExperimentFreezeError(
            "Training, bootstrap, replication, or monitor process is still running"
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "implementation_commit": manifest["implementation_commit"],
        "selected_run_count": len(runs),
        "report_count": len(reports),
        "test_previously_accessed": False,
        "test_access_claim_scope": manifest["pretest_evidence"]["claim_scope"],
        "hashes_consistent": True,
        "conflicting_processes": [],
        "authorization_unconsumed": not marker.exists() and not output.exists(),
    }


def consume_test_authorization(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_json(manifest_path, "experiment freeze manifest")
    marker_path = Path(manifest["test_authorization"]["consumption_marker_path"])
    payload = {
        "schema_version": 1,
        "status": "consumed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_commit": manifest["implementation_commit"],
        "test_dataset_constructed": False,
        "retry_allowed": False,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError as error:
        raise ExperimentFreezeError("Test authorization has already been consumed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def update_consumption_marker(manifest_path: Path, **updates: Any) -> None:
    manifest = load_json(manifest_path, "experiment freeze manifest")
    marker_path = Path(manifest["test_authorization"]["consumption_marker_path"])
    marker = load_json(marker_path, "test consumption marker")
    marker.update(updates)
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker_path)


def authorize_frozen_test_dataset(manifest_path: Path) -> dict[str, str]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_json(manifest_path, "experiment freeze manifest")
    marker_path = Path(manifest["test_authorization"]["consumption_marker_path"])
    marker = load_json(marker_path, "test consumption marker")
    if (
        marker.get("manifest_sha256") != sha256_file(manifest_path)
        or marker.get("status") not in {"consumed", "running"}
        or marker.get("retry_allowed") is not False
    ):
        raise ExperimentFreezeError("Test consumption marker is invalid")
    return {
        "manifest_sha256": marker["manifest_sha256"],
        "assignment_sha256": manifest["dataset"]["assignment_sha256"],
        "preprocessing_mode": manifest["dataset"]["preprocessing_mode"],
    }
