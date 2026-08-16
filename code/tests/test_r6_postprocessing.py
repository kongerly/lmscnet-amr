from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = PROJECT_ROOT / "code" / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r6_summary_aggregates_frozen_five_seeds() -> None:
    module = _load_script("summarize_r6_fixed_epoch_validation.py")
    rows = []
    for model_index, model in enumerate(module.MODELS):
        for seed_index, seed in enumerate(module.SEEDS):
            value = 0.5 + model_index * 0.01 + seed_index * 0.001
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "overall_accuracy": value,
                    "macro_f1": value - 0.01,
                    "low_snr_accuracy": value - 0.1,
                    "validation_loss": 1.0 - value,
                }
            )
    aggregate = module._aggregate(rows)
    assert [item["model"] for item in aggregate] == list(module.MODELS)
    assert aggregate[0]["overall_accuracy_mean"] == pytest.approx(0.502)


def test_r6_summary_rejects_missing_seed() -> None:
    module = _load_script("summarize_r6_fixed_epoch_validation.py")
    rows = []
    for model in module.MODELS:
        for seed in module.SEEDS:
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "overall_accuracy": 0.5,
                    "macro_f1": 0.5,
                    "low_snr_accuracy": 0.4,
                    "validation_loss": 1.0,
                }
            )
    rows.pop()
    with pytest.raises(module.SummaryError, match="frozen five seeds"):
        module._aggregate(rows)


def test_r6_queue_metrics_require_fixed_epoch() -> None:
    module = _load_script("audit_r6_fixed_epoch_queue.py")
    metrics = {
        "test_accessed": False,
        "purpose": "publication_candidate",
        "selection_metric": "validation_macro_f1",
        "selected_checkpoint_epoch": 100,
        "best_epoch": 100,
        "history": [{"epoch": epoch} for epoch in range(1, 101)],
        "model": {"name": "lmscnet_s2"},
        "bindings": {
            "project_commit": module.EXPECTED_COMMIT,
            "split_manifest_sha256": module.EXPECTED_SPLIT,
            "assignment_sha256": module.EXPECTED_ASSIGNMENT,
            "seed": 13,
        },
    }
    with pytest.raises(module.R6AuditError, match="selection metric"):
        module._audit_metrics(metrics, run_id="lmscnet_s2-seed-13", model="lmscnet_s2", seed=13)


def test_r6_freeze_collects_artifacts_and_rejects_test_access(tmp_path: Path) -> None:
    module = _load_script("generate_r6_validation_freeze.py")
    namespace = tmp_path / "namespace"
    evidence = namespace / "reports" / "example"
    evidence.mkdir(parents=True)
    (evidence / "report.json").write_text('{"test_accessed": false}\n', encoding="utf-8")
    rows = module._artifact_rows(namespace, ("reports/example",))
    assert rows[0]["path"] == "reports/example/report.json"

    (evidence / "report.json").write_text('{"test_accessed": true}\n', encoding="utf-8")
    with pytest.raises(module.FreezeError, match="reports test access"):
        module._artifact_rows(namespace, ("reports/example",))
