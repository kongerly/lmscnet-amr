from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from na_lmscnet.training import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "code/configs/experiments"
CONFIG_NAMES = (
    "lmscnet_s0_k15_radioml_2018_01a_selected.yml",
    "lmscnet_s1_radioml_2018_01a_selected.yml",
    "lmscnet_s2_radioml_2018_01a_selected.yml",
    "se_msfn_1d_radioml_2018_01a_selected.yml",
)
ASSIGNMENT = "db3854fb698cd0b66a5ae67f1286535b06d890264ee3141895adafea0371fc01"


def _load_script(name: str) -> object:
    path = PROJECT_ROOT / "code/scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replication_configs_freeze_only_required_dataset_changes() -> None:
    configs = [load_experiment_config(CONFIG_ROOT / name) for name in CONFIG_NAMES]

    assert [config.model["name"] for config in configs] == [
        "lmscnet_s0_k15",
        "lmscnet_s1",
        "lmscnet_s2",
        "se_msfn_1d",
    ]
    for config in configs:
        assert config.data["dataset_id"] == "radioml_2018_01a"
        assert config.data["assignment_sha256"] == ASSIGNMENT
        assert config.data["batch_size"] == 256
        assert config.model["num_classes"] == 24
        assert config.model["dropout"] == 0.2
        assert config.optimizer["learning_rate"] == 0.001
        assert config.training["max_epochs"] == 100
        assert config.training["early_stopping_patience"] == 12
        assert config.test_access == "forbidden"


def test_s0_s1_s2_replication_protocols_match_except_model() -> None:
    configs = [load_experiment_config(CONFIG_ROOT / name) for name in CONFIG_NAMES[:3]]
    protocols = []
    for config in configs:
        protocol = asdict(config)
        protocol.pop("experiment_id")
        protocol.pop("model")
        protocols.append(protocol)
    assert protocols[0] == protocols[1] == protocols[2]


def test_replication_queue_constants_match_preregistration() -> None:
    module = _load_script("run_radioml_2018_replication.py")
    assert module.MODELS == ("lmscnet_s0_k15", "lmscnet_s1", "lmscnet_s2", "se_msfn_1d")
    assert module.SEEDS == (13, 37, 73)
    assert tuple(module.CONFIG_NAMES) == CONFIG_NAMES


def test_replication_model_shards_are_frozen_subsets() -> None:
    module = _load_script("run_radioml_2018_replication.py")
    assert module._validate_models(None) == module.MODELS
    assert module._validate_models(["lmscnet_s2"]) == ("lmscnet_s2",)
    assert module._validate_models(["se_msfn_1d", "lmscnet_s1"]) == (
        "lmscnet_s1",
        "se_msfn_1d",
    )
    with pytest.raises(ValueError, match="non-empty"):
        module._validate_models([])
    with pytest.raises(ValueError, match="unique"):
        module._validate_models(["lmscnet_s1", "lmscnet_s1"])
    with pytest.raises(ValueError, match="frozen subset"):
        module._validate_models(["eca"])


def test_replication_parser_accepts_single_model_shard(tmp_path: Path) -> None:
    module = _load_script("run_radioml_2018_replication.py")
    args = module.parse_args(
        [
            "--hdf5", str(tmp_path / "source.h5"),
            "--source-manifest", str(tmp_path / "source.json"),
            "--split-artifact", str(tmp_path / "split.h5"),
            "--split-manifest", str(tmp_path / "split.json"),
            "--output-root", str(tmp_path / "output"),
            "--models", "lmscnet_s2",
        ]
    )
    assert args.models == ["lmscnet_s2"]


def test_merged_shard_audit_requires_exact_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("audit_radioml_2018_shards.py")
    provenance = {
        "source_manifest_sha256": "source",
        "split_manifest_sha256": "split-manifest",
        "split_artifact_sha256": "split-artifact",
        "assignment_sha256": ASSIGNMENT,
        "project_commit": "commit",
        "preprocessing_mode": "per_sample_max_abs",
        "input_shape": [2, 1024],
    }

    def valid_report(_root: Path, _log: Path, expected_models: tuple[str, ...]) -> dict:
        model = expected_models[0]
        return {
            "bindings": {},
            "queue_provenance": provenance,
            "queue_protocol_sha256": f"protocol-{model}",
            "queue_summary_sha256": f"summary-{model}",
            "run_count": 3,
            "runs": [{"model": model, "seed": seed} for seed in module.SEEDS],
        }

    monkeypatch.setattr(module, "audit_queue", valid_report)
    report = module.audit_shards(tmp_path, tmp_path)
    assert report["run_count"] == 12
    assert report["bindings"] == provenance

    def duplicate_report(root: Path, log: Path, expected_models: tuple[str, ...]) -> dict:
        report = valid_report(root, log, expected_models)
        if expected_models == ("lmscnet_s1",):
            report["runs"][0] = {"model": "lmscnet_s0_k15", "seed": 13}
        return report

    monkeypatch.setattr(module, "audit_queue", duplicate_report)
    with pytest.raises(module.AuditError, match="Duplicate"):
        module.audit_shards(tmp_path, tmp_path)

    def missing_report(root: Path, log: Path, expected_models: tuple[str, ...]) -> dict:
        report = valid_report(root, log, expected_models)
        if expected_models == ("lmscnet_s2",):
            report["runs"].pop()
        return report

    monkeypatch.setattr(module, "audit_queue", missing_report)
    with pytest.raises(module.AuditError, match="incomplete"):
        module.audit_shards(tmp_path, tmp_path)


def test_queue_protocol_writer_refuses_different_existing_protocol(tmp_path: Path) -> None:
    module = _load_script("run_radioml_2018_replication.py")
    path = tmp_path / "protocol.json"
    module._write_or_validate_json(path, {"version": 1})
    module._write_or_validate_json(path, {"version": 1})
    with pytest.raises(ValueError, match="differs"):
        module._write_or_validate_json(path, {"version": 2})
