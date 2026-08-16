from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from na_lmscnet.training.multiseed import (
    SEEDS,
    MultiSeedError,
    multi_seed_run_specs,
    run_multi_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTED = PROJECT_ROOT / "code/configs/experiments"
BASE_CONFIGS = [
    SELECTED / "cnn2_radioml_2016_10a_selected.yml",
    SELECTED / "cldnn_radioml_2016_10a_selected.yml",
    SELECTED / "resnet1d_radioml_2016_10a_selected.yml",
]


def test_multi_seed_specs_cover_models_and_seeds() -> None:
    specs = multi_seed_run_specs(("cnn2", "cldnn", "resnet1d"))

    assert len(specs) == 15
    assert [(spec.model, spec.seed) for spec in specs] == [
        (model, seed) for model in ("cnn2", "cldnn", "resnet1d") for seed in SEEDS
    ]
    assert len({spec.run_id for spec in specs}) == 15
    assert specs[0].run_id == "cnn2-seed-13"
    assert specs[-1].run_id == "resnet1d-seed-137"


def test_derived_config_overrides_seed_and_experiment_id(tmp_path: Path) -> None:
    from na_lmscnet.training.engine import load_experiment_config
    from na_lmscnet.training.multiseed import _derived_config

    config = load_experiment_config(BASE_CONFIGS[0])
    spec = multi_seed_run_specs(("cnn2",))[1]
    derived = _derived_config(config, spec)

    assert derived["experiment_id"] == "cnn2_radioml_2016_10a_seed37"
    assert derived["training"]["seed"] == 37
    assert derived["model"]["name"] == "cnn2"
    assert derived["optimizer"]["learning_rate"] == 0.0003
    assert derived["model"]["dropout"] == 0.0


def test_multi_seed_specs_accept_explicit_three_seed_protocol() -> None:
    specs = multi_seed_run_specs(("lmscnet_s2",), (13, 37, 73))
    assert [spec.seed for spec in specs] == [13, 37, 73]


@pytest.mark.parametrize("seeds", [(), (13, 13), (-1,)])
def test_multi_seed_specs_reject_invalid_seed_sets(seeds: tuple[int, ...]) -> None:
    with pytest.raises(MultiSeedError, match="seeds"):
        multi_seed_run_specs(("cnn2",), seeds)


def test_derived_config_uses_base_dataset_id(tmp_path: Path) -> None:
    from na_lmscnet.training.engine import load_experiment_config
    from na_lmscnet.training.multiseed import _derived_config

    raw = yaml.safe_load(BASE_CONFIGS[0].read_text(encoding="utf-8"))
    raw["data"]["dataset_id"] = "radioml_2018_01a"
    path = tmp_path / "2018.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(path)

    derived = _derived_config(config, multi_seed_run_specs(("cnn2",), (13,))[0])
    assert derived["experiment_id"] == "cnn2_radioml_2018_01a_seed13"


def test_rejects_base_config_with_wrong_assignment(tmp_path: Path) -> None:
    from na_lmscnet.training.engine import load_experiment_config

    raw = yaml.safe_load(BASE_CONFIGS[0].read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["data"]["assignment_sha256"] = "9" * 64
    config_path = tmp_path / "tampered.yml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_experiment_config(config_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(MultiSeedError, match="assignment"):
        run_multi_seed(
            base_config_paths=[config_path],
            train_dataset=[],
            validation_dataset=[],
            output_dir=output,
            project_root=PROJECT_ROOT,
            project_commit="1" * 40,
            split_manifest_sha256="2" * 64,
            assignment_sha256="0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941",
            device=torch.device("cpu"),
        )


def test_multi_seed_runs_fifteen_resumes_and_rejects_binding_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "multi"
    output.mkdir()
    assignment = "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941"
    split_manifest = "2" * 64
    project_commit = "1" * 40
    calls: list[str] = []

    def fake_run_training(**kwargs: object) -> dict[str, object]:
        config = kwargs["config"]
        config_path = kwargs["config_path"]
        run_dir = kwargs["output_dir"]
        model = config.model["name"]
        seed = config.training["seed"]
        run_id = f"{model}-seed-{seed}"
        calls.append(run_id)
        checkpoint = run_dir / "best.pt"
        checkpoint.write_bytes(run_id.encode())
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        from na_lmscnet.training.engine import experiment_config_sha256

        config_sha = experiment_config_sha256(config_path)
        metrics = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "purpose": config.purpose,
            "test_accessed": False,
            "bindings": {
                "experiment_config_sha256": config_sha,
                "split_manifest_sha256": split_manifest,
                "assignment_sha256": assignment,
                "project_commit": project_commit,
                "seed": seed,
            },
            "epochs_completed": 1,
            "best_epoch": 1,
            "best_validation_macro_f1": 0.5,
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "train_samples": 154000,
                    "validation_loss": 0.5,
                    "validation": {"accuracy": 0.5, "macro_f1": 0.5},
                }
            ],
            "artifacts": {"checkpoint_sha256": checkpoint_sha},
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        return metrics

    monkeypatch.setattr("na_lmscnet.training.multiseed.run_training", fake_run_training)
    kwargs = {
        "base_config_paths": BASE_CONFIGS,
        "train_dataset": [],
        "validation_dataset": [],
        "output_dir": output,
        "project_root": PROJECT_ROOT,
        "project_commit": project_commit,
        "split_manifest_sha256": split_manifest,
        "assignment_sha256": assignment,
        "device": torch.device("cpu"),
    }

    summary = run_multi_seed(**kwargs)
    assert len(calls) == 15
    assert summary["run_count"] == 15
    assert summary["test_accessed"] is False
    assert (output / "multi-seed-summary.json").is_file()
    assert {run["model"] for run in summary["runs"]} == {"cnn2", "cldnn", "resnet1d"}
    assert {run["seed"] for run in summary["runs"]} == set(SEEDS)

    run_multi_seed(**kwargs)
    assert len(calls) == 15

    tampered = output / "cnn2-seed-13" / "metrics.json"
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["bindings"]["project_commit"] = "3" * 40
    tampered.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MultiSeedError, match="bindings"):
        run_multi_seed(**kwargs)
