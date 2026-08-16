from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from na_lmscnet.training.sweep import (
    SweepError,
    load_sweep_contract,
    run_validation_sweep,
    select_best_run,
    sweep_run_specs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEPS = (
    ("cnn2", PROJECT_ROOT / "code/configs/experiments/cnn2_radioml_2016_10a_sweep.yml"),
    ("cldnn", PROJECT_ROOT / "code/configs/experiments/cldnn_radioml_2016_10a_sweep.yml"),
    (
        "na_lmscnet",
        PROJECT_ROOT / "code/configs/experiments/na_lmscnet_radioml_2016_10a_sweep.yml",
    ),
    ("resnet1d", PROJECT_ROOT / "code/configs/experiments/resnet1d_radioml_2016_10a_sweep.yml"),
)
EXPECTED_PREFIXES = {
    "cnn2": "cnn2_radioml_2016_10a_",
    "cldnn": "cldnn_radioml_2016_10a_",
    "na_lmscnet": "na_lmscnet_radioml_2016_10a_",
    "resnet1d": "resnet1d_radioml_2016_10a_",
}


@pytest.mark.parametrize(("model", "contract_path"), SWEEPS)
def test_repository_sweep_contract_expands_exact_grid(model: str, contract_path: Path) -> None:
    contract = load_sweep_contract(contract_path)
    specs = sweep_run_specs(contract)

    assert contract["sweep_id"] == f"{model}_radioml_2016_10a_validation_sweep_v1"
    assert contract["base_config"] == f"{model}_radioml_2016_10a.yml"
    assert [(spec.learning_rate, spec.dropout) for spec in specs] == [
        (0.001, 0.0),
        (0.001, 0.2),
        (0.0003, 0.0),
        (0.0003, 0.2),
    ]
    assert len({spec.run_id for spec in specs}) == 4
    assert all("seed-13" in spec.run_id for spec in specs)


@pytest.mark.parametrize(("model", "contract_path"), SWEEPS)
@pytest.mark.parametrize("mutation", ["test_access", "grid", "parallel", "seed", "sweep_id"])
def test_rejects_sweep_contract_mutations(
    model: str, contract_path: Path, tmp_path: Path, mutation: str
) -> None:
    contract = deepcopy(load_sweep_contract(contract_path))
    if mutation == "test_access":
        contract["test_access"] = "allowed"
    elif mutation == "grid":
        contract["grid"]["dropout"] = [0.2]
    elif mutation == "parallel":
        contract["execution"]["mode"] = "parallel"
    elif mutation == "seed":
        contract["seed"] = 37
    elif mutation == "sweep_id":
        contract["sweep_id"] = "other_radioml_2016_10a_validation_sweep_v1"
    path = tmp_path / "sweep.yml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    with pytest.raises(SweepError):
        load_sweep_contract(path)


def _run(run_id: str, f1: float, loss: float, *, test_accessed: bool = False) -> dict[str, object]:
    return {
        "run_id": run_id,
        "best_validation_macro_f1": f1,
        "best_validation_loss": loss,
        "test_accessed": test_accessed,
    }


def test_selects_best_run_with_documented_tie_breaks() -> None:
    runs = [
        _run("d", 0.8, 0.7),
        _run("c", 0.8, 0.6),
        _run("b", 0.8, 0.6),
        _run("a", 0.7, 0.5),
    ]

    assert select_best_run(runs)["run_id"] == "b"


def test_selection_rejects_incomplete_or_test_accessed_runs() -> None:
    with pytest.raises(SweepError, match="four"):
        select_best_run([_run("a", 0.8, 0.6)])
    runs = [_run(str(index), 0.8, 0.6) for index in range(4)]
    runs[-1]["test_accessed"] = True
    with pytest.raises(SweepError, match="test"):
        select_best_run(runs)


@pytest.mark.parametrize(("model", "contract_path"), SWEEPS)
def test_validation_sweep_runs_four_resumes_and_rejects_binding_tamper(
    model: str, contract_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "sweep"
    output.mkdir()
    assignment = "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941"
    split_manifest = "2" * 64
    project_commit = "1" * 40
    prefix = EXPECTED_PREFIXES[model]
    calls: list[str] = []

    def fake_run_training(**kwargs: object) -> dict[str, object]:
        config = kwargs["config"]
        config_path = kwargs["config_path"]
        run_dir = kwargs["output_dir"]
        run_id = config.experiment_id.removeprefix(prefix)
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
                "seed": 13,
            },
            "epochs_completed": 1,
            "best_epoch": 1,
            "best_validation_macro_f1": (0.6 if run_id == "lr-0p001_dropout-0_seed-13" else 0.4),
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "train_samples": 154000,
                    "validation_loss": 0.25 if "lr-0p001" in run_id else 0.5,
                    "validation": {"accuracy": 0.6, "macro_f1": 0.6},
                }
            ],
            "artifacts": {"checkpoint_sha256": checkpoint_sha},
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        return metrics

    monkeypatch.setattr("na_lmscnet.training.sweep.run_training", fake_run_training)
    kwargs = {
        "sweep_contract_path": contract_path,
        "train_dataset": [],
        "validation_dataset": [],
        "output_dir": output,
        "project_root": PROJECT_ROOT,
        "project_commit": project_commit,
        "split_manifest_sha256": split_manifest,
        "assignment_sha256": assignment,
        "device": __import__("torch").device("cpu"),
    }

    summary = run_validation_sweep(**kwargs)
    assert len(calls) == 4
    assert summary["sweep_id"] == f"{model}_radioml_2016_10a_validation_sweep_v1"
    assert summary["selected_run_id"] == "lr-0p001_dropout-0_seed-13"
    assert summary["test_accessed"] is False
    assert (output / "sweep-summary.json").is_file()

    run_validation_sweep(**kwargs)
    assert len(calls) == 4

    tampered = output / "lr-0p001_dropout-0_seed-13" / "metrics.json"
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["bindings"]["project_commit"] = "3" * 40
    tampered.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SweepError, match="bindings"):
        run_validation_sweep(**kwargs)
