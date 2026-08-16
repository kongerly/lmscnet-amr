"""Audit the frozen final S0/S1/S2 family without accessing the test split."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.evaluation import count_macs, count_parameters  # noqa: E402
from na_lmscnet.models import build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.engine import experiment_config_sha256  # noqa: E402

CONFIG_NAMES = (
    "lmscnet_s0_k3_radioml_2016_10a_selected.yml",
    "lmscnet_s0_k7_radioml_2016_10a_selected.yml",
    "lmscnet_s0_k15_radioml_2016_10a_selected.yml",
    "lmscnet_s0_wide_radioml_2016_10a_selected.yml",
    "lmscnet_s1_radioml_2016_10a_selected.yml",
    "lmscnet_s2_radioml_2016_10a_selected.yml",
)
FORBIDDEN_TOKENS = ("snr", "noise_embedding", "constant_embedding")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml",
    )
    parser.add_argument(
        "--conversion-contract",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml",
    )
    parser.add_argument(
        "--split-contract",
        type=Path,
        default=PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build(config_path: Path) -> tuple[object, torch.nn.Module]:
    config = load_experiment_config(config_path)
    model = build_model(
        str(config.model["name"]),
        num_classes=int(config.model["num_classes"]),
        dropout=float(config.model["dropout"]),
        expansion=float(config.model["expansion"]),
        kernel=int(config.model["kernel"]) if "kernel" in config.model else None,
    )
    return config, model


def _non_gate_schema(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(value.shape) for key, value in model.state_dict().items() if ".gate." not in key
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output.parents:
        raise ValueError("Audit output must remain outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")

    config_root = PROJECT_ROOT / "code/configs/experiments"
    rows: list[dict[str, object]] = []
    models: dict[str, torch.nn.Module] = {}
    shared_protocols: list[dict[str, object]] = []
    for name in CONFIG_NAMES:
        path = config_root / name
        config, model = _build(path)
        model_name = str(config.model["name"])
        models[model_name] = model
        if config.test_access != "forbidden":
            raise ValueError(f"{model_name} does not forbid test access")
        signature = inspect.signature(model.forward)
        if tuple(signature.parameters) != ("iq",):
            raise ValueError(f"{model_name} forward must accept only iq")
        named_paths = tuple(name.lower() for name, _ in model.named_modules()) + tuple(
            name.lower() for name, _ in model.named_parameters()
        )
        if any(token in path_name for token in FORBIDDEN_TOKENS for path_name in named_paths):
            raise ValueError(f"{model_name} contains a forbidden SNR/embedding path")
        outputs = model(torch.zeros((2, 2, 128), dtype=torch.float32))
        if set(outputs) != {"logits", "scale_weights"} or outputs["logits"].shape != (2, 11):
            raise ValueError(f"{model_name} output interface differs")
        rows.append(
            {
                "model": model_name,
                "config_filename": path.name,
                "config_sha256": experiment_config_sha256(path),
                "parameters": count_parameters(model),
                "macs": count_macs(model, (1, 2, 128), torch.device("cpu")),
                "kernels": list(model.kernels),
                "expansion": model.expansion,
                "content_adaptive": model.content_adaptive,
                "scale_weights_shape": list(outputs["scale_weights"].shape),
                "forward_inputs": list(signature.parameters),
                "output_keys": sorted(outputs),
                "forbidden_path_tokens_absent": True,
            }
        )
        shared_protocol = asdict(config)
        shared_protocol.pop("experiment_id")
        shared_protocol.pop("model")
        shared_protocols.append(shared_protocol)
    if any(protocol != shared_protocols[0] for protocol in shared_protocols[1:]):
        raise ValueError("S0/S1/S2 configs differ outside the model identity")

    s1 = models["lmscnet_s1"]
    s2 = models["lmscnet_s2"]
    if _non_gate_schema(s1) != _non_gate_schema(s2):
        raise ValueError("S1 and S2 differ outside the S2 content gate")
    s2_gate_parameters = sorted(name for name, _ in s2.named_parameters() if ".gate." in name)
    if len(s2_gate_parameters) != 24:
        raise ValueError("S2 must contain exactly four gate tensors per residual block")
    s2_macs = next(int(row["macs"]) for row in rows if row["model"] == "lmscnet_s2")
    wide_macs = next(int(row["macs"]) for row in rows if row["model"] == "lmscnet_s0_wide")
    mac_gap_fraction = abs(wide_macs - s2_macs) / s2_macs
    if mac_gap_fraction > 0.05:
        raise ValueError("Widened S0 differs from S2 by more than five percent MACs")

    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
        "preprocessing": "per_sample_max_abs",
    }
    with (
        RadioML2016HDF5Dataset(split="train", **common) as train_dataset,
        RadioML2016HDF5Dataset(split="validation", **common) as validation_dataset,
    ):
        if train_dataset.preprocessing != "per_sample_max_abs":
            raise ValueError("Train preprocessing is not frozen to per_sample_max_abs")
        if validation_dataset.preprocessing != "per_sample_max_abs":
            raise ValueError("Validation preprocessing is not frozen to per_sample_max_abs")
        if train_dataset.assignment_sha256 != validation_dataset.assignment_sha256:
            raise ValueError("Train/validation assignment bindings differ")
        sample = train_dataset[0]
        maximum = sample["iq"].square().sum(dim=0).sqrt().max()
        if not torch.isclose(maximum, torch.tensor(1.0), atol=1e-6, rtol=1e-6):
            raise ValueError("Bound train input does not satisfy max-abs normalization")
        data_binding = {
            "preprocessing_mode": train_dataset.preprocessing,
            "assignment_sha256": train_dataset.assignment_sha256,
            "split_manifest_sha256": _sha256_file(args.split_manifest),
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "sample_max_complex_amplitude": float(maximum),
            "constructed_splits": ["train", "validation"],
        }

    report = {
        "schema_version": 1,
        "purpose": "final_s0_s1_s2_structure_and_input_binding_audit",
        "test_accessed": False,
        "models": rows,
        "comparability": {
            "shared_non_model_protocol": True,
            "s1_s2_non_gate_state_schema_equal": True,
            "s2_gate_parameter_names": s2_gate_parameters,
            "widened_s0_to_s2_macs_gap_fraction": mac_gap_fraction,
        },
        "data_binding": data_binding,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
