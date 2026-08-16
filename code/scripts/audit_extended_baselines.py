"""Audit extended baseline structure and frozen complexity controls without test access."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation import count_macs, count_parameters  # noqa: E402
from na_lmscnet.models import build_model  # noqa: E402
from na_lmscnet.training import load_experiment_config  # noqa: E402
from na_lmscnet.training.engine import experiment_config_sha256  # noqa: E402

MODELS = ("resnet1d_macs", "mobilenetv2_1d", "mcldnn", "se_msfn_1d")
S2_PARAMETERS = 124_861
S2_MACS = 4_654_792


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    if output == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output.parents:
        raise ValueError("Audit output must remain outside the repository")
    rows = []
    for name in MODELS:
        config_path = (
            PROJECT_ROOT / "code/configs/experiments" / f"{name}_radioml_2016_10a.yml"
        )
        config = load_experiment_config(config_path)
        if config.test_access != "forbidden" or int(config.training["seed"]) != 13:
            raise ValueError(f"{name} config changed test isolation or seed")
        model = build_model(name, num_classes=11, dropout=0.2)
        if tuple(inspect.signature(model.forward).parameters) != ("iq",):
            raise ValueError(f"{name} forward must accept only iq")
        outputs = model(torch.zeros((2, 2, 128), dtype=torch.float32))
        if outputs.shape != (2, 11) or not bool(torch.isfinite(outputs).all()):
            raise ValueError(f"{name} output interface differs")
        rows.append(
            {
                "model": name,
                "config_filename": config_path.name,
                "config_sha256": experiment_config_sha256(config_path),
                "parameters": count_parameters(model),
                "macs": count_macs(model, (1, 2, 128), torch.device("cpu")),
                "forward_inputs": ["iq"],
                "output_shape": [2, 11],
                "test_access": "forbidden",
            }
        )
    by_model = {str(row["model"]): row for row in rows}
    mac_gap = abs(int(by_model["resnet1d_macs"]["macs"]) - S2_MACS) / S2_MACS
    parameter_gap = (
        abs(int(by_model["mobilenetv2_1d"]["parameters"]) - S2_PARAMETERS) / S2_PARAMETERS
    )
    if mac_gap >= 0.05 or parameter_gap >= 0.05:
        raise ValueError("Extended complexity controls differ from the five-percent gate")
    report = {
        "schema_version": 1,
        "purpose": "extended_baseline_structure_and_complexity_audit",
        "test_accessed": False,
        "s2_reference": {"parameters": S2_PARAMETERS, "macs": S2_MACS},
        "controls": {
            "resnet1d_macs_gap_fraction": mac_gap,
            "mobilenetv2_1d_parameter_gap_fraction": parameter_gap,
        },
        "models": rows,
        "source_boundaries": {
            "resnet1d_macs": "project ResNet1D control with pre-training frozen width",
            "mobilenetv2_1d": "paper-derived 1D inverted-residual adaptation",
            "mcldnn": "MIT SigDA b68c856 model adaptation; no external data or checkpoint",
            "se_msfn_1d": "arXiv 2209.03764 source-informed 128-sample adaptation",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
