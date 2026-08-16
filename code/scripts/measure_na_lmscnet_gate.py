"""Measure NA-LMSCNet parameters and MACs before complete training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation import count_macs, count_parameters  # noqa: E402
from na_lmscnet.models import NALMSCNet  # noqa: E402

PARAMETER_LIMIT = 500_000
MAC_LIMIT = 5_470_976


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure NA-LMSCNet efficiency gates.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = NALMSCNet(num_classes=11, dropout=0.2).to(device)
    parameter_count = count_parameters(model)
    macs = count_macs(model, (1, 2, 128), device)
    result = {
        "model": "na_lmscnet",
        "input_shape": [1, 2, 128],
        "parameter_count": parameter_count,
        "macs": macs,
        "parameter_limit": PARAMETER_LIMIT,
        "mac_limit": MAC_LIMIT,
        "parameter_gate_passed": parameter_count <= PARAMETER_LIMIT,
        "mac_gate_passed": macs <= MAC_LIMIT,
        "device": str(device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["parameter_gate_passed"] or not result["mac_gate_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
