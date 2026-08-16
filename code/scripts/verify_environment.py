from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib.metadata import version

MODULES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "h5py": "h5py",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "pytest": "pytest",
    "ruff": "ruff",
    "pip-audit": "pip_audit",
    "torch": "torch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the project environment.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when a CUDA device is unavailable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] != (3, 11):
        print(f"Expected Python 3.11, got {sys.version.split()[0]}", file=sys.stderr)
        return 1

    versions: dict[str, str] = {}
    for distribution, module in MODULES.items():
        importlib.import_module(module)
        versions[distribution] = version(distribution)

    import torch

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        print("CUDA is required but no CUDA device is available.", file=sys.stderr)
        return 1

    device = torch.device("cuda" if cuda_available else "cpu")
    tensor = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    checksum = float((tensor @ tensor.T).sum().item())
    report = {
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "tensor_checksum": checksum,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
