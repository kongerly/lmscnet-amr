"""Phase R2 efficiency measurement for the revision-controlled models.

Reports parameters, MACs, checkpoint size, and batch-1 GPU/CPU latency for
the R2 models (S1-static, S1-wide-static, S2-mean, S2-shuffled, SKNet-1D,
AFNet adaptation) plus the reference S2. Latency is a measurement, not a
deployment claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.evaluation import count_macs, count_parameters  # noqa: E402
from na_lmscnet.models import build_model  # noqa: E402

MODELS = (
    "lmscnet_s2",
    "lmscnet_s1_static",
    "lmscnet_s1_wide_static",
    "lmscnet_s2_mean",
    "lmscnet_s2_shuffled",
    "sknet_1d_adaptation",
    "afnet_adaptation",
)


class EfficiencyError(ValueError):
    """Raised when R2 efficiency measurement cannot be completed."""


def _measure_latency(model: torch.nn.Module, device: torch.device, warmup: int, iterations: int) -> float:
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


def _efficiency(model: torch.nn.Module, checkpoint_path: Path | None, device: torch.device, warmup: int, iterations: int) -> dict[str, object]:
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
        "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path is not None else None,
        "gpu_latency_ms": gpu_latency,
        "gpu_throughput_samples_per_s": 1000.0 / gpu_latency
        if math.isfinite(gpu_latency)
        else float("nan"),
        "cpu_latency_ms": cpu_latency,
        "cpu_threads": 1,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-queue-root", type=Path, required=True)
    parser.add_argument("--s2-queue-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    r2_root = args.r2_queue_root.resolve(strict=True)
    s2_root = args.s2_queue_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_dir.parents:
        raise EfficiencyError("Output must remain outside the repository")
    device = torch.device(args.device)
    s2_checkpoint = s2_root / "final-family-multiseed" / "lmscnet_s2-seed-13" / "best.pt"
    if not s2_checkpoint.is_file():
        raise EfficiencyError(f"Missing S2 checkpoint: {s2_checkpoint}")
    rows: list[dict[str, object]] = []
    for model in MODELS:
        if model == "lmscnet_s2_mean" or model == "lmscnet_s2_shuffled" or model == "lmscnet_s2":
            checkpoint_path = s2_checkpoint
        else:
            checkpoint_path = r2_root / f"{model}-seed-13" / "best.pt"
        if not checkpoint_path.is_file():
            raise EfficiencyError(f"Missing checkpoint for {model}: {checkpoint_path}")
        model_instance = build_model(
            model,
            num_classes=11,
            dropout=0.2,
            expansion=1.8 if model == "lmscnet_s1_wide_static" else 1.25,
            permutation_seed=13,
        )
        row = _efficiency(model_instance, checkpoint_path, device, args.warmup, args.iterations)
        rows.append({"model": model, **row})
    report = {
        "schema_version": 1,
        "purpose": "phase_r2_efficiency_measurement",
        "test_accessed": False,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "input_shape": [1, 2, 128],
        "rows": rows,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "r2-efficiency.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "model_count": len(rows), "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
