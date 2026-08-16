"""Run frozen validation sweeps and five-seed training for four extended baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("resnet1d_macs", "mobilenetv2_1d", "mcldnn", "se_msfn_1d")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _project_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Extended validation queue requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(script: str, arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "code/scripts" / script), *arguments],
        cwd=PROJECT_ROOT,
        check=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commit = _project_commit()
    output_root = args.output_root.resolve()
    if output_root == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_root.parents:
        raise ValueError("Validation artifacts must remain outside the repository")
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "purpose": "extended_baseline_validation_queue",
        "project_commit": commit,
        "preprocessing_mode": "per_sample_max_abs",
        "split_manifest_sha256": _sha256_file(args.split_manifest),
        "assignment_sha256": "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941",
        "baselines": list(BASELINES),
        "sweep_grid": {"learning_rate": [0.001, 0.0003], "dropout": [0.0, 0.2]},
        "seeds": [13, 37, 73, 101, 137],
        "execution_order": ["baseline_seed13_sweeps", "selected_baseline_five_seed"],
        "test_accessed": False,
    }
    protocol_path = output_root / "queue-protocol.json"
    if protocol_path.exists():
        if _load_json(protocol_path) != protocol:
            raise ValueError("Existing extended queue protocol differs")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    common = [
        "--hdf5",
        str(args.hdf5.resolve(strict=True)),
        "--conversion-manifest",
        str(args.conversion_manifest.resolve(strict=True)),
        "--split-manifest",
        str(args.split_manifest.resolve(strict=True)),
        "--leakage-audit",
        str(args.leakage_audit.resolve(strict=True)),
        "--device",
        args.device,
    ]
    selected_configs = []
    sweep_root = output_root / "sweeps"
    sweep_root.mkdir(exist_ok=True)
    for baseline in BASELINES:
        sweep_output = sweep_root / baseline
        sweep_output.mkdir(exist_ok=True)
        _run(
            "run_validation_sweep.py",
            [
                "--sweep",
                str(
                    PROJECT_ROOT
                    / "code/configs/experiments"
                    / f"{baseline}_radioml_2016_10a_sweep.yml"
                ),
                *common,
                "--output-dir",
                str(sweep_output),
            ],
        )
        summary = _load_json(sweep_output / "sweep-summary.json")
        if summary.get("test_accessed") is not False:
            raise ValueError(f"{baseline} sweep did not preserve test isolation")
        selected = summary.get("selected_run_id")
        if not isinstance(selected, str):
            raise ValueError(f"{baseline} sweep lacks a selected run")
        selected_configs.append(sweep_output / "configs" / f"{selected}.yml")
    multiseed_root = output_root / "multiseed"
    multiseed_root.mkdir(exist_ok=True)
    _run(
        "run_multi_seed.py",
        [
            "--configs",
            *[str(path) for path in selected_configs],
            *common,
            "--output-dir",
            str(multiseed_root),
        ],
    )
    summary = {
        **protocol,
        "selected_configs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in selected_configs
        ],
        "multiseed_summary_sha256": _sha256_file(multiseed_root / "multi-seed-summary.json"),
        "status": "complete",
    }
    (output_root / "queue-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "test_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
