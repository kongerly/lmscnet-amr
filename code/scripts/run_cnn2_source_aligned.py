"""Run the original-author VT-CNN2 protocol while preserving project test isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data import RadioML2016HDF5Dataset  # noqa: E402
from na_lmscnet.data.contracts import ModulationSample  # noqa: E402
from na_lmscnet.models import SourceVTCNN2  # noqa: E402
from na_lmscnet.training.metrics import classification_metrics  # noqa: E402
from na_lmscnet.training.progress import ProgressReporter  # noqa: E402

SOURCE_REPOSITORY = "radioML/examples"
SOURCE_COMMIT = "6c9ac6029ab1d0803442da7de8b7be04714bdebb"
SOURCE_NOTEBOOK = "modulation_recognition/RML2016.10a_VTCNN2_example.ipynb"
SOURCE_NOTEBOOK_BLOB = "b059e90db81612715deeaea38576f77969683227"
DEFAULT_DATASET_SPEC = PROJECT_ROOT / "code/configs/data/radioml_2016_10a.yml"
DEFAULT_CONVERSION_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_conversion.yml"
DEFAULT_SPLIT_CONTRACT = PROJECT_ROOT / "code/configs/data/radioml_2016_10a_split.yml"


class SourceSubset(Dataset[ModulationSample]):
    def __init__(
        self,
        train_dataset: RadioML2016HDF5Dataset,
        validation_dataset: RadioML2016HDF5Dataset,
        pool_rows: tuple[int, ...],
        positions: np.ndarray,
    ) -> None:
        self.datasets = (train_dataset, validation_dataset)
        lookup = {
            row: (dataset_index, local_index)
            for dataset_index, dataset in enumerate(self.datasets)
            for local_index, row in enumerate(dataset.rows)
        }
        self.references = tuple(lookup[pool_rows[int(position)]] for position in positions)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> ModulationSample:
        dataset_index, local_index = self.references[index]
        return self.datasets[dataset_index][local_index]

    def __getitems__(self, indices: list[int]) -> list[ModulationSample]:
        references = [self.references[index] for index in indices]
        grouped: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
        for output_index, (dataset_index, local_index) in enumerate(references):
            grouped[dataset_index].append((output_index, local_index))
        output: list[ModulationSample | None] = [None] * len(indices)
        for dataset_index, requests in grouped.items():
            if not requests:
                continue
            samples = self.datasets[dataset_index].__getitems__(
                [local_index for _, local_index in requests]
            )
            for (output_index, _), sample in zip(requests, samples, strict=True):
                output[output_index] = sample
        if any(sample is None for sample in output):
            raise RuntimeError("Source subset batch assembly is incomplete")
        return [sample for sample in output if sample is not None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _project_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--conversion-contract", type=Path, default=DEFAULT_CONVERSION_CONTRACT)
    parser.add_argument("--split-contract", type=Path, default=DEFAULT_SPLIT_CONTRACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, object]:
    model.eval()
    loss_sum = 0.0
    count = 0
    predictions = []
    targets = []
    snrs = []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        target = batch["modulation"].to(device, non_blocking=True)
        logits = model(iq)
        loss = nn.functional.cross_entropy(logits, target)
        loss_sum += float(loss) * len(target)
        count += len(target)
        predictions.append(logits.argmax(1).cpu())
        targets.append(target.cpu())
        snrs.append(batch["snr"].cpu())
    metrics = classification_metrics(
        torch.cat(predictions), torch.cat(targets), torch.cat(snrs), num_classes=11
    )
    return loss_sum / count, metrics


def main() -> int:
    args = parse_args()
    project_commit = _project_commit()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_root = args.output_dir.resolve()
    if output_root == PROJECT_ROOT.resolve() or PROJECT_ROOT.resolve() in output_root.parents:
        raise ValueError("Source-aligned artifacts must remain outside the repository")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "source-aligned"
    run_dir.mkdir(exist_ok=True)
    if (run_dir / "metrics.json").is_file():
        print(json.dumps({"status": "already_complete", "test_accessed": False}))
        return 0
    if any(run_dir.iterdir()):
        raise RuntimeError("Incomplete source-aligned output directory must be cleared before rerun")

    common = {
        "hdf5_path": args.hdf5,
        "conversion_manifest_path": args.conversion_manifest,
        "split_manifest_path": args.split_manifest,
        "leakage_audit_path": args.leakage_audit,
        "split_contract_path": args.split_contract,
        "dataset_spec_path": args.dataset_spec,
        "conversion_contract_path": args.conversion_contract,
        "preprocessing": "raw",
    }
    with (
        RadioML2016HDF5Dataset(split="train", **common) as project_train,
        RadioML2016HDF5Dataset(split="validation", **common) as project_validation,
    ):
        pool_rows = tuple(sorted((*project_train.rows, *project_validation.rows)))
        rng = np.random.RandomState(2016)
        train_positions = rng.choice(len(pool_rows), size=len(pool_rows) // 2, replace=False)
        evaluation_positions = np.asarray(
            sorted(set(range(len(pool_rows))) - set(int(value) for value in train_positions)),
            dtype=np.int64,
        )
        source_train = SourceSubset(project_train, project_validation, pool_rows, train_positions)
        source_validation = SourceSubset(
            project_train, project_validation, pool_rows, evaluation_positions
        )
        protocol = {
            "schema_version": 1,
            "purpose": "source_aligned_vtcnn2_reproduction",
            "test_accessed": False,
            "source": {
                "repository": SOURCE_REPOSITORY,
                "commit": SOURCE_COMMIT,
                "notebook": SOURCE_NOTEBOOK,
                "notebook_blob": SOURCE_NOTEBOOK_BLOB,
            },
            "aligned": {
                "dataset_version": "RML2016.10A",
                "input": "raw float32 I/Q [2,128]",
                "split": "numpy RandomState(2016), random 50/50 without replacement",
                "architecture": "zero-pad, Conv2d 256 (1x3), Conv2d 80 (2x3), Dense 256",
                "dropout": 0.5,
                "optimizer": "Adam(lr=0.001, betas=(0.9,0.999), eps=1e-8)",
                "batch_size": 1024,
                "max_epochs": 100,
                "early_stopping": "validation loss, patience 5",
                "checkpoint": "minimum validation loss",
                "augmentation": "none",
            },
            "controlled_adaptations": [
                "PyTorch port replaces Keras 1.0.4/Theano",
                "project frozen test rows are excluded; source 50/50 split is rebuilt inside train+validation pool",
                "torch seed 2016 resolves source notebook backend-seed ambiguity",
            ],
            "bindings": {
                "project_commit": project_commit,
                "assignment_sha256": project_train.assignment_sha256,
                "split_manifest_sha256": _sha256_file(args.split_manifest),
                "allowed_pool_rows_sha256": _rows_sha256(np.asarray(pool_rows)),
                "source_train_positions_sha256": _rows_sha256(train_positions),
                "source_validation_positions_sha256": _rows_sha256(evaluation_positions),
                "hdf5_sha256": _sha256_file(args.hdf5),
            },
        }
        protocol_path = output_root / "source-aligned-protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bindings = {**protocol["bindings"], "source_protocol_sha256": _sha256_file(protocol_path)}

        random.seed(2016)
        np.random.seed(2016)
        torch.manual_seed(2016)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2016)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model = SourceVTCNN2(dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        loader_generator = torch.Generator().manual_seed(2016)
        train_loader = DataLoader(
            source_train, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True,
            persistent_workers=True, generator=loader_generator,
        )
        validation_loader = DataLoader(
            source_validation, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True,
            persistent_workers=True,
        )
        reporter = ProgressReporter()
        history = []
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        checkpoint_path = run_dir / "best.pt"
        for epoch in range(1, 101):
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for batch_index, batch in enumerate(train_loader, start=1):
                iq = batch["iq"].to(device, non_blocking=True)
                target = batch["modulation"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(iq), target)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(target)
                train_count += len(target)
                reporter.on_batch({
                    "event": "batch_complete", "epoch": epoch, "batch": batch_index,
                    "total_batches": len(train_loader), "max_epochs": 100,
                    "train_loss": train_loss_sum / train_count,
                })
            validation_loss, metrics = _validate(model, validation_loader, device)
            record = {
                "epoch": epoch,
                "learning_rate": 0.001,
                "train_loss": train_loss_sum / train_count,
                "train_samples": train_count,
                "validation_loss": validation_loss,
                "validation": asdict(metrics),
            }
            history.append(record)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
                torch.save({
                    "schema_version": 1,
                    "model_name": "source_vtcnn2",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation": asdict(metrics),
                    "bindings": bindings,
                }, checkpoint_path)
            else:
                stale += 1
            reporter.on_epoch({**record, "max_epochs": 100}, run_id="source-aligned")
            if stale >= 5:
                break
        reporter.finish()
        result = {
            "schema_version": 1,
            "experiment_id": "source_aligned_vtcnn2_rml2016_10a_v1",
            "purpose": "source_aligned_reproduction",
            "test_accessed": False,
            "bindings": bindings,
            "environment": {
                "python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "deterministic_algorithms": True,
            },
            "model": {"name": "source_vtcnn2", "dropout": 0.5,
                      "parameter_count": sum(value.numel() for value in model.parameters())},
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
            "artifacts": {"checkpoint_filename": "best.pt", "checkpoint_sha256": _sha256_file(checkpoint_path)},
        }
        _atomic_json(result, run_dir / "metrics.json")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_loss": best_loss, "test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
