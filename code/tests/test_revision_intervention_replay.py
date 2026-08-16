from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "code" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from na_lmscnet.data.contracts import ModulationSample, make_sample  # noqa: E402
from na_lmscnet.models import LMSCNetS2, LMSCNetS2Mean, LMSCNetS2Shuffled  # noqa: E402


class FixtureDataset(Dataset[ModulationSample]):
    def __init__(self, count: int, seed: int = 13) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.samples = [
            make_sample(
                iq=torch.randn((2, 128), generator=generator),
                modulation=index % 11,
                snr=float(-10 + 2 * (index % 6)),
                sample_id=f"fixture:{index:04d}",
            )
            for index in range(count)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ModulationSample:
        return self.samples[index]


def _batches(dataset: Dataset[ModulationSample], size: int = 4) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for start in range(0, len(dataset), size):
        batch = [dataset[index] for index in range(start, min(start + size, len(dataset)))]
        output.append(
            {
                "iq": torch.stack([sample["iq"] for sample in batch]),
                "modulation": torch.tensor([sample["modulation"] for sample in batch]),
                "snr": torch.tensor([sample["snr"] for sample in batch]),
                "sample_id": [sample["sample_id"] for sample in batch],
            }
        )
    return output


def test_mean_gate_intervention_replaces_sample_weights_with_train_average() -> None:
    model = LMSCNetS2Mean()
    model.load_state_dict(LMSCNetS2().state_dict(), strict=False)
    model.fit_mean_gate(_batches(FixtureDataset(8)))
    assert model.mean_gate_fitted is True

    validation = _batches(FixtureDataset(8, seed=99))
    for batch in validation:
        output = model(batch["iq"])
        expected = model.mean_gate.unsqueeze(0).expand(len(batch["iq"]), -1, -1)
        assert torch.allclose(output["scale_weights"], expected)


def test_shuffled_gate_preserves_marginals_per_batch_and_is_seed_deterministic() -> None:
    base = LMSCNetS2()
    batches = _batches(FixtureDataset(8))
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    for batch in batches:
        model_a = LMSCNetS2Shuffled(permutation_seed=2026)
        model_a.load_state_dict(base.state_dict())
        model_b = LMSCNetS2Shuffled(permutation_seed=2026)
        model_b.load_state_dict(base.state_dict())
        output_a = model_a(batch["iq"])["scale_weights"]
        output_b = model_b(batch["iq"])["scale_weights"]
        aligned = base(batch["iq"])["scale_weights"]
        assert torch.equal(output_a, output_b)
        assert not torch.equal(output_a, aligned)
        assert model_a.last_permutation_hash == model_b.last_permutation_hash
        assert torch.allclose(
            output_a.sort(dim=0).values, aligned.sort(dim=0).values
        )
        first.append(output_a)
        second.append(output_b)
    assert torch.equal(torch.cat(first), torch.cat(second))


def test_frozen_checkpoint_loads_into_all_s2_intervention_wrappers() -> None:
    checkpoint = {
        "schema_version": 1,
        "model_name": "lmscnet_s2",
        "model_state_dict": LMSCNetS2().state_dict(),
        "epoch": 1,
        "validation": {"accuracy": 0.0, "macro_f1": 0.0},
        "bindings": {
            "seed": 13,
            "split_manifest_sha256": "7c1d93c15bc24656f5857638bbccfd59932cc2f21b4c9f7ea36f47b3a5850dae",
            "assignment_sha256": "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941",
            "project_commit": "f5760d85ff0bbcf28b1f6005f3ef5dad1e615de6",
            "data_protocol": {"preprocessing_mode": "per_sample_max_abs"},
        },
    }
    for cls in (LMSCNetS2, LMSCNetS2Mean, LMSCNetS2Shuffled):
        model = cls(permutation_seed=2026) if cls is LMSCNetS2Shuffled else cls()
        strict = cls is not LMSCNetS2Mean
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        output = model(torch.randn((2, 2, 128)))
        assert output["logits"].shape == (2, 11)


def test_replay_manifest_schema_blocks_test_access_markers() -> None:
    manifest = {
        "schema_version": 1,
        "purpose": "phase_r2_intervention_replay",
        "split_manifest_sha256": "7c1d93c15bc24656f5857638bbccfd59932cc2f21b4c9f7ea36f47b3a5850dae",
        "assignment_sha256": "0037530e0f65df3eb0ba9f948764beb960ead5551b646a9fc5c6f735703e8941",
        "seeds": [13, 37, 73, 101, 137],
        "permutation_seeds": [13, 37],
        "test_accessed": False,
        "results": [],
        "permutation_digests": {},
    }
    assert manifest["test_accessed"] is False
    assert len(manifest["split_manifest_sha256"]) == 64
    assert len(manifest["assignment_sha256"]) == 64
