from __future__ import annotations

from pathlib import Path

import pytest
import torch

from na_lmscnet.evaluation import count_parameters
from na_lmscnet.models import (
    AFNetAdaptation,
    LMSCNetS1Static,
    LMSCNetS1WideStatic,
    LMSCNetS2,
    LMSCNetS2Mean,
    LMSCNetS2Shuffled,
    SKNet1DAdaptation,
    shuffled_gate_weights,
)
from na_lmscnet.training import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _iq(batch: int = 4) -> torch.Tensor:
    return torch.randn((batch, 2, 128), generator=torch.Generator().manual_seed(13))


def test_s1_static_weights_are_global_and_normalized() -> None:
    model = LMSCNetS1Static().eval()
    first = model(_iq())['scale_weights']
    second = model(_iq())['scale_weights']
    assert first.shape == (4, 6, 3)
    assert torch.allclose(first.sum(dim=-1), torch.ones((4, 6)))
    assert torch.equal(first[0], first[1])
    assert torch.equal(first, second)
    assert len([name for name, _ in model.named_parameters() if '.static_logits' in name]) == 6


def test_wide_static_is_parameter_matched_without_validation_inputs() -> None:
    static = LMSCNetS1WideStatic()
    adaptive = LMSCNetS2()
    assert abs(count_parameters(static) - count_parameters(adaptive)) / count_parameters(adaptive) <= 0.05
    assert static.PRE_REGISTERED_EXPANSION == 1.8


def test_mean_gate_uses_only_explicit_train_batches() -> None:
    model = LMSCNetS2Mean().eval()
    train_batches = [(_iq(2), torch.zeros(2, dtype=torch.long)), (_iq(1), torch.ones(1, dtype=torch.long))]
    model.fit_mean_gate(train_batches)
    assert model.mean_gate_fitted is True
    output = model(_iq(3))
    assert torch.allclose(output['scale_weights'], model.mean_gate.unsqueeze(0).expand(3, -1, -1))
    with pytest.raises(ValueError, match='at least one train batch'):
        LMSCNetS2Mean().fit_mean_gate([])


def test_shuffled_gate_preserves_marginals_and_hashes_permutation() -> None:
    aligned = LMSCNetS2().eval()(_iq())['scale_weights']
    shuffled_a, hash_a = shuffled_gate_weights(aligned, 2026)
    shuffled_b, hash_b = shuffled_gate_weights(aligned, 2026)
    assert torch.equal(shuffled_a, shuffled_b)
    assert hash_a == hash_b and len(hash_a) == 64
    assert torch.allclose(shuffled_a.sort(dim=0).values, aligned.sort(dim=0).values)


def test_shuffled_model_records_fixed_hash() -> None:
    model = LMSCNetS2Shuffled(permutation_seed=2026).eval()
    output = model(_iq())
    assert output['scale_weights'].shape == (4, 6, 3)
    assert len(model.last_permutation_hash) == 64


@pytest.mark.parametrize('model_cls', [SKNet1DAdaptation, AFNetAdaptation])
def test_neighbor_adaptations_keep_the_dictionary_classifier_contract(model_cls: type[torch.nn.Module]) -> None:
    output = model_cls().eval()(_iq())
    assert set(output) == {'logits', 'scale_weights'}
    assert output['logits'].shape == (4, 11)


def test_sknet_weights_are_channel_normalized_and_input_dependent() -> None:
    model = SKNet1DAdaptation().eval()
    first = model(_iq())['scale_weights']
    different = model(torch.randn((4, 2, 128), generator=torch.Generator().manual_seed(7)))['scale_weights']
    assert first.shape == (4, 6, 3)
    assert torch.allclose(first.sum(dim=-1), torch.ones((4, 6)))
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first, different)
    block = model.stages[0][0]
    output, _ = block(model.stem(_iq()))
    assert output.shape == (4, 32, 128)


def test_afnet_lambda_scales_match_source_specification() -> None:
    model = AFNetAdaptation().eval()
    first = model(_iq())['scale_weights']
    assert first.shape == (4, 6, 2)
    assert torch.allclose(first.sum(dim=-1), torch.ones((4, 6)))
    block = model.stages[0][0]
    assert block.fusion1_scale == 1.0
    assert block.fusion2_scale == 2.0


@pytest.mark.parametrize(
    'filename',
    [
        'revision_r1_s1_static_radioml_2016_10a_smoke.yml',
        'revision_r1_s1_wide_static_radioml_2016_10a_smoke.yml',
        'revision_r1_s2_mean_radioml_2016_10a_smoke.yml',
        'revision_r1_s2_shuffled_radioml_2016_10a_smoke.yml',
        'revision_r1_sknet_1d_adaptation_radioml_2016_10a_smoke.yml',
        'revision_r1_afnet_adaptation_radioml_2016_10a_smoke.yml',
    ],
)
def test_revision_smoke_configs_are_test_isolated(filename: str) -> None:
    config = load_experiment_config(PROJECT_ROOT / 'code/configs/experiments' / filename)
    assert config.purpose == 'infrastructure_smoke_only'
    assert config.test_access == 'forbidden'
    assert config.data['dataset_id'] == 'radioml_2016_10a'


@pytest.mark.parametrize(
    'filename',
    [
        'revision_r6_s2_fixed_epoch_radioml_2016_10a.yml',
        'revision_r6_s1_static_fixed_epoch_radioml_2016_10a.yml',
        'revision_r6_s1_wide_static_fixed_epoch_radioml_2016_10a.yml',
        'revision_r6_sknet_1d_fixed_epoch_radioml_2016_10a.yml',
        'revision_r6_afnet_fixed_epoch_radioml_2016_10a.yml',
    ],
)
def test_r6_configs_use_fixed_epoch_without_test_access(filename: str) -> None:
    config = load_experiment_config(PROJECT_ROOT / 'code/configs/experiments' / filename)
    assert config.purpose == 'publication_candidate'
    assert config.selection_metric == 'fixed_epoch'
    assert config.training['checkpoint_epoch'] == 100
    assert config.training['max_epochs'] == 100
    assert config.training['early_stopping_patience'] == 100
    assert config.test_access == 'forbidden'
