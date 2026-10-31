import pytest
import torch

from mahjong.contracts import MahjongError, clone_state, derive_seed, validate_step_result
from mahjong.models import MockMLP
from tests.unit.data.replacements import FixedSource, ObjectPayload, ReadFailureSource
from tests.unit.models.replacements import (AllSkipModel, NonfiniteLossModel,
                                           PrivateGeneratorModel, ScalarModel)


@pytest.mark.parametrize('which', ['source', 'model', 'both'])
def test_replacement_adapters_support_real_update_eval_and_state_round_trip(which):
    from mahjong.data import SyntheticTrainSource
    schema = 'fixture-object/v1' if which == 'both' else 'synthetic-regression/v1'
    source = (SyntheticTrainSource(seed=1, sampler_seed=2, input_dim=2, train_size=4, batch_size=2)
              if which == 'model' else FixedSource(object_payload=which == 'both', input_schema=schema))
    model = (MockMLP(input_dim=2, init_seed=3, private_seed=4) if which == 'source'
             else ScalarModel(object_payload=which == 'both', input_schema=schema))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, foreach=False, fused=False)
    before = clone_state(model.state_dict())
    lease = source.acquire()
    result = model(model.prepare_batch(lease.batch, torch.device('cpu')), phase='train')
    validate_step_result(result, phase='train')
    result.loss.backward()
    optimizer.step()
    source.commit(lease.token)
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    saved = clone_state(model.state_dict())
    model.load_state_dict(saved, strict=True)
    model.eval()
    with torch.inference_mode():
        eval_result = model(model.prepare_batch(source.acquire().batch, torch.device('cpu')), phase='eval')
    validate_step_result(eval_result, phase='eval')


def test_private_rng_fixture_export_import_and_reset_are_owned():
    learner = PrivateGeneratorModel(input_dim=2, init_seed=3, private_seed=4)
    provider = FixedSource()
    batch = provider.acquire().batch
    initial = learner.auxiliary_state_dict()
    first = learner(batch, phase='train')
    learner.load_auxiliary_state_dict(initial)
    second = learner(batch, phase='train')
    assert torch.equal(first.loss, second.loss)
    initial['private_rng'].zero_()
    initial['owned']['nested'][1][7].zero_()
    assert learner.owned['nested'][1][7].item() == 3.
    learner.reset_auxiliary_rng(12)
    prior = learner.auxiliary_state_dict()
    learner(batch, phase='eval')
    assert not torch.equal(prior['private_rng'], learner.auxiliary_state_dict()['private_rng'])


def test_fault_fixtures_have_stable_error_or_legal_skip():
    provider = FixedSource()
    batch = provider.acquire().batch
    skipped = AllSkipModel()
    before = clone_state(skipped.state_dict())
    validate_step_result(skipped(batch, phase='train'), phase='train')
    assert all(torch.equal(value, skipped.state_dict()[key]) for key, value in before.items())
    for value in [float('nan'), float('inf')]:
        result = NonfiniteLossModel(bad_value=value)(batch, phase='train')
        with pytest.raises(MahjongError) as failure:
            validate_step_result(result, phase='train')
        assert failure.value.code == 'NONFINITE_LOSS'
    with pytest.raises(MahjongError) as failure:
        ReadFailureSource(fail_at=1).acquire()
    assert failure.value.code == 'DATA_READ_FAILED'
    with pytest.raises(MahjongError) as failure:
        FixedSource(values=torch.empty(0, 2)).acquire()
    assert failure.value.code == 'EMPTY_TRAIN_DATA'
