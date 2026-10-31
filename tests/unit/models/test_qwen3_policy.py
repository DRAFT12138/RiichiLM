"""Legal sets of different sizes must remain independent when batched."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mahjong.app.assembly import build_model, build_train_source
from mahjong.app.config import parse_model, parse_new
from mahjong.contracts import MahjongError, validate_step_result
from mahjong.data.mahjong.policy import _collate
from mahjong.models.qwen3 import POLICY_INPUT_SCHEMA, Qwen3PolicyAdapter


def test_variable_legal_sets_match_individual_losses_and_backward():
    torch.set_num_threads(1)
    config = parse_model([
        'model.num_hidden_layers=1', 'model.hidden_size=32',
        'model.intermediate_size=64', 'model.num_attention_heads=4',
        'model.num_key_value_heads=2', 'model.head_dim=8',
        'model.vocab_size=32', 'model.max_position_embeddings=32',
        'model.tie_word_embeddings=false',
    ])
    model = Qwen3PolicyAdapter(build_model(config).causal_lm)
    samples = [
        dict(input_ids=[1, 5, 6], candidates=[[7, 8], [9]], target_index=0, sample_id='a'),
        dict(input_ids=[1, 3, 4, 5], candidates=[[7], [8, 9], [10]], target_index=2, sample_id='b'),
    ]
    batch = _collate(samples)
    assert batch.input_schema == POLICY_INPUT_SCHEMA
    result = model(batch, phase='train')
    validate_step_result(result, phase='train')
    singles = [model(_collate([s]), phase='eval') for s in samples]
    counts = [len(s['candidates'][s['target_index']]) for s in samples]
    expected = sum(r.loss * n for r, n in zip(singles, counts)) / sum(counts)
    torch.testing.assert_close(result.loss, expected)
    # Independent autoregressive reference: call the official vocabulary head
    # using only the context and already observed action tokens at each step.
    terms = []
    for sample in samples:
        target = sample['candidates'][sample['target_index']]
        for offset, token in enumerate(target):
            observed = target[:offset]
            ids = torch.tensor([sample['input_ids'] + observed])
            logits = model.causal_lm(input_ids=ids, use_cache=False).logits[0, -1]
            allowed = sorted({a[offset] for a in sample['candidates']
                              if len(a) > offset and a[:offset] == observed})
            terms.append(torch.logsumexp(logits[allowed], 0) - logits[token])
    torch.testing.assert_close(result.loss, torch.stack(terms).mean())
    assert result.metrics['ntp_loss'].denominator == 3
    result.loss.backward()
    gradient = model.causal_lm.get_input_embeddings().weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    # Neither padded input tokens nor padded legal-action slots train PAD.
    assert torch.count_nonzero(gradient[0]) == 0
    # An independent lm_head must participate; an embedding classifier would
    # leave this gradient absent when input/output embeddings are untied.
    head_gradient = model.causal_lm.get_output_embeddings().weight.grad
    assert head_gradient is not None and torch.count_nonzero(head_gradient) > 0
    assert torch.count_nonzero(head_gradient[0]) == 0


class TransitionLM(torch.nn.Module):
    """Known vocabulary logits at each previous token for hand-computed NTP."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32, max_position_embeddings=32)
        table = torch.zeros(32, 32)
        table[:, 25] = 1000  # An illegal, dominant logit must have no influence.
        table[6, 7] = 2
        table[7, 9] = 1
        self.table = torch.nn.Parameter(table)

    def forward(self, *, input_ids, **kwargs):
        self.last_ids = input_ids.detach().clone()
        self.last_mask = kwargs['attention_mask'].detach().clone()
        return SimpleNamespace(logits=self.table[input_ids])


def branching_sample():
    return dict(input_ids=[1, 5, 6], candidates=[
        [7, 8, 30], [7, 9, 30], [10, 11, 30], [10, 12, 30]],
        target_index=1, sample_id='branched-action')


def test_hand_computed_shifted_ntp_ignores_context_illegal_logits_and_future_branches():
    lm = TransitionLM()
    model = Qwen3PolicyAdapter(lm)
    sample = branching_sample()
    result = model(_collate([sample]), phase='train')
    # First token: {7,10}; after 7: {8,9}, not {11,12}; after 7,9: {30}.
    expected = (-torch.log_softmax(torch.tensor([2., 0.]), 0)[0]
                -torch.log_softmax(torch.tensor([0., 1.]), 0)[1]) / 3
    torch.testing.assert_close(result.loss, expected)
    assert lm.last_ids.tolist() == [[1, 5, 6, 7, 9, 30]]
    assert result.metrics['loss'].denominator == 3
    assert result.metrics['token_accuracy'].numerator == 3
    result.loss.backward()
    gradient = lm.table.grad
    assert torch.isfinite(gradient).all()
    assert set(map(tuple, torch.nonzero(gradient).tolist())) == {(6, 7), (6, 10), (7, 8), (7, 9)}


def test_teacher_forcing_labels_and_padding_only_select_target_action_tokens():
    model = Qwen3PolicyAdapter(TransitionLM())
    other = dict(input_ids=[1], candidates=[[11, 30], [12, 30]], target_index=0, sample_id='short')
    batch = _collate([branching_sample(), other])
    payload, active = model._payload(batch)
    ids, mask, labels, legal = model._teacher_forcing(payload, active)
    assert ids.tolist() == [[1, 5, 6, 7, 9, 30], [1, 11, 30, 0, 0, 0]]
    assert labels.tolist() == [[-100, -100, -100, 7, 9, 30], [-100, 11, 30, -100, -100, -100]]
    assert mask.tolist() == [[True]*6, [True]*3+[False]*3]
    assert [torch.nonzero(row).flatten().tolist() for row in legal] == [[7, 10], [8, 9], [30], [11, 12], [30]]
    result = model(batch, phase='train')
    assert result.metrics['loss'].denominator == 5


@pytest.mark.parametrize('candidates', [[[7], [7, 8]], [[7, 8], [7]], [[7], [7]]])
def test_rejects_ambiguous_or_duplicate_action_suffixes(candidates):
    sample = dict(input_ids=[1], candidates=candidates, target_index=0, sample_id='bad')
    with pytest.raises(MahjongError, match='prefix|duplicate'):
        Qwen3PolicyAdapter(TransitionLM())(_collate([sample]), phase='train')


def test_context_budget_includes_longest_legal_suffix():
    lm = TransitionLM()
    lm.config.max_position_embeddings = 4
    sample = dict(input_ids=[1, 5, 6], candidates=[[7], [8, 9]], target_index=0, sample_id='long')
    with pytest.raises(MahjongError, match='longest legal action exceeds context'):
        Qwen3PolicyAdapter(lm)(_collate([sample]), phase='train')


def test_softmax_then_renormalize_equals_masked_logits_but_double_softmax_does_not():
    logits = torch.tensor([2., 1., 4.], dtype=torch.float64, requires_grad=True)
    direct = logits[:2].softmax(0)
    probabilities = logits.softmax(0)[:2]
    renormalized = probabilities / probabilities.sum()
    torch.testing.assert_close(direct, renormalized)
    assert not torch.allclose(direct, probabilities.softmax(0))
    direct_gradient = torch.autograd.grad(-direct[0].log(), logits, retain_graph=True)[0]
    renormalized_gradient = torch.autograd.grad(-renormalized[0].log(), logits)[0]
    torch.testing.assert_close(direct_gradient, renormalized_gradient)
    # Full-vocabulary softmax can underflow every legal probability to zero.
    extreme = torch.tensor([2., 1., 1000.])
    assert extreme.softmax(0)[:2].sum() == 0
    assert torch.isfinite(torch.nn.functional.cross_entropy(extreme[:2].unsqueeze(0), torch.tensor([0])))


def test_synthetic_source_still_accepts_default_app_configuration():
    source = build_train_source(parse_new([], Path.cwd()))
    lease = source.acquire()
    assert lease.batch.input_schema == 'synthetic-regression/v1'
    source.commit(lease.token)
    assert not source.has_open_lease()
