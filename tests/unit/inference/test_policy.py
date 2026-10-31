"""Independent next-token references and fail-before-forward boundaries."""
from dataclasses import replace
import math
import time

import pytest
import torch

from mahjong.app.assembly import build_model
from mahjong.app.config import parse_model
from mahjong.contracts import MahjongError
from mahjong.data.mahjong.policy import _collate
from mahjong.inference import (DecisionKey, WindowKey, DecisionView, LegalMove,
    DecodeOptions, PolicyInference)
from mahjong.models.qwen3 import Qwen3PolicyAdapter
from mahjong.tokenizer import Action, Atom, HandMeldState, PolicySequenceInput


@pytest.fixture
def inference():
    torch.set_num_threads(1)
    config = parse_model(['model.num_hidden_layers=1', 'model.hidden_size=32',
        'model.intermediate_size=64', 'model.num_attention_heads=4',
        'model.num_key_value_heads=2', 'model.head_dim=8',
        'model.vocab_size=1024', 'model.max_position_embeddings=256'])
    return PolicyInference(Qwen3PolicyAdapter(build_model(config).causal_lm))


def view(actions=None, revision=0):
    a = Atom
    sequence = PolicySequenceInput(dict(mode=a('MODE_HANCHAN'), round_wind=a('ROUND_E'),
        round_number=1, honba=0, kyotaku=0, dealer=a('SELF'), scores=[25000]*4,
        seat_winds=[a('TILE_'+x) for x in 'ESWN']), [], HandMeldState([a('TILE_1m')]*13, []))
    actions = actions or [Action('A_PASS'), Action('A_CHI',
        (a('KAMICHA'), a('TILE_5m'), a('TILE_3m'), a('TILE_4m'))),
        Action('A_CHI', (a('KAMICHA'), a('TILE_5m'), a('TILE_4m'), a('TILE_6m')))]
    return DecisionView(DecisionKey(WindowKey('env', 'game', 0, 14, revision), 0),
        sequence, [LegalMove(action, str(i)) for i, action in enumerate(actions)], 'rules', True)


def test_cache_free_greedy_logprobs_equal_independent_reference_and_training_loss(inference):
    request = inference.prepare(view())
    outputs = [inference.generate(request, options=DecodeOptions(use_kv_cache=cache)) for cache in (False, True)]
    assert outputs[0].suffix_ids == outputs[1].suffix_ids
    assert outputs[0].token_logprobs == pytest.approx(outputs[1].token_logprobs, abs=1e-5)
    generated = []
    reference = []
    with torch.inference_mode():
        for token in outputs[0].suffix_ids:
            ids = torch.tensor([list(request.prefix_ids)+generated])
            logits = inference.lm(input_ids=ids, use_cache=False).logits[0,-1]
            allowed = sorted({a[len(generated)] for a in request.legal_suffixes if a[:len(generated)] == tuple(generated)})
            assert token == min(allowed, key=lambda t: (-float(logits[t]), t))
            reference.append(float(torch.log_softmax(logits[allowed], 0)[allowed.index(token)]))
            generated.append(token)
    assert outputs[0].token_logprobs == pytest.approx(reference, abs=1e-5)
    batch = _collate([dict(input_ids=list(request.prefix_ids), candidates=[list(s) for s in request.legal_suffixes],
        target_index=outputs[0].candidate_index, sample_id='inference-reference')])
    loss = inference.model(batch, phase='eval').loss.item()
    assert loss == pytest.approx(-math.fsum(reference)/len(reference), abs=1e-6)
    assert outputs[0].model_action_logprob == pytest.approx(math.fsum(reference))
    assert reference[-1] == 0


def test_known_logits_illegal_dominance_ties_and_branch_conditioning(inference, monkeypatch):
    original = inference.lm.forward
    def forward(**kwargs):
        out = original(**kwargs)
        out.logits.fill_(0)
        out.logits[..., 2] = 1000  # EOS remains illegal, without underflowing legal probabilities.
        return out
    monkeypatch.setattr(inference.lm, 'forward', forward)
    req = inference.prepare(view())
    result = inference.generate(req)
    assert result.suffix_ids == min(req.legal_suffixes)
    assert result.token_logprobs[0] == pytest.approx(-math.log(2))
    assert result.suffix_ids[-1] == 30
    original_view = view()
    other = inference.prepare(DecisionView(original_view.key, original_view.sequence,
        reversed(original_view.legal_moves), original_view.rules_digest, True))
    assert other == req
    assert inference.generate(other) == result


def test_single_candidate_and_context_exact_boundary(inference):
    req = inference.prepare(view([Action('A_KAKAN', (123456, Atom('TILE_5mr')))]))
    inference.lm.config.max_position_embeddings = len(req.prefix_ids) + len(req.legal_suffixes[0])
    result = inference.generate(req)
    assert result.token_logprobs == (0.,)*len(result.suffix_ids)
    inference.lm.config.max_position_embeddings -= 1
    with pytest.raises(MahjongError, match='CONTEXT_TOO_LONG'):
        inference.generate(req)


@pytest.mark.parametrize('when', ['cancelled', 'expired', 'prepare_expired', 'during_forward', 'verify'])
def test_cancel_boundaries(inference, monkeypatch, when):
    req = inference.prepare(view())
    calls = []
    cancelled = [when == 'cancelled']
    original = inference.lm.forward
    def forward(**kwargs):
        calls.append(kwargs)
        out = original(**kwargs)
        if when == 'during_forward':
            cancelled[0] = True
        return out
    monkeypatch.setattr(inference.lm, 'forward', forward)
    if when == 'prepare_expired':
        validate = inference._validate
        def expire(*args):
            result = validate(*args)
            cancelled[0] = True
            return result
        monkeypatch.setattr(inference, '_validate', expire)
    if when == 'verify':
        decode = inference.codec.decode_action_suffix
        def cancel_at_finish(ids):
            if calls:
                cancelled[0] = True
            return decode(ids)
        monkeypatch.setattr(inference.codec, 'decode_action_suffix', cancel_at_finish)
    with pytest.raises(MahjongError, match='CANCELLED|DEADLINE_EXCEEDED'):
        inference.generate(req, cancel=lambda: cancelled[0],
            deadline_monotonic=time.monotonic()-1 if when == 'expired' else None)
    if when in ('cancelled', 'expired', 'prepare_expired'):
        assert len(calls) == 0
    else:
        assert len(calls) >= 1
    # No cache stored on the inference object across failures.
    assert not hasattr(inference, 'cache')


def test_immutable_view_request_and_bad_candidates(inference):
    v = view()
    seq = v.sequence
    seq.header['scores'][0] = -999
    assert v.sequence.header['scores'][0] == 25000
    req = inference.prepare(v)
    with pytest.raises(MahjongError, match='STALE_DECISION'):
        inference.generate(replace(req, prefix_ids=req.prefix_ids+(2,)))
    with pytest.raises(MahjongError, match='duplicate'):
        inference.prepare(view([Action('A_PASS'), Action('A_PASS')]))
    with pytest.raises(MahjongError, match='STALE_DECISION'):
        inference.prepare(view([Action('A_PASS')]))
    for suffix in ((2,30), (832,30), (30,), (req.legal_suffixes[0]+(2,))):
        with pytest.raises(MahjongError):
            inference._validate(req.prefix_ids, (suffix,))
    with pytest.raises(MahjongError, match='LEGAL_SET_INCOMPLETE'):
        inference.prepare(DecisionView(v.key, v.sequence, v.legal_moves, v.rules_digest, False))


def test_nonfinite_legal_logit_rejected(inference, monkeypatch):
    req = inference.prepare(view())
    original = inference.lm.forward
    def forward(**kwargs):
        out = original(**kwargs)
        out.logits.fill_(torch.nan)
        return out
    monkeypatch.setattr(inference.lm, 'forward', forward)
    with pytest.raises(MahjongError, match='NONFINITE_LOGITS'):
        inference.generate(req)


def test_cache_raw_logits_positions_and_key_mask_lengths(inference,monkeypatch):
    request=inference.prepare(view())
    records=[]
    original=inference.lm.forward
    def spy(**kwargs):
        output=original(**kwargs)
        records.append((kwargs['input_ids'].shape[1],kwargs['attention_mask'].shape[1],
            kwargs['position_ids'].tolist(),output.logits[0,-1].clone()))
        return output
    monkeypatch.setattr(inference.lm,'forward',spy)
    a=inference.generate(request,options=DecodeOptions(use_kv_cache=False));off=list(records);records.clear()
    b=inference.generate(request,options=DecodeOptions(use_kv_cache=True));on=list(records)
    assert a.suffix_ids==b.suffix_ids
    for i,(x,y) in enumerate(zip(off,on)):
        torch.testing.assert_close(x[3],y[3],atol=1e-5,rtol=1e-5)
        assert y[0]==(len(request.prefix_ids) if i==0 else 1)
        assert y[1]==len(request.prefix_ids)+i
        assert y[2][0][-1]==len(request.prefix_ids)+i-1


def test_longest_candidate_is_budgeted_before_any_forward(inference,monkeypatch):
    v=view();length=len(v.prefix_ids)
    inference.lm.config.max_position_embeddings=length+2 # PASS fits, CHI does not.
    monkeypatch.setattr(inference.lm,'forward',lambda **_:pytest.fail('forward before length rejection'))
    with pytest.raises(MahjongError,match='CONTEXT_TOO_LONG') as error:
        inference.prepare(v)
    assert error.value.details==dict(prefix_length=length,longest_suffix=6,context=length+2)


def test_model_exception_is_structured_and_next_request_has_fresh_cache(inference,monkeypatch):
    request=inference.prepare(view())
    original=inference.lm.forward
    def fail_forward(**kwargs):raise RuntimeError('injected model failure')
    monkeypatch.setattr(inference.lm,'forward',fail_forward)
    with pytest.raises(MahjongError,match='MODEL_EXECUTION_FAILED') as error:
        inference.generate(request)
    assert error.value.details['stage']=='forward'
    assert error.value.details['decision_key']['viewer']==0
    monkeypatch.setattr(inference.lm,'forward',original)
    assert inference.generate(request).finish_reason=='end_decision'


def test_controlled_logits_visit_every_complete_action_leaf(inference,monkeypatch):
    a=Atom
    actions=[Action('A_PASS'),Action('A_DISCARD',(a('TILE_5mr'),a('TEDASHI'))),
        Action('A_RIICHI_DISCARD',(a('TILE_5m'),a('TSUMOGIRI'))),Action('A_TSUMO_WIN'),
        Action('A_RON',(a('TOIMEN'),a('TILE_E'))),
        Action('A_CHI',(a('KAMICHA'),a('TILE_5m'),a('TILE_3m'),a('TILE_4m'))),
        Action('A_PON',(a('SHIMOCHA'),a('TILE_5m'),a('TILE_5m'),a('TILE_5mr'))),
        Action('A_DAIMINKAN',(a('TOIMEN'),a('TILE_E'),a('TILE_E'),a('TILE_E'),a('TILE_E'))),
        Action('A_ANKAN',(a('TILE_5m'),a('TILE_5m'),a('TILE_5m'),a('TILE_5mr'))),
        Action('A_KAKAN',(123456,a('TILE_5mr'))),Action('A_ABORT_KYUUSHU')]
    request=inference.prepare(view(actions))
    original=inference.lm.forward
    for expected in request.legal_suffixes:
        calls=[]
        def forward(**kwargs):
            out=original(**kwargs)
            out.logits.fill_(0)
            out.logits[...,expected[len(calls)]]=100
            out.logits[...,2]=1000
            calls.append(1)
            return out
        monkeypatch.setattr(inference.lm,'forward',forward)
        result=inference.generate(request)
        assert result.suffix_ids==expected
        node = request.legal_suffixes
        branch_end = 0
        for offset, token in enumerate(expected):
            if len({s[offset] for s in node}) > 1:
                branch_end = offset + 1
            node = [s for s in node if s[offset] == token]
        assert len(calls)==branch_end
        assert result.candidate_index==request.legal_suffixes.index(expected)


@pytest.mark.parametrize('cache', [False, True])
@pytest.mark.parametrize('strategy', ['greedy', 'categorical'])
def test_deterministic_tail_matches_reference_and_sampling_rng(inference, monkeypatch, cache, strategy):
    # Includes a singleton action, a branch immediately, and a shared first
    # token followed by a branch. The latter must retain its intervening K/V.
    actions = [[Action('A_PASS')], None,
        [Action('A_DISCARD', (Atom('TILE_1m'), Atom('TEDASHI'))),
         Action('A_DISCARD', (Atom('TILE_2m'), Atom('TEDASHI')))]]
    rngs = [torch.Generator().manual_seed(42) for _ in range(2)]
    calls = []
    original = inference.lm.forward
    def forward(**kwargs):
        calls.append(kwargs['input_ids'].shape[-1])
        return original(**kwargs)
    monkeypatch.setattr(inference.lm, 'forward', forward)
    for revision in range(15):
        request = inference.prepare(view(actions[revision % len(actions)], revision))
        outputs, counts = [], []
        for fast, rng in zip((False, True), rngs):
            calls.clear()
            outputs.append(inference.generate(request,
                options=DecodeOptions(strategy, cache, fast), generator=rng))
            counts.append(len(calls))
        assert outputs[0] == outputs[1]
        assert torch.equal(rngs[0].get_state(), rngs[1].get_state())
        assert counts[1] < counts[0]
        if revision % len(actions) == 0:
            assert counts[1] == 0
        if revision % len(actions) == 2:
            assert counts[1] >= 2


def test_forced_action_never_executes_model_but_remains_cancellable(inference, monkeypatch):
    request = inference.prepare(view([Action('A_PASS')]))
    monkeypatch.setattr(inference.lm, 'forward', lambda **_: pytest.fail('forced model forward'))
    assert inference.generate(request).token_logprobs == (0., 0.)
    checks = []
    def cancel():
        checks.append(True)
        return len(checks) >= 3
    with pytest.raises(MahjongError, match='CANCELLED'):
        inference.generate(request, cancel=cancel)


def test_empty_legal_set_and_old_enumeration_mapping_are_rejected(inference):
    v=view()
    with pytest.raises(MahjongError,match='LEGAL_SET_EMPTY'):
        inference.prepare(DecisionView(v.key,v.sequence,(),v.rules_digest,True))
    inference.prepare(v)
    changed=[LegalMove(m.action,'changed') for m in v.legal_moves]
    with pytest.raises(MahjongError,match='STALE_DECISION'):
        inference.prepare(DecisionView(v.key,v.sequence,changed,v.rules_digest,True))
