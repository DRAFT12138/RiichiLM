"""Native integration fixtures use the reviewed source profile, without a model."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mahjong.contracts import MahjongError
from mahjong.envs.policy_bridge import RiichiDecisionBridge, PROFILE_DIGEST
from mahjong.inference import GeneratedDecision
from mahjong.inference.policy import legal_digest
from mahjong.tokenizer import MahjongTokenizer


PROFILE = json.loads(Path('configs/inference-profile.json').read_text())


def bridge(seed=42):
    b = RiichiDecisionBridge(PROFILE, model_binding_digest='test')
    b.reset(seed=seed)
    return b


def select(b, v, move=None):
    codec = MahjongTokenizer()
    suffixes = tuple(sorted(tuple(codec.encode_action_suffix(m.action)) for m in v.legal_moves))
    move = move or v.legal_moves[0]
    suffix = tuple(codec.encode_action_suffix(move.action))
    generated = GeneratedDecision(v.key, legal_digest(v,suffixes), suffixes.index(suffix),
        suffix, (0.,)*len(suffix), 0., 'end_decision', 'test')
    return b.bind_result(v, generated)


def test_native_reset_observations_and_snapshot_are_read_only():
    b = bridge()
    v = b.snapshot(0)
    assert v.rules_digest == PROFILE_DIGEST
    assert [e.kind for e in v.sequence.visible_history] == ['EV_DORA_REVEAL', 'EV_DRAW']
    assert [e.event_id for e in v.sequence.visible_history] == [1,2]
    assert len(b._transitions[(b._instance,0)].observations) == 12
    counts = b._seq
    for _ in range(3):
        assert b.snapshot(0) == v
    assert b._seq == counts
    assert b.snapshot(1) is None
    transition = b._transitions[(b._instance,0)]
    b._ingest_transition(transition)
    assert b.snapshot(0) == v
    with pytest.raises(MahjongError, match='EVENT_STREAM_INCOMPLETE'):
        b._ingest_transition(replace(transition, after_phase='corrupt'))
    with pytest.raises(MahjongError, match='EVENT_STREAM_INCOMPLETE'):
        b.snapshot(0)


def test_native_decision_loop_and_frozen_result_rejection():
    b = bridge()
    first = b.snapshot(0)
    old = select(b,first)
    result = b.apply(old)
    assert result.state_revision == 1
    with pytest.raises(MahjongError, match='STALE_DECISION|RESPONSE_BATCH_REQUIRED'):
        b.apply(old)
    # Bounded legal policy reaches a real native draw without any model or target.
    decisions = 1
    while result.status == 'advanced' and decisions < 200:
        views = [b.snapshot(p) for p in range(4)]
        views = [v for v in views if v is not None]
        from riichienv import Phase
        if b._native.phase == Phase.WaitResponse:
            responses = {v.key.viewer: select(b,v,next(m for m in v.legal_moves if m.action.kind=='A_PASS')) for v in views}
            result = b.apply_responses(views[0].key.window,responses)
        else:
            v = views[0]
            move = next((m for m in v.legal_moves if m.action.kind=='A_TSUMO_WIN'), None)
            move = move or next(m for m in v.legal_moves if m.action.kind=='A_DISCARD')
            result = b.apply(select(b,v,move))
        decisions += 1
    assert result.status in ('round_ended','game_ended')
    assert decisions < 200
    assert all(b.snapshot(p) is None for p in range(4))
    b.reset(seed=42)
    with pytest.raises(MahjongError, match='STALE_DECISION'):
        b.bind_result(first, old.generation)


def test_profile_rejection_before_native_or_snapshot():
    for mutate in (lambda p:p.update(revision='assumed-baseline-1'),
                   lambda p:p['score_policy'].update(dealer_first_discard='tedashi'),
                   lambda p:p['payment_policy'].update(honba_ron_each=400)):
        p = json.loads(json.dumps(PROFILE)); mutate(p)
        with pytest.raises(MahjongError, match='UNSUPPORTED_PROFILE'):
            RiichiDecisionBridge(p, model_binding_digest='test')


def test_binding_and_same_key_content_cannot_be_substituted():
    b = bridge(); v = b.snapshot(0); decision = select(b,v)
    with pytest.raises(MahjongError, match='CONTRACT_MISMATCH'):
        b.bind_result(v,replace(decision.generation,model_binding_digest='wrong'))
    with pytest.raises(MahjongError, match='CONTRACT_MISMATCH'):
        b.bind_result(v,replace(decision.generation,legal_set_digest='wrong'))
    from mahjong.inference import DecisionView
    other = DecisionView(v.key,v.sequence,v.legal_moves[:-1],v.rules_digest,True)
    with pytest.raises(MahjongError, match='STALE_DECISION'):
        b.bind_result(other, decision.generation)
    with pytest.raises(MahjongError, match='CONTRACT_MISMATCH'):
        b.apply(replace(decision, execution_ref='unknown'))


def rig(hands, *, drawn, oya=0, draws=()):
    from tests.unit.inference.bridge_fixtures import wall_for
    b = RiichiDecisionBridge(PROFILE,model_binding_digest='test')
    b._reset(wall=wall_for(hands,drawn=drawn,oya=oya,draws=draws),oya=oya,seed=123)
    return b


def play_kind(b, player, kind, tile=None):
    v = b.snapshot(player)
    move = next(m for m in v.legal_moves if m.action.kind==kind and
                (tile is None or m.action.args[0].name=='TILE_'+tile))
    return b.apply(select(b,v,move))


def test_complete_response_batch_is_atomic_and_collect_order_is_irrelevant():
    import itertools
    def fixture():
        b=rig({0:['5m','5m'],2:['3m','4m','6m']},drawn='9p',draws=['5m'])
        play_kind(b,0,'A_DISCARD','9p')
        from riichienv import Phase
        if b._native.phase == Phase.WaitResponse:
            views=[b.snapshot(p) for p in range(4) if b.snapshot(p) is not None]
            b.apply_responses(views[0].key.window, {v.key.viewer:select(b,v,next(m for m in v.legal_moves if m.action.kind=='A_PASS')) for v in views})
        play_kind(b,1,'A_DISCARD','5m')
        return b
    outcomes=[]
    for order in itertools.permutations([0,2]):
        b=fixture()
        views={p:b.snapshot(p) for p in range(4) if b.snapshot(p) is not None}
        assert {'A_PON','A_PASS'} <= {m.action.kind for m in views[0].legal_moves}
        assert {'A_CHI','A_PASS'} <= {m.action.kind for m in views[2].legal_moves}
        decisions={}
        seq,revision,events=b._seq,b._revision,list(b._builder.events)
        for p in order:
            v=views[p]
            kind='A_PON' if p==0 else 'A_CHI'
            decisions[p]=select(b,v,next(m for m in v.legal_moves if m.action.kind==kind))
            assert b._seq==seq and b._revision==revision and b._builder.events==events
        with pytest.raises(MahjongError,match='RESPONSE_BATCH_REQUIRED'):
            b.apply(decisions[0])
        with pytest.raises(MahjongError,match='RESPONSE_BATCH_INCOMPLETE'):
            b.apply_responses(views[0].key.window,{0:decisions[0]})
        assert b._seq==seq and b._revision==revision
        result=b.apply_responses(views[0].key.window,decisions)
        assert result.state_revision==revision+1 and b._seq==seq+1
        assert b.snapshot(0) is not None
        outcomes.append(b.snapshot(0).prefix_ids)
        with pytest.raises(MahjongError,match='WINDOW_ALREADY_CLOSED'):
            b.apply_responses(views[0].key.window,decisions)
    assert outcomes[0]==outcomes[1]


def test_riichi_is_one_complete_plan_with_two_consumed_transitions():
    b=rig({0:['1m','2m','3m','4m','5m','6m','2p','3p','4p','6s','7s','8s','E']},drawn='9p')
    v=b.snapshot(0)
    before=b._native.mjai_log
    choices=[m for m in v.legal_moves if m.action.kind=='A_RIICHI_DISCARD']
    assert choices
    assert b._native.mjai_log==before
    revision,seq=b._revision,b._seq
    b.apply(select(b,v,choices[0]))
    assert b._revision==revision+1 and b._seq==seq+2
    kinds=[e.kind for e in b._builder.events]
    assert kinds.index('EV_RIICHI_DECLARE') < kinds.index('EV_DISCARD') < kinds.index('EV_RESPONSE_OPEN')
    if b._builder.pending_window:
        assert 'EV_RIICHI_ACCEPT' not in kinds


def test_ankan_attempt_open_close_commit_and_rinshan():
    b=rig({0:['1m']*4+['2p','3p','4p','6p','7p','8p','5s','6s','E']},drawn='9s')
    play_kind(b,0,'A_ANKAN')
    events=b._builder.events
    kinds=[e.kind for e in events]
    assert kinds.index('EV_ANKAN_ATTEMPT') < kinds.index('EV_RESPONSE_OPEN')
    assert kinds.index('EV_RESPONSE_CLOSE') < kinds.index('EV_KAN_COMMIT')
    assert kinds[-1]=='EV_DRAW'
    assert events[-1].payload[-1].name=='DRAW_RINSHAN'
    assert len(b.snapshot(0).sequence.hand_state.melds)==1


def test_tsumo_and_kyuushu_finish_using_full_profile_then_advance():
    b=rig({0:['1m','2m','3m','4m','5m','6m','2p','3p','4p','6s','7s','8s','E']},drawn='E')
    result=play_kind(b,0,'A_TSUMO_WIN')
    assert result.status=='round_ended'
    assert b._round_outcome.next_dealer==0
    assert b.next_round().status=='advanced'
    assert b.snapshot(0).key.window.round_id==1
    assert b.snapshot(0).sequence.header['honba']==1
    b=rig({0:['1m','9m','1p','9p','1s','9s','E','S','W','N','P','F','2m']},drawn='3m')
    assert play_kind(b,0,'A_ABORT_KYUUSHU').status=='round_ended'
    assert b._round_outcome.next_dealer==0


def respond(b, choices=None):
    choices=choices or {}
    views=[b.snapshot(p) for p in range(4) if b.snapshot(p) is not None]
    return b.apply_responses(views[0].key.window,{v.key.viewer:select(b,v,
        next(m for m in v.legal_moves if m.action.kind==choices.get(v.key.viewer,'A_PASS'))) for v in views})


def test_daiminkan_and_red_pon_consumed_are_not_collapsed():
    b=rig({1:['1m']*3},drawn='1m')
    play_kind(b,0,'A_DISCARD','1m')
    respond(b,{1:'A_DAIMINKAN'})
    kinds=[e.kind for e in b._builder.events]
    assert kinds.index('EV_RESPONSE_CLOSE') < kinds.index('EV_DAIMINKAN') < kinds.index('EV_KAN_COMMIT')
    assert b.snapshot(1).sequence.visible_history[-1].payload[-1].name=='DRAW_RINSHAN'
    b=rig({1:['5m','5m','5mr']},drawn='5m')
    play_kind(b,0,'A_DISCARD','5m')
    v=b.snapshot(1)
    pons=[m for m in v.legal_moves if m.action.kind=='A_PON']
    assert {tuple(t.name for t in m.action.args[2:]) for m in pons} == {
        ('TILE_5m','TILE_5m'),('TILE_5m','TILE_5mr')}


def test_ron_pass_sets_furiten_only_at_atomic_adjudication():
    waiting=['P','P','P','1m','2m','3m','4m','5m','6m','2p','3p','E','E']
    for call in ('A_PASS','A_RON'):
        b=rig({2:waiting},drawn='4p')
        play_kind(b,0,'A_DISCARD','4p')
        v=b.snapshot(2)
        assert 'A_RON' in {m.action.kind for m in v.legal_moves}
        decision=select(b,v,next(m for m in v.legal_moves if m.action.kind==call))
        assert b._native.missed_agari_doujun[2] is False
        assert b._session.furiten.temp[2] is False
        respond(b,{2:call})
        if call=='A_PASS':
            # Player1 draws next, so player2's missed ron remains observable.
            assert b._native.missed_agari_doujun[2] is True
            assert b._session.furiten.temp[2] is True
        else:
            assert b._status=='round_ended'
            assert b._session.round_result==dict(kind='win',winners=[2])


def test_no_yaku_completion_blocks_other_end_of_wait_until_own_draw():
    # Open 678p; 1s completes without yaku, while 4s gives tanyao. The missed
    # 1s still forbids ron on 4s before player2's next draw.
    waiting=['5m','5m','3p','4p','5p','2s','3s','5s','6s','7s','6p','7p','E']
    b=rig({2:waiting},drawn='9m',draws=['8p','1s','4s','9p','9s'])

    def pass_responses():
        if b.is_response_window():
            respond(b,{})

    play_kind(b,0,'A_DISCARD','9m')
    pass_responses()
    play_kind(b,1,'A_DISCARD','8p')
    respond(b,{2:'A_CHI'})
    play_kind(b,2,'A_DISCARD','E')
    pass_responses()
    assert b._session.ron_available(2,'1s') is False
    assert b._session.ron_available(2,'4s') is True
    play_kind(b,3,'A_DISCARD','1s')
    assert b._native.missed_agari_doujun[2] is True
    assert b._session.furiten.temp[2] is True
    pass_responses()
    play_kind(b,0,'A_DISCARD','4s')
    assert b._session.ron_available(2,'4s') is False
    view=b.snapshot(2)
    assert view is None or all(m.action.kind!='A_RON' for m in view.legal_moves)
    pass_responses()
    play_kind(b,1,'A_DISCARD','9p')
    pass_responses()
    assert b.snapshot(2) is not None
    assert b._native.missed_agari_doujun[2] is False
    assert b._session.furiten.temp[2] is False


def test_kakan_keeps_original_stable_pon_reference():
    from riichienv import Phase
    b=rig({1:['5m','5m']},drawn='5m',draws=['2p','3s','4p','5mr'])
    play_kind(b,0,'A_DISCARD','5m')
    respond(b,{1:'A_PON'})
    v=b.snapshot(1)
    pon=v.sequence.hand_state.melds[0]
    move=next(m for m in v.legal_moves if m.action.kind=='A_DISCARD')
    b.apply(select(b,v,move))
    for _ in range(3):
        if b._native.phase == Phase.WaitResponse:
            respond(b)
        player=b._native.current_player
        v=b.snapshot(player)
        from riichienv.convert import tid_to_mjai
        drawn=tid_to_mjai(b._native.drawn_tile)
        play_kind(b,player,'A_DISCARD',drawn)
    if b._native.phase == Phase.WaitResponse:
        respond(b)
    v=b.snapshot(1)
    kakan=next(m for m in v.legal_moves if m.action.kind=='A_KAKAN')
    assert kakan.action.args[0]==pon['meld_id']
    b.apply(select(b,v,kakan))
    kinds=[e.kind for e in b._builder.events]
    assert kinds.index('EV_KAKAN_ATTEMPT') < kinds.index('EV_KAN_COMMIT')
    assert b.snapshot(1).sequence.hand_state.melds[0]['meld_id']==pon['meld_id']


def test_ankan_robbery_attempt_visibility_does_not_depend_on_hidden_eligibility():
    kokushi=['1m','9m','1p','9p','1s','9s','E','S','W','N','P','F','C']
    # Wait for 1m with the duplicate elsewhere; all 1m copies belong to actor0.
    kokushi.remove('1m'); kokushi.append('E')
    initial0=['1m']*4+['2p','3p','4p','6p','7p','8p','5s','6s','2m']
    b=rig({0:initial0,1:kokushi},drawn='9s')
    play_kind(b,0,'A_ANKAN')
    v=b.snapshot(1)
    assert {'A_RON','A_PASS'} == {m.action.kind for m in v.legal_moves}
    attempts=[e for e in v.sequence.visible_history if e.kind=='EV_ANKAN_ATTEMPT']
    assert len(attempts)==1
    assert {a.name for a in attempts[0].payload[2]}=={'TILE_HIDDEN'}
    assert all(e.kind!='EV_KAN_COMMIT' for e in v.sequence.visible_history)
    assert len(v.sequence.hand_state.melds)==0
    respond(b,{1:'A_RON'})
    kinds=[e.kind for e in b._builder.events]
    assert kinds.index('EV_RESPONSE_CLOSE') < kinds.index('EV_KAN_ROBBED') < kinds.index('EV_WIN')
    assert 'EV_KAN_COMMIT' not in kinds


def test_hidden_hand_and_wall_changes_preserve_own_prefix_and_complete_set():
    from tests.unit.inference.bridge_fixtures import wall_for
    wall=wall_for({0:['1m','2m','3m','4m','5m','6m','2p','3p','4p','6s','7s','8s','E']},drawn='9p')
    changed=list(wall)
    changed[12],changed[75]=changed[75],changed[12] # opponent3 private hand versus unseen wall
    bridges=[]
    for w in (wall,changed):
        b=RiichiDecisionBridge(PROFILE,model_binding_digest='test');b._reset(wall=w,seed=0)
        bridges.append(b)
    a,z=[b.snapshot(0) for b in bridges]
    assert a.prefix_ids==z.prefix_ids and a.legal_moves==z.legal_moves


def test_hidden_kokushi_eligibility_does_not_change_attempt_open_ids():
    from tests.unit.inference.bridge_fixtures import wall_for
    from mahjong.data.mahjong.cleaning.projection import project_event
    from mahjong.envs.policy_bridge import HISTORY_KINDS
    initial0=['1m']*4+['2p','3p','4p','6p','7p','8p','5s','6s','2m']
    kokushi=['9m','1p','9p','1s','9s','E','S','W','N','P','F','C','E']
    wall=wall_for({0:initial0,1:kokushi},drawn='9s')
    changed=list(wall);changed[4],changed[75]=changed[75],changed[4]
    prefixes=[]
    for w in (wall,changed):
        b=RiichiDecisionBridge(PROFILE,model_binding_digest='test');b._reset(wall=w,seed=0)
        play_kind(b,0,'A_ANKAN')
        attempt=next(e for e in b._builder.events if e.kind=='EV_ANKAN_ATTEMPT')
        opening=next(e for e in b._builder.events if e.kind=='EV_RESPONSE_OPEN')
        assert opening.payload[0]['window_id']==4*attempt.event_id+3
        prefixes.append([project_event(e,2,ankan_public='base_only') for e in b._builder.events
                         if e.kind in HISTORY_KINDS and e.event_id<=opening.event_id])
    assert prefixes[0]==prefixes[1]


def test_chi_success_publishes_after_call_discard_with_exact_consumed_tiles():
    b=rig({1:['3m','4m','6m']},drawn='5m')
    play_kind(b,0,'A_DISCARD','5m')
    before=b.snapshot(1)
    chi=next(m for m in before.legal_moves if m.action.kind=='A_CHI')
    respond(b,{1:'A_CHI'})
    view=b.snapshot(1)
    assert view is not None
    assert view.sequence.hand_state.melds[0]['name'].name=='MELD_CHI'
    assert {m.action.kind for m in view.legal_moves}=={'A_DISCARD'}
    assert all(m.action.args[1].name=='TEDASHI' for m in view.legal_moves)


def test_full_profile_nominal_round_limit_and_final_canonical_result():
    from riichienv import Phase
    b=bridge(seed=11)
    rounds=0
    for _ in range(2000):
        if b._status=='round_ended':
            rounds+=1
            b.next_round()
        if b._status=='game_ended':break
        if b._native.phase==Phase.WaitResponse:
            respond(b)
        else:
            p=b._native.current_player;v=b.snapshot(p)
            move=next((m for m in v.legal_moves if m.action.kind=='A_TSUMO_WIN'),None)
            move=move or next(m for m in v.legal_moves if m.action.kind=='A_DISCARD')
            b.apply(select(b,v,move))
    assert b._status=='game_ended'
    assert 8<=rounds<=24
    assert b._builder.events[-1].kind=='EV_GAME_END'
    assert b._session.finalized
    assert all(b.snapshot(p) is None for p in range(4))
    assert b._session.env.riichi_sticks==0


def test_riichi_second_step_failure_reports_partial_execution_and_invalidates():
    b=rig({0:['1m','2m','3m','4m','5m','6m','2p','3p','4p','6s','7s','8s','E']},drawn='9p')
    v=b.snapshot(0)
    decision=select(b,v,next(m for m in v.legal_moves if m.action.kind=='A_RIICHI_DISCARD'))
    native=b._native
    class FailingSecondStep:
        def __init__(self):self.calls=0
        def __getattr__(self,name):return getattr(native,name)
        def step(self,actions):
            self.calls+=1
            if self.calls==2:raise RuntimeError('injected second step failure')
            return native.step(actions)
    b._native=FailingSecondStep()
    with pytest.raises(MahjongError,match='ENVIRONMENT_STEP_FAILED') as error:
        b.apply(decision)
    assert error.value.details==dict(executed_steps=1,bridge_invalid=True)
    with pytest.raises(MahjongError,match='EVENT_STREAM_INCOMPLETE'):
        b.snapshot(0)


def test_old_transition_duplicate_is_idempotent_after_later_publication():
    b=bridge();original=b._transitions[(b._instance,0)]
    v=b.snapshot(0);b.apply(select(b,v))
    before=(b._revision,b._seq,tuple(b._builder.events),dict(b._views))
    b._ingest_transition(original)
    assert before==(b._revision,b._seq,tuple(b._builder.events),dict(b._views))
    with pytest.raises(MahjongError,match='EVENT_STREAM_INCOMPLETE'):
        b._ingest_transition(replace(original,after_phase='conflicting old transition'))
