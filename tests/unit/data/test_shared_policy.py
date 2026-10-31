"""Grouping retains supervision, deterministic budget splits and committed cursors."""
import copy
from pathlib import Path

import pytest
import torch

from mahjong.app.assembly import build_eval_source, build_model, build_train_source
from mahjong.app.config import parse_new
from mahjong.contracts import MahjongError
from mahjong.data.mahjong import policy
from mahjong.data.mahjong.shared_policy import MahjongSharedPolicyTrainSource, plan_shared_groups
from mahjong.policy.shared_context import INDEPENDENT_LAYOUT, NANIKIRU_LAYOUT
from tests.unit.models.test_qwen3_shared import samples
from tests.unit.data.test_corpus import corpus


def test_grouping_cuts_every_budget_and_retains_tail_non_nested_and_view():
    rows=samples()
    for budgets in [(2,128,10000),(8,24,10000),(8,128,150)]:
        groups=plan_shared_groups(rows,max_decisions=budgets[0],max_tokens=budgets[1],max_branch_cells=budgets[2])
        assert len(groups)>1
        assert [s['sample_id'] for g in groups for s in g]==[s['sample_id'] for s in rows]
        for group in groups:
            P=group[-1]['cut_length']
            B=sum(len(s['input_ids'])-s['cut_length']+len(s['candidates'][s['target_index']]) for s in group)
            assert len(group)<=budgets[0] and P+B<=budgets[1] and B*(P+B)<=budgets[2]
    rows[1]['input_ids'][0]=99
    groups=plan_shared_groups(rows,max_decisions=8,max_tokens=128,max_branch_cells=10000)
    assert len(groups)==3
    rows=samples(); rows[1]['group_key']=('source',0,1)
    groups=plan_shared_groups(rows,max_decisions=8,max_tokens=128,max_branch_cells=10000)
    assert [len(g) for g in groups]==[2,1]


class Dataset:
    fingerprint='shared-fixture'
    def __init__(self,*args,**kwargs):pass
    def split_indices(self,validation):return torch.arange(3)
    def samples_for_game(self,index,**kwargs):return samples() if index==0 else []
    def close(self):pass


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.setattr(policy,'_Dataset',Dataset)


def source(**kwargs):
    return MahjongSharedPolicyTrainSource(build_dir='unused',seed=7,batch_size=2,
        max_sequence_length=128,max_tokens=128,shuffle=True,**kwargs)


def assert_state_equal(a,b):
    assert a.keys()==b.keys()
    for key,value in a.items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,b[key])
        else:assert value==b[key]


def test_lease_state_is_committed_and_restores_next_group_through_epoch(dataset):
    train=source()
    for _ in range(7):
        before=train.state_dict(); lease=train.acquire()
        assert_state_equal(before,train.state_dict())
        assert 1<=lease.batch.sample_count<=2
        with pytest.raises(MahjongError,match='lease already open'):train.acquire()
        train.commit(lease.token)
        restored=source(); restored.load_state_dict(train.state_dict())
        expected,actual=train.acquire(),restored.acquire()
        assert expected.batch.sample_ids==actual.batch.sample_ids
        train.commit(expected.token); restored.commit(actual.token)
        assert_state_equal(train.state_dict(),restored.state_dict())


@pytest.mark.parametrize('change',['planner','legacy','cursor'])
def test_changed_planner_old_state_and_invalid_cursor_rejected(dataset,change):
    train=source(); state=copy.deepcopy(train.state_dict())
    if change=='planner':state['planning_fingerprint']='changed'
    elif change=='legacy':
        state.pop('planning_fingerprint')
        state['sample_cursor']=state.pop('group_cursor')
    else:state['group_cursor']=9999
    with pytest.raises(MahjongError,match='CHECKPOINT_INVALID'):source().load_state_dict(state)


def test_empty_games_fail_without_infinite_epoch_loop(dataset,monkeypatch):
    monkeypatch.setattr(Dataset,'samples_for_game',lambda *args,**kwargs:[])
    with pytest.raises(MahjongError,match='no labelled decisions'):source().acquire()


def test_default_assembly_uses_shared_train_but_original_limited_eval_cohort(dataset,monkeypatch):
    monkeypatch.setattr(Dataset,'samples_for_game',lambda *args,**kwargs:samples())
    cfg=parse_new(['data.build_dir=unused','data.validation_max_samples=5'],Path.cwd(),config_name='pretrain')
    train=build_train_source(cfg)
    assert isinstance(train,MahjongSharedPolicyTrainSource) and train.batch_size==8
    evaluator=build_eval_source(cfg)
    batches=list(evaluator.iter_batches())
    corpus_batches=[b for b in batches if b.payload['layout']==INDEPENDENT_LAYOUT]
    questions=[b for b in batches if b.payload['layout']==NANIKIRU_LAYOUT]
    assert sum(b.sample_count for b in corpus_batches)==5
    assert [s for b in corpus_batches for s in b.sample_ids]==['decision-0','decision-1','decision-2','decision-0','decision-1']
    assert len(questions)==(601 if cfg["evaluation"]["nanikiru_dir"] is not None else 0)
    assert all(b.sample_count==1 for b in questions)
    assert batches==corpus_batches+questions
    other=copy.deepcopy(cfg); other['data']['shared_context']=False
    baseline=list(build_eval_source(other).iter_batches())
    assert [s for b in batches for s in b.sample_ids]==[s for b in baseline for s in b.sample_ids]


@pytest.mark.parametrize('override',['data.shared_max_tokens=0','data.shared_max_branch_cells=0',
    'data.input_schema=majang-policy-ntp/v1','data.drop_last=true','model.attn_implementation=eager'])
def test_invalid_shared_configuration_rejected(override):
    with pytest.raises(MahjongError,match='CONFIG_INVALID'):
        parse_new([override],Path.cwd(),config_name='pretrain')


def test_published_text_reader_parts_reconstruct_all_original_sequences(corpus):
    from mahjong.data.mahjong.policy import _Dataset
    from mahjong.policy.context import read_context_prefix,split_policy_prefix
    from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
    dataset=_Dataset(corpus,validation_fraction=.5,split_seed=7,max_sequence_length=8192,
                     input_schema=CONTEXT_SCHEMA)
    try:
        count=0
        for index,(source,_) in enumerate(dataset.games):
            original=dataset.samples_for_game(index)
            shared=dataset.samples_for_game(index,shared_parts=True)
            assert [s['sample_id'] for s in original]==[s['sample_id'] for s in shared]
            for a,b in zip(original,shared):
                assert a['input_ids']==b['input_ids'] and a['candidates']==b['candidates']
                assert a['target_index']==b['target_index']
                _,rid,viewer=b['group_key']
                trunk,private,_=split_policy_prefix(dataset.reader,source,rid,viewer,b['window_id'])
                assert trunk+private==read_context_prefix(dataset.reader,source,rid,viewer,b['window_id'])[0]
                assert len(trunk)==b['cut_length']
                # At late rounds past histories contain an earlier END_HISTORY.
                if rid>=3:assert trunk.count(dataset.reader.codec.token_id('END_HISTORY'))==3
                count+=1
        assert count==20
    finally:dataset.close()


def test_prefetch_matches_groups_rng_and_committed_restore_through_empty_games(dataset):
    plain=source(); ahead=source(); restored=source()
    ahead.configure_prefetch(True)
    try:
        global_rng=torch.get_rng_state().clone()
        for index in range(12):
            a,b=plain.acquire(),ahead.acquire()
            assert a.batch.sample_ids==b.batch.sample_ids
            plain.commit(a.token);ahead.commit(b.token)
            assert_state_equal(plain.state_dict(),ahead.state_dict())
            if index==4:
                restored.configure_prefetch(True)
                restored.load_state_dict(ahead.state_dict())
                assert restored._prefetch_executor is not None
                a,b=ahead.acquire(),restored.acquire()
                assert a.batch.sample_ids==b.batch.sample_ids
                ahead.commit(a.token);restored.commit(b.token)
                plain.load_state_dict(ahead.state_dict())
        assert torch.equal(global_rng,torch.get_rng_state())
    finally:
        plain.close();ahead.close();restored.close()


@pytest.mark.parametrize('action',['close','disable','restore','invalid_restore','consume'])
def test_future_error_is_only_raised_when_consumed(dataset,monkeypatch,action):
    import threading
    failed=threading.Event()
    def prepare(self,index,**kwargs):
        if index==1:
            failed.set();raise ValueError('future game failed')
        return samples()
    monkeypatch.setattr(Dataset,'samples_for_game',prepare)
    train=MahjongSharedPolicyTrainSource(build_dir='unused',seed=7,batch_size=8,
        max_sequence_length=128,max_tokens=128,shuffle=False)
    train.configure_prefetch(True)
    try:
        saved=train.state_dict()  # Restore the valid current game, not the failing next game.
        lease=train.acquire();assert failed.wait(5)
        train.commit(lease.token)
        if action=='consume':
            with pytest.raises(ValueError,match='future game failed'):train.acquire()
        elif action=='disable':train.configure_prefetch(False)
        elif action=='restore':
            train.load_state_dict(saved)
            assert train._prefetch_executor is not None
        elif action=='invalid_restore':
            state=copy.deepcopy(saved);state['group_cursor']=999
            with pytest.raises(MahjongError,match='CHECKPOINT_INVALID'):
                train.load_state_dict(state)
            assert train._prefetch_executor is None and train._prefetch_future is None
    finally:train.close()


def test_prefetch_exclusively_hands_reader_back_before_close(dataset,monkeypatch):
    import threading
    started=threading.Event();release=threading.Event();closed=threading.Event()
    active=threading.Lock()
    def prepare(self,index,**kwargs):
        assert active.acquire(blocking=False),'concurrent reader access'
        try:
            if index==1:
                started.set();assert release.wait(5)
            return samples()
        finally:active.release()
    def close(self):
        assert active.acquire(blocking=False),'reader closed while worker active'
        active.release();closed.set()
    monkeypatch.setattr(Dataset,'samples_for_game',prepare)
    monkeypatch.setattr(Dataset,'close',close)
    train=MahjongSharedPolicyTrainSource(build_dir='unused',seed=7,batch_size=8,
        max_sequence_length=128,max_tokens=128,shuffle=False)
    train.configure_prefetch(True)
    lease=train.acquire();train.commit(lease.token)
    assert started.wait(5)
    closer=threading.Thread(target=train.close)
    closer.start()
    assert not closed.is_set()
    release.set();closer.join(5)
    assert not closer.is_alive() and closed.is_set()
