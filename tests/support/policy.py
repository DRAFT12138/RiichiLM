"""Hand-written policy inputs for public tests; no source parsers or game logs."""
from collections import Counter
from compression import zstd
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

from mahjong.tokenizer import Action, Atom, Event, HandMeldState, MahjongTokenizer
from mahjong.envs.round_types import FrozenAnchor, PreparedRound, PreparedView
from mahjong.data.mahjong.cleaning.catalog import codec_binding
from mahjong.data.mahjong.cleaning.common import atomic_json, canonical, digest, file_identity, read_json
from mahjong.data.mahjong.cleaning.text_codec import encode_round
from mahjong.data.mahjong.cleaning.encoding import encode_view

SEATS = ('SELF', 'SHIMOCHA', 'TOIMEN', 'KAMICHA')


def fixture_round():
    hand = [Atom('TILE_' + t) for t in ('1m','2m','3m','4m','5mr','6m','7m','8m','9m','E','S','W','N')]
    a = HandMeldState([*hand, Atom('TILE_1s')], [])
    b = HandMeldState(hand, [])
    kinds = ['EV_ROUND_START', 'EV_DORA_REVEAL', 'EV_DRAW', 'EV_DISCARD',
             'EV_RESPONSE_OPEN', 'EV_RESPONSE_CLOSE', 'EV_DRAW', 'EV_DISCARD',
             'EV_RESPONSE_OPEN', 'EV_RESPONSE_CLOSE', 'EV_DRAW']
    ledger = [dict(event_id=i, kind=k, source_refs=[dict(raw_line=i + 1)], provenance='recorded')
              for i, k in enumerate(kinds)]
    for index,actor in ((2,0),(3,0),(6,1),(7,1),(10,0)):
        ledger[index]['actor']=actor
    windows = [dict(window_id=8, trigger_id=2, cutoff_event_id=2, viewers=[0]),
               dict(window_id=14, trigger_id=3, cutoff_event_id=4, viewers=[1,2,3]),
               dict(window_id=30, trigger_id=7, cutoff_event_id=8, viewers=[2,3,0]),
               dict(window_id=40, trigger_id=10, cutoff_event_id=10, viewers=[0])]
    histories = []
    views = []
    for viewer in range(4):
        def seat(actor):
            return Atom(SEATS[(actor - viewer) % 4])
        def draw(actor, tile):
            return (seat(actor), Atom('TILE_' + tile) if viewer == actor else Atom('TILE_HIDDEN'), Atom('DRAW_LIVE'))
        history = [Event('EV_DORA_REVEAL',1,(Atom('TILE_P'),)),
                   Event('EV_DRAW',2,draw(0,'1s')),
                   Event('EV_DISCARD',3,(seat(0),Atom('TILE_1s'),Atom('TSUMOGIRI'),False)),
                   Event('EV_DRAW',6,draw(1,'2p')),
                   Event('EV_DISCARD',7,(seat(1),Atom('TILE_2p'),Atom('TSUMOGIRI'),False)),
                   Event('EV_DRAW',10,draw(0,'1s'))]
        header = dict(mode=Atom('MODE_HANCHAN'), round_wind=Atom('ROUND_E'), round_number=1,
                      honba=0, kyotaku=0, dealer=seat(0), scores=[25000]*4,
                      seat_winds=[Atom('TILE_' + 'ESWN'[(viewer+i)%4]) for i in range(4)])
        anchors = [FrozenAnchor.capture(w['window_id'],w['trigger_id'],w['cutoff_event_id'],
                                       a if w['trigger_id'] in (2,10) else b)
                   for w in windows if viewer in w['viewers']]
        # Other players have independently specified 13 tiles at response time.
        if viewer != 0:
            anchors = [replace(x, state=deepcopy(b)) for x in anchors]
        views.append(PreparedView(viewer,header,history,anchors,10))
    return PreparedRound(0,[1,2,3,6,7,10],ledger,windows,views,False)



def manual_build(root, shard):
    """Two games with known discard alternatives, five rounds each."""
    root.mkdir(parents=True);directory=root/'shards/part-000000';directory.mkdir(parents=True)
    codec=MahjongTokenizer();prepared=fixture_round()
    views=[asdict(encode_view(v,prepared)) for v in prepared.views]
    candidates=[codec.encode_action_suffix(Action('A_DISCARD',(Atom('TILE_1s'),Atom('TSUMOGIRI')))),
                codec.encode_action_suffix(Action('A_DISCARD',(Atom('TILE_1m'),Atom('TEDASHI'))))]
    decision=dict(window_id=8,trigger_id=2,viewer=0,response=False,candidates=candidates,target_index=0,target_event_id=3)
    games=[];rounds=[];parts=[];total=Counter();members=[];offset=0
    for game in range(2):
        key=hashlib.sha256(f'{shard}/{game}'.encode()).hexdigest();members.append(key)
        start=offset;game_rounds=[]
        for rid in range(5):
            record=dict(source_key=key,round_id=rid,event_ids=prepared.event_ids,views=views,
                facts=dict(source_refs=[[w['window_id'],[]] for w in prepared.windows],outcomes=[],policy_decisions=[decision]))
            raw=encode_round(record);parts.append(raw)
            game_rounds.append(dict(source_key=key,round_id=rid,offset=offset-start,bytes=len(raw),
                views=[[v['viewer'],v['valid_through_event_id'],len(v['arrays']['anchors'])//5] for v in views]))
            offset+=len(raw)
        counts=dict(policy_samples=5,policy_candidate_sets=5,policy_target_matches=5,
                    policy_target_misses=0,policy_target_unresolved=0)
        games.append(dict(source_key=key,game_id=f'majsoul:artificial:{shard}:game{game}.mjai.json',native_id=None,
            rule_ref=dict(ruleset_id='fixture'),status='success',counts=counts,offset=0,bytes=0,
            raw_bytes=offset-start,frame_game_offset=start,frame_raw_bytes=0))
        rounds.extend(game_rounds);total.update(counts)
    raw=b''.join(parts);compressed=zstd.compress(raw,level=1)
    for game in games:game.update(bytes=len(compressed),frame_raw_bytes=len(raw))
    (directory/'data.blocks.zst').write_bytes(compressed)
    old_id=digest(shard)
    index=dict(binding=dict(storage='mjai-clean-text/3',build_id=old_id,task_id=0,members=members),games=games,rounds=rounds)
    (directory/'index.json.zst').write_bytes(zstd.compress(canonical(index),level=1))
    total.update(games=2,rounds=10,views=40,anchors=5*sum(len(v['arrays']['anchors'])//5 for v in views)*2,
                 tokens=5*sum(sum(len(v['arrays'][k]) for k in ('header','history','hand_blocks')) for v in views)*2)
    summary=dict(counts=dict(total),groups={f'majsoul/{shard}':{}},statuses=dict(success=2,partial=0,failed=0,skipped=0))
    atomic_json(root/'summary.json',summary)
    profile=read_json(Path('configs/inference-profile.json'))
    binding=dict(storage='mjai-clean-text/3',compression='zstd-blocks/1',backend='rust',codec=codec_binding(),
        input_digest=digest(members),implementation={'manual_fixture':'1'},rules={profile['ruleset_id']:{'profile':profile}})
    products=[dict(name=name,bytes=(identity:=file_identity(directory/name))[0],sha256=identity[1])
              for name in ('data.blocks.zst','index.json.zst')]
    manifest=dict(build_id=old_id,binding=binding,summary_sha256=file_identity(root/'summary.json')[1],
        capabilities=dict(legal_actions=True,action_labels=True),
        shards=[dict(task_id=0,products=products,**{k:total[k] for k in ('games','rounds','views','anchors','tokens')})])
    atomic_json(root/'manifest.json',manifest)
    atomic_json(root/'build.json',dict(sources=[dict(revision='artificial')]))
    return root
