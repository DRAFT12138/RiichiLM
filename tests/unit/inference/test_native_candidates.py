"""The corrected Rust producer must agree with true native physical copies."""
from dataclasses import asdict
import hashlib
import json

import pytest

from mahjong.data.mahjong.cleaning.catalog import codec_binding
from mahjong.data.mahjong.cleaning.common import canonical
from mahjong.data.mahjong.cleaning.source import REQUIRED, OPTIONAL
from mahjong.data.mahjong.cleaning.text_codec import decode_round
from mahjong.tokenizer import Atom, MahjongTokenizer
from tests.unit.inference.test_bridge import rig, select


def test_rust_candidates_keep_both_origins_and_match_actual_native_set():
    module=pytest.importorskip('mahjong_cleaning_native')
    b=rig({0:['5m','5m']},drawn='5m')
    view=b.snapshot(0)
    codec=MahjongTokenizer()
    expected={tuple(codec.encode_action_suffix(m.action)) for m in view.legal_moves}
    chosen=next(m for m in view.legal_moves if m.action.kind=='A_DISCARD' and
                m.action.args==(Atom('TILE_5m'),Atom('TEDASHI')))
    b.apply(select(b,view,chosen))
    events=[]
    for event in b._native.mjai_log:
        allowed=set(REQUIRED[event['type']]) | set(OPTIONAL.get(event['type'],())) | {'type'}
        events.append({k:v for k,v in event.items() if k in allowed})
    # A valid JSON source with an intentionally impossible early draw result
    # retains the preceding decision as a partial; no production source is used.
    events += [dict(type='ryukyoku',deltas=[0]*4),dict(type='end_kyoku'),dict(type='end_game')]
    payload=b''.join(canonical(e)+b'\n' for e in events)
    config=dict(source=dict(platform='majsoul',revision='artificial',shard='inference',
        relative_path='duplicate.mjai.json',raw_bytes=len(payload),raw_sha256=hashlib.sha256(payload).hexdigest(),
        source_key='inference-duplicate'),codec=codec_binding(),offline_options=asdict(b._options),
        rule_ref=dict(ruleset_id=b._profile['ruleset_id'],ruleset_revision=b._profile['revision'],
            ruleset_hash=view.rules_digest,profile_index=0))
    meta,body=module.process_game_bytes(payload,canonical(config).decode())
    record=decode_round(body.split(b'Z\n')[0]+b'Z\n')
    decision=next(d for d in record['facts']['policy_decisions'] if d['window_id']==view.key.window.window_id and d['viewer']==0)
    assert {tuple(s) for s in decision['candidates']}==expected
    assert tuple(decision['candidates'][decision['target_index']])==tuple(codec.encode_action_suffix(chosen.action))
    assert json.loads(meta)['counts']['policy_target_misses']==0
