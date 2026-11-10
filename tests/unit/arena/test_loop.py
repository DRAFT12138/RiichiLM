from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from mahjong.arena.bridge import ArenaDecisionBridge, profile_for
from mahjong.arena.evaluation import evaluate_cli, run_match, model_summaries
from mahjong.arena.llm import Completion, LLMConfig, LLMPlayer, decision_prompt
from mahjong.arena.loop import DecisionLoop
from mahjong.arena.players import PlayerChoice, RandomPlayer
from mahjong.contracts import MahjongError
from mahjong.tokenizer import Atom
from mahjong.tokenizer.policy_context import PolicyContextInput
from tests.unit.inference.bridge_fixtures import wall_for
from tests.unit.inference.test_weights import weights


class ChoosingClient:
    def __init__(self):
        self.prompts = []

    def complete(self, messages, **kwargs):
        prompt = json.loads(messages[1]["content"])
        self.prompts.append(prompt)
        actions = prompt["legal_actions"]
        move = next((m for m in actions if m["kind"] in ("A_TSUMO_WIN", "A_RON")), actions[0])
        return Completion(json.dumps(dict(candidate_id=move["candidate_id"])), 25, 4)


def llm():
    client = ChoosingClient()
    return LLMPlayer(LLMConfig(id="test", name="Test", model="fake", base_url="http://localhost/v1"), client=client)


def bridge(kinds):
    b = ArenaDecisionBridge(profile_for("tonpuu"), model_binding_digest="unused")
    b.bind_controllers([(kind, None) for kind in kinds])
    return b


def test_all_llm_headless_match_preserves_private_projection_and_previous_rounds():
    players = [llm() for _ in range(4)]
    b = bridge(["llm"] * 4)
    result = run_match(b, players, seed=42, max_windows=4000)
    assert result["status"] == "complete", result
    assert result["rounds"] >= 4 and sorted(result["ranks"]) == [1, 2, 3, 4]
    assert sum(result["scores"]) == 100000
    assert all(p.stats["decisions"] > 0 for p in players)
    assert all(len(p.client.prompts[0]["observation"]["hand_state"]["hands"]) >= 13 for p in players)
    assert any(p["observation"]["previous_rounds"] for p in players[0].client.prompts)
    for player in players:
        for prompt in player.client.prompts:
            for event in prompt["observation"]["visible_history"]:
                if event["kind"] == "EV_DRAW" and event["payload"][0] != "SELF":
                    assert event["payload"][1] == "TILE_HIDDEN"
            assert not any(k in json.dumps(prompt) for k in ("reserved", "oracle", "tehais", "native_candidates"))


def test_current_hand_and_other_players_draws_are_seat_specific():
    b = bridge(["llm"] * 4)
    b.reset(seed=42)
    v = b.decision_views()[0]
    assert isinstance(v.sequence, PolicyContextInput)
    prompt = decision_prompt(v)
    own_draws = [e.payload[1] for e in v.sequence.visible_history if e.kind == "EV_DRAW" and e.payload[0] == Atom("SELF")]
    assert own_draws and own_draws[-1] != Atom("TILE_HIDDEN")
    assert prompt["observation"]["hand_state"]["hands"] == [a.name for a in v.sequence.hand_state.hands]
    assert set(prompt["observation"]) == {"header", "visible_history", "hand_state", "previous_rounds"}


def test_invalid_and_late_choices_cannot_be_applied():
    b = bridge(["llm"] * 4)
    b.reset(seed=42)
    v = b.decision_views()[0]
    class BadPlayer:
        def __init__(self, key, ref):
            self.key, self.ref = key, ref
        def choose(self, view, **kwargs):
            return PlayerChoice(self.key, self.ref)
    loop = DecisionLoop(b, [BadPlayer(replace(v.key, viewer=1), "0")] * 4)
    with pytest.raises(MahjongError, match="CONTRACT_MISMATCH"):
        loop.select(v)
    loop.players = tuple([BadPlayer(v.key, "nonexistent")] * 4)
    with pytest.raises(MahjongError, match="CONTRACT_MISMATCH"):
        loop.select(v)
    selected = b.select_legal(v.key, v.legal_moves[0].execution_ref)
    b.apply(selected)
    with pytest.raises(MahjongError, match="STALE_DECISION"):
        b.apply(selected)


def test_response_choices_are_all_frozen_before_any_apply_and_forced_pass_is_skipped():
    b = bridge(["llm"] * 4)
    b._reset(wall=wall_for({1: ["5p", "5p"]}, drawn="5p"), seed=42)
    v = b.snapshot(0)
    move = next(m for m in v.legal_moves if m.action.kind == "A_DISCARD" and m.action.args[0] == Atom("TILE_5p"))
    b.apply(b.select_legal(v.key, move.execution_ref))
    revision = b.public_table(0)["revision"]
    seen = []
    class SeatPlayer:
        def choose(self, view, **kwargs):
            assert b.public_table(0)["revision"] == revision
            seen.append(view.key)
            move = next((m for m in view.legal_moves if m.action.kind == "A_PASS"), view.legal_moves[0])
            return PlayerChoice(view.key, move.execution_ref)
    loop = DecisionLoop(b, [SeatPlayer()] * 4)
    pending = loop.collect()
    assert pending.response and seen
    assert all(key.window == seen[0].window for key in seen)
    assert all(len(v.legal_moves) > 1 for v in pending.views if v.key in seen)
    assert b.public_table(0)["revision"] == revision
    loop.apply(pending)
    assert b.public_table(0)["revision"] > revision


def test_failed_or_cancelled_evaluation_has_no_scores_or_silent_random_replacement():
    class FailingPlayer:
        def choose(self, view, **kwargs):
            raise MahjongError("LLM_DECISION_FAILED", "test failure", 'inference', dict(attempts=2))
    result = run_match(bridge(["llm"] * 4), [FailingPlayer()] * 4, seed=42)
    assert result["status"] == "failed" and result["scores"] is None
    assert result["error"]["code"] == "LLM_DECISION_FAILED"
    assert result['error']['details'] == dict(attempts=2)
    assert result['error']['phase'] == 'inference'
    assert 'test failure' in result['error']['traceback']
    assert 'MahjongError' in result['error']['traceback']
    result = run_match(bridge(["random"] * 4), [RandomPlayer(i) for i in range(4)], seed=42, cancel=lambda: True)
    assert result["status"] == "cancelled" and result["scores"] is None
    result = run_match(bridge(["random"] * 4), [RandomPlayer(i) for i in range(4)], seed=42, max_windows=1)
    assert result["status"] == "failed" and result["windows"] == 1


def test_cli_fixed_seed_rotations_report_groups_and_do_not_overwrite(tmp_path):
    output = tmp_path / "arena.json"
    args = SimpleNamespace(players=["random"] * 4, mode="single", device="cpu", llm_config=None,
                           seeds=[42], rotate=True, decision_timeout=10, max_windows=1000, output=str(output))
    persisted = []
    def persist_match(outcome, identities):
        assert not output.exists()  # Callback runs before the final batch report.
        assert outcome['status'] == 'complete' and len(identities) == 4
        persisted.append(outcome['rotation'])
        # The consumer cannot mutate the evaluation's stored match or identities.
        outcome['status'] = 'failed'
        identities[0]['kind'] = 'changed'
    assert evaluate_cli(args, on_match=persist_match) == 0
    assert persisted == [0, 1, 2, 3]
    report = json.loads(output.read_text())
    assert report["completed_seed_groups"] == 1 and len(report["outcomes"]) == 4
    assert report["failed_matches"] == 0 and report["strength_claim"] is False
    assert report["seed_groups"][0]["ranks"] is None
    assert {tuple(o["player_indices"]) for o in report["outcomes"]} == {(0,1,2,3), (3,0,1,2), (2,3,0,1), (1,2,3,0)}
    original = output.read_bytes()
    with pytest.raises(MahjongError, match="已存在"):
        evaluate_cli(args)
    assert output.read_bytes() == original


def test_incomplete_seed_groups_remain_reported_but_do_not_enter_score_summary(tmp_path, capsys):
    args = SimpleNamespace(players=["random"] * 4, mode="single", device="cpu", llm_config=None,
                           seeds=[42], rotate=True, decision_timeout=10, max_windows=1, output=str(tmp_path / "failed.json"))
    assert evaluate_cli(args) == 1
    report = json.loads((tmp_path / "failed.json").read_text())
    assert report["completed_seed_groups"] == 0 and report["seed_groups"] == []
    assert report["failed_matches"] == 4 and len(report["outcomes"]) == 4
    assert '对局超过窗口预算' in capsys.readouterr().out


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='GPU unavailable'))])
def test_cli_weights_share_web_execution_and_deduplicate_paths(weights, tmp_path, monkeypatch, device):
    if device == 'cuda:0' and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 GPU unavailable')
    from mahjong.arena import evaluation
    from mahjong.policy.execution import policy_precision
    path, _, _ = weights
    path.with_suffix('.json').write_text(json.dumps({'attn_implementation': 'eager'}))
    original = evaluation.run_match
    seen = []

    def checked_match(bridge, players, **kwargs):
        a, b = players[:2]
        assert a.policy is b.policy
        assert a.memory_budget is b.memory_budget
        assert a.prefix_cache is not b.prefix_cache
        assert a.generator is not b.generator
        assert a.generator.initial_seed() != b.generator.initial_seed()
        assert policy_precision(a.policy.model) == ('bf16-mixed' if device == 'cuda:0' else '32-true')
        assert a.policy.lm.config._attn_implementation == ('sdpa' if device == 'cuda:0' else 'eager')
        seen.append(a.policy)
        return original(bridge, players, **kwargs)

    monkeypatch.setattr(evaluation, 'run_match', checked_match)
    output = tmp_path / 'custom-arena.json'
    args = SimpleNamespace(players=[f'weights:{path}', f'weights:{path.parent}/./{path.name}',
                                   'random', 'random'], mode='single', device=device,
                           llm_config=None, seeds=[42, 43], rotate=False,
                           decision_timeout=30, max_windows=1000, output=str(output))
    assert evaluate_cli(args) == 0
    assert len(seen) == 2 and seen[0] is seen[1]
    report = json.loads(output.read_text())
    assert report['failed_matches'] == 0 and report['completed_seed_groups'] == 2
    assert report['gpu_memory']['requested_gib'] == 16
    assert all(sum(c['hits'] for c in match['prefix_cache'] if c is not None) > 0
               for match in report['outcomes'])
    assert report['players'][0] == report['players'][1]
    assert report['players'][0]['source_kind'] == 'weights'
    summary, = report['model_summaries']
    assert summary['appearances'] == 4 and summary['player_indices'] == [0, 1]
    total = sum(sum(match['scores'][:2]) for match in report['outcomes'])
    assert summary['total_score'] == total
    assert summary['net_points'] == total - 100000
    assert summary['mean_score'] == total / 4
    assert summary['mean_rank'] is None


def test_model_summary_counts_repeated_seats_rotations_and_only_complete_groups():
    a, b, c = [dict(kind='model', checkpoint_id=name) for name in ('A', 'B', 'C')]
    identities = [a, a, b, c]
    outcomes = [
        dict(seed=42, status='complete', player_indices=[0, 1, 2, 3],
             scores=[30000, 20000, 40000, 10000], ranks=[2, 3, 1, 4]),
        dict(seed=43, status='complete', player_indices=[2, 0, 3, 1],
             scores=[27000, 23000, 15000, 35000], ranks=[2, 3, 4, 1]),
        dict(seed=44, status='complete', player_indices=[0, 1, 2, 3],
             scores=[100000, 0, 0, 0], ranks=[1, 2, 3, 4]),
    ]
    result = model_summaries(identities, outcomes, [dict(seed=42), dict(seed=43)], [25000]*4)
    assert [row['appearances'] for row in result] == [4, 2, 2]
    assert [row['total_score'] for row in result] == [108000, 67000, 25000]
    assert [row['net_points'] for row in result] == [8000, 17000, -25000]
    assert [row['mean_score'] for row in result] == [27000, 33500, 12500]
    assert [row['mean_rank'] for row in result] == [2.25, 1.5, 4]
    assert sum(row['net_points'] for row in result) == 0
    empty = model_summaries(identities, outcomes, [], [25000]*4)
    assert all(row['appearances'] == 0 and row['mean_score'] is None for row in empty)
