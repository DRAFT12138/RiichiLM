"""Mixed controllers, public visibility, actual rule boundaries and full games."""

from dataclasses import asdict, replace
import json
import random
from types import SimpleNamespace

import pytest

from mahjong.contracts import MahjongError
from mahjong.envs.policy_bridge import RiichiDecisionBridge, SelectedDecision
from mahjong.inference.policy import PolicyInference
from mahjong.tokenizer.policy_context import (
    CONTEXT_SCHEMA,
    LEGACY_SCHEMA,
    PolicyContextInput,
)
from mahjong.web.worker import WebDecisionBridge, TableEngine, profile_for
from tests.unit.inference.bridge_fixtures import wall_for
from tests.unit.inference.test_bridge import rig, play_kind, respond


def bridge(mode="hanchan"):
    result = WebDecisionBridge(profile_for(mode), model_binding_digest="unused")
    result.bind_controllers([("human", None), *[("random", None)] * 3])
    result.reset(seed=42)
    return result


def test_non_model_choices_are_explicit_and_cannot_bypass_model_bindings():
    b = bridge()
    view = b.snapshot(0)
    decision = b.select_legal(view.key, view.legal_moves[0].execution_ref)
    assert isinstance(decision, SelectedDecision) and not hasattr(
        decision, "generation"
    )
    with pytest.raises(MahjongError, match="CONTRACT_MISMATCH"):
        b.apply(replace(decision, action=view.legal_moves[-1].action))
    b.apply(decision)
    with pytest.raises(MahjongError, match="STALE_DECISION"):
        b.select_legal(view.key, decision.execution_ref)
    regular = RiichiDecisionBridge(profile_for("hanchan"), model_binding_digest="model")
    regular.reset(seed=42)
    v = regular.snapshot(0)
    with pytest.raises(MahjongError, match="CONTRACT_MISMATCH"):
        regular.select_legal(v.key, v.legal_moves[0].execution_ref)


def test_each_seat_retains_its_input_schema_and_model_gate():
    models = [
        SimpleNamespace(binding_digest=f"m{i}", input_schema=schema, source=None)
        for i, schema in enumerate((CONTEXT_SCHEMA, LEGACY_SCHEMA, CONTEXT_SCHEMA))
    ]
    b = WebDecisionBridge(profile_for("tonpuu"), model_binding_digest="unused")
    b.bind_controllers(
        [
            ("model", models[0]),
            ("model", models[1]),
            ("model", models[2]),
            ("random", None),
        ]
    )
    b.reset(seed=42)
    v = b.snapshot(0)
    assert isinstance(v.sequence, PolicyContextInput)
    assert v.sequence.header["mode"].name == "MODE_TONPUU"
    with pytest.raises(MahjongError, match="CONTRACT_MISMATCH"):
        b.select_legal(v.key, v.legal_moves[0].execution_ref)
    # Views in a response are projected independently; no environment per seat.
    from tests.unit.inference.test_bridge import select

    # The model binding is genuinely checked, including forced single candidates.
    with pytest.raises(MahjongError, match="CONTRACT_MISMATCH"):
        select(b, v)
    with pytest.raises(MahjongError, match="UNSUPPORTED_PROFILE"):
        RiichiDecisionBridge(profile_for("tonpuu"), model_binding_digest="m")
    changed = profile_for("tonpuu")
    changed["payment_policy"]["honba_ron_each"] = 400
    with pytest.raises(MahjongError, match="UNSUPPORTED_PROFILE"):
        WebDecisionBridge(changed, model_binding_digest="m")


def test_continuous_public_table_excludes_all_other_private_material():
    b = bridge()
    initial = b.public_table(0)
    assert initial["players"][0]["drawn"] is not None
    assert len(initial["players"][0]["hand"]) == 13
    assert all(p["hand"] is None and p["drawn"] is None for p in initial["players"][1:])
    v = b.snapshot(0)
    b.apply(b.select_legal(v.key, v.legal_moves[0].execution_ref))
    snapshot = b.public_table(0)
    assert snapshot["players"][0]["river"] and snapshot["dora_indicators"]
    assert not any(
        key in json.dumps(snapshot)
        for key in ("reserved", "oracle", "tehais", "native_candidates", "wall")
    )
    snapshot["players"][0]["score"] = -999
    assert b.public_table(0)["players"][0]["score"] != -999


def test_public_response_trigger_identifies_discard_source_and_clears_after_pass():
    b = rig({1: ["5p", "5p"]}, drawn="5p")
    assert b.public_table(0)["response_trigger"] is None
    play_kind(b, 0, "A_DISCARD", "5p")
    table = b.public_table(1)
    assert table["response_trigger"] == dict(source=0, tile="5p", kind="discard")
    assert any(c["kind"] == "A_PON" for c in table["candidates"])
    assert all(p["hand"] is None for p in table["players"] if p["seat"] != 1)
    respond(b)
    assert b.public_table(1)["response_trigger"] is None


def test_public_robbed_kan_trigger_does_not_masquerade_as_a_discard():
    kokushi = ["9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "E"]
    b = rig({0: ["1m"] * 4 + ["2p", "3p", "4p", "6p", "7p", "8p", "5s", "6s", "2m"],
             1: kokushi}, drawn="9s")
    play_kind(b, 0, "A_ANKAN")
    table = b.public_table(1)
    assert table["response_trigger"] == dict(source=0, tile="1m", kind="kan")
    assert any(c["kind"] == "A_RON" for c in table["candidates"])
    assert table["players"][0]["river"] == []


@pytest.mark.parametrize("mode", ["single", "tonpuu", "hanchan"])
def test_full_human_plus_random_games_are_playable_without_any_model_or_archive(
    mode, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    latest = []
    t = TableEngine(
        dict(mode=mode, opponents=[None] * 3, seed=42),
        emit=lambda value: latest.__setitem__(slice(None), [value]),
    )
    t.drive()
    rng = random.Random(7)
    rounds = set()
    for step in range(1500):
        snapshot = latest[0]["table"]
        rounds.add(snapshot["round_id"])
        if t.status == "finished":
            break
        if t.status == "round_result":
            t.continue_round(snapshot["round_id"], snapshot["revision"])
        else:
            assert snapshot["window"] is not None
            t.action(snapshot["window"], rng.choice(snapshot["candidates"])["id"])
    assert t.status == "finished"
    assert not list(tmp_path.iterdir())
    assert len(rounds) == 1 if mode == "single" else len(rounds) >= 4
    assert ("final" in latest[0]["table"]) == (mode != "single")


@pytest.mark.parametrize(
    "kind, hand, drawn",
    [
        (
            "A_TSUMO_WIN",
            [
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "2p",
                "3p",
                "4p",
                "6s",
                "7s",
                "8s",
                "E",
            ],
            "E",
        ),
        (
            "A_ABORT_KYUUSHU",
            ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "2m"],
            "3m",
        ),
    ],
)
def test_single_stops_on_first_result_without_finalizing_or_distributing_pool(
    kind, hand, drawn
):
    t = TableEngine(dict(mode="single", opponents=[None] * 3, seed=42))
    t.bridge._reset(wall=wall_for({0: hand}, drawn=drawn), oya=0, seed=123)
    view = t.bridge.snapshot(0)
    move = next(m for m in view.legal_moves if m.action.kind == kind)
    t.bridge.apply(t.bridge.select_legal(view.key, move.execution_ref))
    t.drive()
    assert t.status == "finished" and t.bridge._round == 0
    assert not t.bridge._session.finalized
    result = t.bridge.round_result()
    assert result["kind"] == ("win" if kind == "A_TSUMO_WIN" else "abort")
    assert all(p["hand"] is None for p in t.bridge.public_table(0)["players"][1:])


def test_score_display_comes_from_selected_interpretation_and_not_native_log():
    b = rig(
        {
            0: [
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "2p",
                "3p",
                "4p",
                "6s",
                "7s",
                "8s",
                "E",
            ]
        },
        drawn="E",
    )
    play_kind(b, 0, "A_TSUMO_WIN")
    public = b.round_result()
    winner = public["winners"][0]
    assert winner["yaku"]
    assert winner["yakuman_units"] > 0 or (winner["han"] >= 1 and winner["fu"] >= 20)
    assert winner["winning_tile"] == "E"
    assert len(winner["hand"]) == 13
    assert public["score_delta"][0] > 0
    assert sum(public["scores"]) + public["kyotaku"] * 1000 == 100000


@pytest.mark.parametrize(
    "position,scores,result,over,next_position",
    [
        (4, [25000] * 4, dict(kind="win", winners=[1]), False, 5),
        (4, [30000, 25000, 25000, 20000], dict(kind="win", winners=[0]), True, None),
        (4, [25000] * 4, dict(kind="abort"), False, 4),
        (8, [25000] * 4, dict(kind="win", winners=[1]), True, None),
        (1, [-100, 40100, 30000, 30000], dict(kind="win", winners=[1]), True, None),
    ],
)
def test_tonpuu_uses_actual_profile_boundary_table(
    position, scores, result, over, next_position
):
    from mahjong.envs.rule_profile import offline_options_from_profile
    from mahjong.envs.round_rules import RoundRules

    profile = profile_for("tonpuu")
    rules = RoundRules.from_dict(
        offline_options_from_profile(
            profile,
            dict(revision="mjai/recorded-deltas-1", settlement_deltas="per_event"),
        ).round_rules
    )
    header = dict(
        bakaze="ES"[(position - 1) // 4], kyoku=(position - 1) % 4 + 1, oya=0, honba=2
    )
    decision = rules.decide(header, scores, 2, result, starting_dealer=0)
    assert len(decision.outcomes) == 1
    assert (
        decision.game_over is over
        and decision.outcomes[0].next_position == next_position
    )


def test_ordinary_riichi_ura_and_multiron_result_material_is_separate():
    from tests.unit.inference.test_bridge import respond
    from riichienv import Phase

    b = rig(
        {
            0: [
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "2p",
                "3p",
                "4p",
                "6s",
                "7s",
                "8s",
                "E",
            ]
        },
        drawn="9p",
        draws=["1s", "2s", "3s", "E"],
    )
    play_kind(b, 0, "A_RIICHI_DISCARD", "9p")
    for p, tile in [(1, "1s"), (2, "2s"), (3, "3s")]:
        if b._native.phase == Phase.WaitResponse:
            respond(b)
        play_kind(b, p, "A_DISCARD", tile)
    if b._native.phase == Phase.WaitResponse:
        respond(b)
    play_kind(b, 0, "A_TSUMO_WIN")
    w = b.round_result()["winners"][0]
    assert w["han"] is not None and {"Y_RIICHI", "Y_DOUBLE_RIICHI"}.intersection(
        w["yaku"]
    )
    assert len(w["ura_indicators"]) == len(b.public_table(0)["dora_indicators"])
    b = rig(
        {
            1: [
                "P",
                "P",
                "P",
                "1m",
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "2p",
                "3p",
                "E",
                "E",
            ],
            2: [
                "F",
                "F",
                "F",
                "1s",
                "2s",
                "3s",
                "4s",
                "5s",
                "6s",
                "2p",
                "3p",
                "S",
                "S",
            ],
        },
        drawn="4p",
    )
    play_kind(b, 0, "A_DISCARD", "4p")
    respond(b, {1: "A_RON", 2: "A_RON"})
    winners = b.round_result()["winners"]
    assert {w["actor"] for w in winners} == {1, 2}
    assert all(w["target"] == 0 and w["han"] >= 1 for w in winners)
    assert winners[0]["hand"] != winners[1]["hand"]


def test_multiple_scoring_interpretations_display_exact_selected_value():
    from tests.unit.inference.test_bridge import respond

    b = rig(
        {
            1: [
                "1m",
                "1m",
                "2m",
                "2m",
                "3m",
                "3m",
                "4m",
                "4m",
                "5m",
                "5m",
                "6m",
                "6m",
                "7m",
            ]
        },
        drawn="7m",
    )
    play_kind(b, 0, "A_DISCARD", "7m")
    material = b._session.result_material(1, "7m", False)
    assert len(material["native_candidates"]) > 1
    respond(b, {1: "A_RON"})
    w = b.round_result()["winners"][0]
    assert "Y_RYANPEIKOU" in w["yaku"] and "Y_CHIITOITSU" not in w["yaku"]
    assert w["han"] >= 9 and w["fu"] == 30
