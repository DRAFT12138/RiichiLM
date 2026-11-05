"""Four-seat playback, atomic explanations, viewer isolation and real IPC control."""

from dataclasses import asdict
from functools import partial
import json
import multiprocessing
import time
import uuid

import pytest

from mahjong.arena.llm import ChatCompletionClient, Completion, LLMConfig
from mahjong.contracts import MahjongError
from mahjong.web.session import Session
from mahjong.web.worker import TableEngine
from tests.unit.web.test_session import wait_until


def table(mode="single", opponents=None, emit=lambda _: None):
    return TableEngine(dict(mode=mode, opponents=[None] * 4 if opponents is None else opponents,
                            spectator=True, seed=42), emit=emit)


def test_pause_step_view_and_speed_never_expose_actions_or_other_hands():
    engine = table()
    engine.control("pause")
    first = engine.publish()
    engine.drive()
    assert engine.publish()["revision"] == first["revision"]
    engine.control("step")
    snapshot = engine.publish()
    assert snapshot["revision"] > first["revision"] and snapshot["paused"]
    assert snapshot["candidates"] == [] and snapshot["window"] is None
    assert snapshot["decision_notes"] and snapshot["decision_notes"][0]["sampling"] == "uniform"
    for seat in range(4):
        engine.control("view", seat)
        snapshot = engine.publish()
        assert snapshot["players"][seat]["hand"] is not None
        assert all(p["hand"] is None and p["drawn"] is None for p in snapshot["players"] if p["seat"] != seat)
        assert all(n["seat"] == seat for n in snapshot["decision_notes"])
    engine.control("speed", 0.25)
    assert engine.playback_interval == 0.25
    with pytest.raises(MahjongError):
        engine.action({}, "0")
    for control, value in [("view", True), ("view", 4), ("speed", float("nan")), ("speed", 10), ("invalid", None)]:
        with pytest.raises(MahjongError):
            engine.control(control, value)


@pytest.mark.parametrize("mode", ["single", "tonpuu", "hanchan"])
def test_four_random_ais_complete_game_without_any_human_input_or_storage(mode, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = table(mode)
    rounds = set()
    for _ in range(5000):
        snapshot = engine.publish()
        rounds.add(snapshot["round_id"])
        assert not snapshot["candidates"] and engine.waiting is None
        if engine.status == "finished":
            break
        if engine.status == "round_result":
            engine.continue_round(snapshot["round_id"], snapshot["revision"])
        else:
            engine.drive()
    assert engine.status == "finished"
    assert len(rounds) == 1 if mode == "single" else len(rounds) >= 4
    assert len(engine.decision_notes) <= 64
    assert not list(tmp_path.iterdir())


def test_llm_summaries_only_publish_after_the_whole_window_commits(monkeypatch):
    def complete(self, messages, **kwargs):
        assert "brief decision summary" in messages[0]["content"]
        prompt = json.loads(messages[1]["content"])
        return Completion(json.dumps(dict(candidate_id=prompt["legal_actions"][0]["candidate_id"], reason="保留手牌的灵活性。")))
    monkeypatch.setattr(ChatCompletionClient, "complete", complete)
    config = LLMConfig(id="test", name="Test", model="test", base_url="http://localhost/v1")
    snapshots = []
    engine = table(opponents=[dict(kind="llm", config=asdict(config))] * 4,
                   emit=lambda message: snapshots.append(message["table"]))
    for _ in range(16):
        engine.drive()
        if engine.status != "playing":
            break
    assert any(s["thinking"] is not None for s in snapshots)
    assert any(s["decision_notes"] for s in snapshots)
    for snapshot in snapshots:
        for note in snapshot["decision_notes"]:
            assert note["revision"] < snapshot["revision"]
            assert note["summary"] == "保留手牌的灵活性。"
            assert note["seat"] == snapshot["viewer"]
        assert "base_url" not in json.dumps(snapshot)
    # Explanations remain a spectator feature; human opponents' private rationale
    # is never returned in the ordinary game snapshot.
    human = TableEngine(dict(mode="single", opponents=[None] * 3, seed=42))
    human.decision_notes.extend(engine.decision_notes)
    assert human.publish()["decision_notes"] == []
    with pytest.raises(MahjongError):
        human.control("view", 1)


def test_real_spawn_paces_pauses_steps_and_rejects_duplicate_control_mutations(tmp_path):
    session = Session(tmp_path, lease_seconds=90)
    page = uuid.uuid4().hex
    try:
        session.init_page(page, session.generation)
        session.create(page, "spectator-create", "single", ["random"] * 4, True)
        wait_until(lambda: session.snapshot(page)["state"] != "loading", timeout=30)
        current = session.snapshot(page)
        assert current["spectator"] and current["state"] == "playing"
        def control(name, value=None, request=None):
            return session.submit(page, current["table_id"], request or uuid.uuid4().hex,
                                  "control", control=name, value=value)
        control("pause")
        wait_until(lambda: session.snapshot(page).get("paused") and not session.snapshot(page)["busy"])
        revision = session.snapshot(page)["revision"]
        time.sleep(0.2)
        assert session.snapshot(page)["revision"] == revision and session.deadline is None
        control("view", 2)
        wait_until(lambda: session.snapshot(page).get("viewer") == 2 and not session.snapshot(page)["busy"])
        assert session.snapshot(page)["players"][2]["hand"] is not None
        control("step", request="single-step-request")
        wait_until(lambda: session.snapshot(page)["revision"] > revision and not session.snapshot(page)["busy"])
        stepped = session.snapshot(page)["revision"]
        assert control("step", request="single-step-request")["status"] == "done"
        assert session.snapshot(page)["revision"] == stepped
        control("resume")
        wait_until(lambda: session.snapshot(page)["revision"] > stepped)
        with pytest.raises(MahjongError):
            session.submit(page, current["table_id"], "observer-action", "action", window=None, candidate="0")
    finally:
        session.close()


def blocked_inference_worker(connection, config, device, decision_timeout, *, release):
    """Hold a real decision until the parent explicitly releases the model."""
    try:
        engine = TableEngine(config, emit=connection.send, decision_timeout=decision_timeout)
        engine.loop.on_thinking = lambda seat: engine.publish(thinking=seat)
        choose = engine.loop.players[0].choose

        def blocked_choose(view, **kwargs):
            if not release.wait(30):
                raise RuntimeError("test inference was not released")
            return choose(view, **kwargs)

        engine.loop.players[0].choose = blocked_choose
        engine.drive()
        while True:
            message = connection.recv()
            engine.control(message["control"], message.get("value"))
            connection.send(dict(type="ack", request_id=message["request_id"]))
    except EOFError:
        pass
    finally:
        connection.close()


def test_view_switches_during_blocked_inference_and_pending_pause(tmp_path):
    release = multiprocessing.get_context("spawn").Event()
    session = Session(tmp_path, worker=partial(blocked_inference_worker, release=release),
                      lease_seconds=90, decision_timeout=30)
    page = uuid.uuid4().hex
    try:
        session.init_page(page, session.generation)
        created = session.create(page, "blocked-create", "single", ["random"] * 4, True)
        wait_until(lambda: session.snapshot(page).get("thinking") == 0)
        first = session.snapshot(page)
        deadline = session.deadline
        session.submit(page, created["table_id"], "pending-pause", "control", control="pause")
        for seat in (2, 1, 3, 0, 2):
            request_id = uuid.uuid4().hex
            result = session.submit(page, created["table_id"], request_id, "control", control="view", value=seat)
            current = session.snapshot(page)
            assert result["status"] == "done" and current["viewer"] == seat
            assert current["busy"] and session.pending == "pending-pause"
            assert current["revision"] == first["revision"] and current["thinking"] == 0
            assert session.deadline == deadline
            assert current["players"][seat]["hand"] is not None
            assert all(p["hand"] is None and p["drawn"] is None for p in current["players"] if p["seat"] != seat)
            assert current["candidates"] == [] and current["window"] is None
            assert current["decision_notes"] == [] and "spectator_views" not in current
            assert session.submit(page, created["table_id"], request_id, "control", control="view", value=seat) == result
            assert session.snapshot(page)["view_version"] == current["view_version"]
        with pytest.raises(MahjongError):
            session.submit(page, created["table_id"], "invalid-seat", "control", control="view", value=True)
        release.set()
        wait_until(lambda: not session.snapshot(page)["busy"])
        current = session.snapshot(page)
        assert current["viewer"] == 2 and current["paused"]
        assert current["revision"] > first["revision"] and current["thinking"] is None
        assert current["players"][2]["hand"] is not None
        assert all(n["seat"] == 2 for n in current["decision_notes"])
        assert session.deadline is None
    finally:
        release.set()
        session.close()
