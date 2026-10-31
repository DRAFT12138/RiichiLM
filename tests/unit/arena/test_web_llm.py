from dataclasses import asdict
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from mahjong.arena.llm import LLMConfig
from mahjong.contracts import MahjongError
from mahjong.web.server import create_app
from mahjong.web.session import Session
from mahjong.web.worker import TableEngine
from tests.unit.web.test_session import wait_until


def test_three_real_http_llm_opponents_complete_a_web_single_game(llm_server):
    config = LLMConfig(id="test", name="LLM 对手", model="test", base_url=llm_server.url)
    snapshots = []
    engine = TableEngine(dict(mode="single", opponents=[dict(kind="llm", config=asdict(config))] * 3, seed=42),
                         emit=lambda message: snapshots.__setitem__(slice(None), [message["table"]]))
    engine.drive()
    for _ in range(300):
        table = snapshots[0]
        assert all(p["hand"] is None for p in table["players"][1:])
        assert "base_url" not in json.dumps(table) and "api_key_env" not in json.dumps(table)
        if engine.status == "finished":
            break
        engine.action(table["window"], table["candidates"][0]["id"])
    assert engine.status == "finished" and llm_server.calls
    assert [o["kind"] for o in snapshots[0]["opponents"]] == ["llm"] * 3
    assert all(player.stats["decisions"] > 0 for player in engine.loop.players[1:])
    assert not snapshots[0]["cross_mode"]


def test_catalog_only_lists_server_configured_models_and_never_key_or_endpoint(tmp_path, monkeypatch):
    path = tmp_path / "llms.json"
    path.write_text(json.dumps([dict(id="test", name="测试服务", model="model", base_url="https://example.invalid/v1", api_key_env="TEST_LLM_KEY")]))
    monkeypatch.setenv("TEST_LLM_KEY", "secret-value")
    app = create_app(models_root=tmp_path / "missing-models", llm_config=path)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        bootstrap = client.get("/api/bootstrap").json()
        headers = {"X-Session-Token": bootstrap["token"], "Origin": "http://127.0.0.1:8765"}
        catalog = client.get("/api/catalog", headers=headers).json()
        assert catalog["llms"] == [dict(id="llm:test", name="测试服务", model="model")]
        assert all(word not in json.dumps(catalog) for word in ("secret-value", "TEST_LLM_KEY", "example.invalid"))
        refreshed = client.post("/api/catalog/refresh", headers=headers, json={}).json()
        assert refreshed["llms"] == catalog["llms"]


def test_unknown_llm_and_browser_supplied_endpoint_are_rejected_before_spawn(tmp_path):
    session = Session(tmp_path)
    try:
        page = uuid.uuid4().hex
        session.init_page(page, session.generation)
        with pytest.raises(MahjongError, match="未配置"):
            session.create(page, "create-unknown", "single", ["llm:unknown", "random", "random"])
        with pytest.raises(MahjongError):
            session.create(page, "create-object", "single", [dict(kind="llm", base_url="https://untrusted.invalid"), "random", "random"])
        assert session.process is None and session.table is None
    finally:
        session.close()


def test_server_configuration_survives_real_spawn_and_http_choices_return_through_ipc(tmp_path, llm_server):
    path = tmp_path / "llms.json"
    path.write_text(json.dumps([dict(id="test", name="Spawn LLM", model="test", base_url=llm_server.url)]))
    session = Session(tmp_path / "models", llm_config=path, lease_seconds=90)
    try:
        page = uuid.uuid4().hex
        session.init_page(page, session.generation)
        session.create(page, "create-llm-test", "single", ["llm:test", "random", "random"])
        wait_until(lambda: session.snapshot(page)["state"] != "loading", timeout=30)
        table = session.snapshot(page)
        assert table["state"] == "playing", table
        session.submit(page, table["table_id"], "llm-action-test", "action",
                       window=table["window"], candidate=table["candidates"][0]["id"])
        wait_until(lambda: bool(llm_server.calls) and not session.snapshot(page)["busy"], timeout=30)
        table = session.snapshot(page)
        assert table["state"] in ("playing", "round_result", "finished"), table
        assert table["opponents"][0]["kind"] == "llm"
        assert all(p["hand"] is None for p in table["players"][1:])
        assert "base_url" not in json.dumps(table)
    finally:
        session.close()
