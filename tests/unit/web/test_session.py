import time
import uuid
import pytest
from fastapi.testclient import TestClient
from mahjong.contracts import MahjongError
from mahjong.web.session import Session
from mahjong.web.server import create_app


def hanging_worker(connection, config, device, decision_timeout):
    time.sleep(60)


def budget_probe_worker(connection, config, device, decision_timeout):
    connection.send(dict(type='failed', error=f"budget={config['gpu_memory_gib']}"))
    connection.close()


def test_configured_memory_budget_reaches_real_spawn(tmp_path):
    s = Session(tmp_path, gpu_memory_gib=6.5, worker=budget_probe_worker)
    try:
        assert s.bootstrap()['gpu_memory_gib'] == 6.5
        page = uuid.uuid4().hex
        s.init_page(page, s.generation)
        s.create(page, 'budget-create', 'single', ['random']*3)
        wait_until(lambda: s.snapshot(page)['state'] == 'failed')
        assert s.snapshot(page)['error'] == 'budget=6.5'
    finally:
        s.close()


def wait_until(check, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.05)
    raise AssertionError("deadline exceeded")


def setup_session(tmp_path, **options):
    s = Session(tmp_path, **options)
    page = uuid.uuid4().hex
    s.init_page(page, s.generation)
    return s, page


def test_real_spawn_actions_idempotency_and_page_takeover(tmp_path):
    s, page = setup_session(tmp_path, lease_seconds=90)
    try:
        result = s.create(page, "create-id", "single", ["random"] * 3)
        assert s.create(page, "create-id", "single", ["random"] * 3) == result
        wait_until(lambda: s.snapshot(page)["state"] != "loading")
        t = s.snapshot(page)
        assert t["state"] == "playing", t
        assert all(p["hand"] is None for p in t["players"][1:])
        candidate = t["candidates"][0]["id"]
        kwargs = dict(window=t["window"], candidate=candidate)
        first = s.submit(page, t["table_id"], "action-id", "action", **kwargs)
        second = s.submit(page, t["table_id"], "action-id", "action", **kwargs)
        assert first["request_id"] == second["request_id"]
        with pytest.raises(MahjongError):
            s.submit(page, t["table_id"], "other-id", "action", **kwargs)
        with pytest.raises(MahjongError):
            s.submit(
                page,
                t["table_id"],
                "action-id",
                "action",
                window={},
                candidate=candidate,
            )
        process = s.process
        generation = s.generation
        replacement = uuid.uuid4().hex
        s.init_page(replacement, generation)
        assert s.snapshot(replacement) is None and s.process is None
        with pytest.raises(MahjongError):
            s.snapshot(page)
        with pytest.raises(MahjongError):
            s.close_table(page, t["table_id"])
        with pytest.raises(MahjongError):
            s.init_page(page, generation)
        with pytest.raises(MahjongError):
            s.init_page(uuid.uuid4().hex, generation)
        assert s.init_page(replacement, generation)["page_id"] == replacement
        with pytest.raises(ValueError):
            process.is_alive()  # closed process handle
        assert not list(tmp_path.iterdir())
    finally:
        s.close()


def test_watchdog_stops_computation_without_waiting_for_its_command_loop(tmp_path):
    s, page = setup_session(
        tmp_path, worker=hanging_worker, load_timeout=0.2, lease_seconds=30
    )
    try:
        s.create(page, "create-id", "single", ["random"] * 3)
        wait_until(lambda: s.snapshot(page)["state"] == "failed", timeout=8)
        assert s.process is None and "超时" in s.snapshot(page)["error"]
    finally:
        s.close()


def test_lease_expires_even_when_worker_hangs_and_service_restart_has_no_table(
    tmp_path,
):
    s, page = setup_session(
        tmp_path, worker=hanging_worker, load_timeout=60, lease_seconds=0.2
    )
    try:
        s.create(page, "create-id", "single", ["random"] * 3)
        wait_until(lambda: s.table is None and s.process is None)
        assert s.snapshot(page) is None
        with pytest.raises(MahjongError, match="本次开局已结束"):
            s.create(page, "create-id", "single", ["random"] * 3)
        assert s.snapshot(page) is None
    finally:
        s.close()
    s, page = setup_session(tmp_path)
    try:
        assert s.snapshot(page) is None
    finally:
        s.close()


def test_closed_creation_cannot_be_resurrected_by_a_late_retry(tmp_path):
    s, page = setup_session(tmp_path, worker=hanging_worker, lease_seconds=30)
    try:
        created = s.create(page, "create-id", "single", ["random"] * 3)
        s.close_table(page, created["table_id"])
        with pytest.raises(MahjongError, match="本次开局已结束"):
            s.create(page, "create-id", "single", ["random"] * 3)
        assert s.snapshot(page) is None
        next_table = s.create(page, "another-create-id", "single", ["random"] * 3)
        assert next_table["table_id"] != created["table_id"]
    finally:
        s.close()


def test_rejected_command_reason_survives_a_following_worker_snapshot(tmp_path):
    s, page = setup_session(tmp_path)
    try:
        s.table = dict(table_id="table", page_id=page, view_version=1, busy=True, state="playing")
        s.pending = "action-id"
        s.requests["action-id"] = dict(payload={}, result=dict(status="processing", request_id="action-id"))
        s._receive(dict(type="rejected", request_id="action-id", error="动作已失效"))
        s._receive(dict(type="snapshot", table=dict(state="playing")))
        assert s.requests["action-id"]["result"] == dict(status="rejected", request_id="action-id", error="动作已失效")
        assert s.snapshot(page)["command_error"] == "动作已失效"
    finally:
        s.close()


def test_same_origin_token_and_host_checks_and_no_history_api(tmp_path):
    app = create_app(models_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        b = client.get("/api/bootstrap").json()
        data = dict(page_id=uuid.uuid4().hex, generation=b["generation"])
        assert client.post("/api/page", json=data).status_code == 403
        assert (
            client.post(
                "/api/page",
                json=data,
                headers={
                    "Origin": "https://attacker.example",
                    "X-Session-Token": b["token"],
                },
            ).status_code
            == 403
        )
        headers = {"Origin": "http://127.0.0.1:8765", "X-Session-Token": b["token"]}
        assert client.post("/api/page", json=data, headers=headers).status_code == 200
        assert (
            client.get(
                "/api/bootstrap", headers={"Host": "attacker.example"}
            ).status_code
            == 403
        )
        assert client.get("/api/catalog", headers=headers).status_code == 200
        assert client.get("/api/history", headers=headers).status_code == 404
        assert client.get("/").status_code == 200
        response = client.get(f"/api/table?page_id={data['page_id']}", headers=headers)
        assert (
            response.json() is None and response.headers["cache-control"] == "no-store"
        )
