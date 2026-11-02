"""Single ephemeral page/table owner with independent watchdog and process cleanup."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
import multiprocessing
from pathlib import Path
import secrets
import threading
import time
import uuid

from mahjong.contracts import MahjongError
from mahjong.arena.llm import load_llm_configs
from .catalog import Catalog


def worker_entry(connection, config, device, decision_timeout):
    from .worker import run_worker

    run_worker(connection, config, device, decision_timeout)


def require(condition, message):
    if not condition:
        raise MahjongError("STALE_DECISION", message)


class Session:
    def __init__(
        self,
        models_root,
        *,
        weights_root=None,
        device="cpu",
        lease_seconds=30,
        load_timeout=300,
        decision_timeout=120,
        llm_config=None,
        worker=worker_entry,
    ):
        for value in (lease_seconds, load_timeout, decision_timeout):
            if not math.isfinite(value) or value <= 0:
                raise MahjongError("CONFIG_INVALID", "超时参数必须为有限正数")
        if device not in ("cpu", "cuda:0"):
            raise MahjongError("CONFIG_INVALID", "不支持的设备")
        self.catalog = Catalog(models_root, weights_root=weights_root)
        self.llm_configs = load_llm_configs(llm_config)
        self.device, self.lease_seconds = device, lease_seconds
        self.load_timeout, self.decision_timeout = load_timeout, decision_timeout
        self.worker = worker
        self.lock = threading.RLock()
        self.page = None
        self.generation = 0
        self.retired_pages = set()
        self.table = None
        self.process = self.connection = None
        self.requests = {}
        self.retired_create_ids = set()
        self.active_create_id = None
        self.pending = None
        self.deadline = None
        self.last_poll = time.monotonic()
        self.stopped = threading.Event()
        self.monitor = threading.Thread(
            target=self._watch, name="web-play-watchdog", daemon=True
        )
        self.monitor.start()

    def bootstrap(self):
        with self.lock:
            return dict(
                generation=self.generation,
                device=self.device,
                lease_seconds=self.lease_seconds,
            )

    def init_page(self, page_id, generation):
        with self.lock:
            require(
                isinstance(page_id, str) and 8 <= len(page_id) <= 128, "页面身份无效"
            )
            if page_id == self.page:
                return dict(page_id=page_id)
            require(
                page_id not in self.retired_pages
                and type(generation) is int
                and generation == self.generation,
                "页面已过期，请刷新",
            )
            self._stop()
            if self.page:
                self.retired_pages.add(self.page)
            self.page = page_id
            self.generation += 1
            self.table = None
            self.requests = {}
            self.retired_create_ids = set()
            self.active_create_id = None
            return dict(page_id=page_id)

    def _owner(self, page_id):
        require(
            page_id is not None and page_id == self.page, "页面已失效，请刷新并重新开局"
        )

    def snapshot(self, page_id):
        with self.lock:
            self._owner(page_id)
            self.last_poll = time.monotonic()
            return deepcopy(self.table)

    def _stop(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            require(not self.process.is_alive(), "旧牌桌尚未回收")
            self.process.close()
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None
        self.pending = self.deadline = None

    def _request(self, request_id, payload):
        require(
            isinstance(request_id, str) and 8 <= len(request_id) <= 128, "请求身份无效"
        )
        previous = self.requests.get(request_id)
        if previous is not None:
            require(previous["payload"] == payload, "同一请求ID不能更换内容")
            return previous["result"]
        return None

    def create(self, page_id, request_id, mode, opponents, spectator=False):
        with self.lock:
            self._owner(page_id)
            payload = dict(operation="create", mode=mode, opponents=opponents, spectator=spectator)
            require(isinstance(request_id, str) and 8 <= len(request_id) <= 128, "请求身份无效")
            require(request_id not in self.retired_create_ids, "本次开局已结束，请重新开局")
            previous = self._request(request_id, payload)
            if previous is not None:
                return deepcopy(previous)
            require(self.table is None, "请先退出当前桌")
            require(
                mode in ("single", "tonpuu", "hanchan")
                and type(spectator) is bool
                and isinstance(opponents, list)
                and len(opponents) == (4 if spectator else 3),
                "请选择赛制；真人对战需要三席AI，观战需要四席AI",
            )
            resolved = []
            for item in opponents:
                require(isinstance(item, str), "请选择有效的玩家策略")
                if item == "random":
                    resolved.append(None)
                elif item in self.llm_configs:
                    resolved.append(dict(kind="llm", config=asdict(self.llm_configs[item])))
                elif item.startswith("llm:"):
                    raise MahjongError("CONFIG_INVALID", "LLM 策略未配置，请检查服务启动参数")
                else:
                    resolved.append(self.catalog.resolve(item))
            config = dict(mode=mode, opponents=resolved, spectator=spectator, seed=secrets.randbits(32))
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(
                target=self.worker,
                args=(child, config, self.device, self.decision_timeout),
                daemon=True,
            )
            table_id = uuid.uuid4().hex
            self.table = dict(
                table_id=table_id,
                page_id=page_id,
                state="loading",
                mode=mode,
                spectator=spectator,
                view_version=1,
                busy=False,
                error=None,
            )
            self.process, self.connection = process, parent
            self.deadline = time.monotonic() + self.load_timeout
            self.last_poll = time.monotonic()
            try:
                process.start()
            except Exception:
                parent.close()
                self.process = self.connection = self.table = None
                raise
            finally:
                child.close()
            result = dict(table_id=table_id, status="accepted")
            self.requests[request_id] = dict(payload=payload, result=result)
            self.active_create_id = request_id
            return deepcopy(result)

    def submit(self, page_id, table_id, request_id, operation, **arguments):
        with self.lock:
            self._owner(page_id)
            require(
                self.table is not None and self.table["table_id"] == table_id,
                "牌桌已结束或已更换",
            )
            payload = dict(operation=operation, table_id=table_id, **arguments)
            previous = self._request(request_id, payload)
            if previous is not None:
                return deepcopy(previous)
            require(
                self.pending is None and self.connection is not None, "正在处理上一操作"
            )
            if operation == "action":
                require(
                    not self.table.get("spectator") and self.table["state"] == "playing"
                    and self.table.get("window") == arguments.get("window"),
                    "动作窗口已过期",
                )
                require(
                    any(
                        c["id"] == arguments.get("candidate")
                        for c in self.table.get("candidates", [])
                    ),
                    "候选动作无效",
                )
            elif operation == "continue":
                require(
                    self.table["state"] == "round_result"
                    and (self.table["round_id"], self.table["revision"])
                    == (arguments.get("round_id"), arguments.get("revision")),
                    "局末结果已过期",
                )
            elif operation == "control":
                control, value = arguments.get("control"), arguments.get("value")
                require(self.table.get("spectator") and self.table["state"] in ("playing", "round_result"), "当前不能控制观战")
                require(control in ("pause", "resume", "step", "view", "speed"), "未知观战操作")
                if control in ("pause", "resume", "step"):
                    require(self.table["state"] == "playing", "当前不是进行中的牌局")
                elif control == "view":
                    require(type(value) is int and 0 <= value < 4, "观战席位无效")
                elif control == "speed":
                    require(type(value) in (int, float) and math.isfinite(value) and 0.1 <= value <= 5, "播放间隔必须在0.1—5秒之间")
            else:
                require(False, "未知操作")
            result = dict(status="processing", request_id=request_id)
            self.requests[request_id] = dict(payload=payload, result=result)
            self.pending = request_id
            self.table["busy"] = True
            self.table["command_error"] = None
            self.table["view_version"] += 1
            self.connection.send(
                dict(operation=operation, request_id=request_id, **arguments)
            )
            return deepcopy(result)

    def close_table(self, page_id, table_id):
        with self.lock:
            self._owner(page_id)
            if self.table is None:
                return dict(status="closed")
            require(self.table["table_id"] == table_id, "不能退出其他牌桌")
            self._stop()
            if self.active_create_id is not None:
                self.retired_create_ids.add(self.active_create_id)
                self.active_create_id = None
            self.table = None
            self.requests = {}
            return dict(status="closed")

    def _failed(self, error):
        self.table.update(
            state="failed",
            error=error,
            candidates=[],
            busy=False,
            view_version=self.table["view_version"] + 1,
        )
        self._stop()

    def _receive(self, message):
        kind = message["type"]
        if kind == "snapshot":
            metadata = {k: self.table[k] for k in ("table_id", "page_id")}
            metadata["command_error"] = self.table.get("command_error")
            version = self.table["view_version"] + 1
            self.table = dict(
                message["table"],
                **metadata,
                view_version=version,
                busy=self.pending is not None,
            )
            timeout = message.get("timeout")
            self.deadline = None if timeout is None else time.monotonic() + timeout
        elif kind in ("ack", "rejected"):
            request_id = message["request_id"]
            if request_id in self.requests:
                self.requests[request_id]["result"].update(
                    status="done" if kind == "ack" else "rejected",
                    **({"error": message.get("error")} if kind == "rejected" else {}),
                )
            self.pending = None
            self.table.update(
                busy=False,
                command_error=message.get("error"),
                view_version=self.table["view_version"] + 1,
            )
        elif kind == "failed":
            self._failed(message["error"])

    def _watch(self):
        while not self.stopped.wait(0.05):
            with self.lock:
                if self.table is None:
                    continue
                now = time.monotonic()
                if now - self.last_poll > self.lease_seconds:
                    self._stop()
                    if self.active_create_id is not None:
                        self.retired_create_ids.add(self.active_create_id)
                        self.active_create_id = None
                    self.table = None
                    self.requests = {}
                    continue
                try:
                    while self.connection is not None and self.connection.poll():
                        self._receive(self.connection.recv())
                    if self.table["state"] == "finished":
                        self._stop()
                        self.table["busy"] = False
                    elif self.deadline is not None and now > self.deadline:
                        self._failed("加载或模型推理超时，请重新开局")
                    elif self.process is not None and not self.process.is_alive():
                        self._failed("牌桌进程意外退出，请重新开局")
                except EOFError, BrokenPipeError, OSError:
                    if self.table["state"] == "finished":
                        self._stop()
                        self.table["busy"] = False
                    else:
                        self._failed("牌桌进程已断开，请重新开局")

    def close(self):
        self.stopped.set()
        self.monitor.join(timeout=3)
        with self.lock:
            self._stop()
            self.table = None
