"""OpenAI-compatible chat player; independent of Web and model execution."""

from dataclasses import dataclass, fields, is_dataclass
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from mahjong.contracts import MahjongError
from mahjong.tokenizer import Atom
from .players import PlayerChoice, check_deadline


PROMPT_VERSION = "riichi-legal-choice/1"
SYSTEM_PROMPT = """You play four-player Japanese riichi mahjong. Maximize your final placement
and score using only the supplied observation. Choose exactly one entry from legal_actions.
Reply with a JSON object containing only its candidate_id, e.g. {"candidate_id":"3"}.
Candidate IDs are opaque strings valid only for this decision. Do not invent actions.
All seats are relative: SELF is you; SHIMOCHA is to your right, TOIMEN opposite,
KAMICHA to your left. Score and wind arrays use that order. TILE_1m..9m are characters,
1p..9p circles, 1s..9s bamboo; 5mr/5pr/5sr are red fives; E/S/W/N winds,
P/F/C white/green/red dragons. TILE_HIDDEN means unknown, never infer it as a known tile.
hand_state is your current concealed hand and melds; current header is the round's
initial state, followed by ordered visible_history events. previous_rounds are oldest first.
Event payload field order is given in event_fields. Other players' draws are hidden.
A_DISCARD discards its tile; A_RIICHI_DISCARD declares riichi and discards it.
TSUMOGIRI means drawn tile, TEDASHI means from hand. A_PASS declines a response;
A_CHI/A_PON/A_DAIMINKAN name source seat, called tile, then consumed tiles;
A_ANKAN lists four tiles; A_KAKAN names meld ID and added tile; A_RON names source
and winning tile; A_TSUMO_WIN wins by self draw; A_ABORT_KYUUSHU aborts the round.
Use the exact legal list even if only one entry is supplied. Do not output reasoning."""

SUMMARY_PROMPT_VERSION = "riichi-legal-choice-summary/1"
SUMMARY_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    'Reply with a JSON object containing only its candidate_id, e.g. {"candidate_id":"3"}.',
    'Reply with JSON containing candidate_id and reason, e.g. {"candidate_id":"3","reason":"Keep a flexible hand."}.'
).replace("Do not output reasoning.",
          "Include only a brief decision summary in reason (one or two sentences, preferably Chinese). "
          "Do not provide detailed internal reasoning or a step-by-step thought process.")


@dataclass(frozen=True)
class LLMConfig:
    id: str
    name: str
    model: str
    base_url: str
    api_key_env: str | None = None
    temperature: float | None = None
    max_tokens: int = 512
    max_attempts: int = 2
    request_timeout: float = 60
    token_limit_field: str = "max_tokens"

    def __post_init__(self):
        if not isinstance(self.id, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.id) is None:
            raise MahjongError("CONFIG_INVALID", "LLM id 仅支持字母、数字、下划线和连字符")
        if any(not isinstance(v, str) or not v.strip() for v in (self.name, self.model, self.base_url)):
            raise MahjongError("CONFIG_INVALID", "LLM name、model、base_url 不能为空")
        try:
            parsed = urlsplit(self.base_url)
            parsed.port
        except ValueError:
            raise MahjongError("CONFIG_INVALID", "LLM base_url 地址无效") from None
        if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise MahjongError("CONFIG_INVALID", "LLM base_url 必须是无凭据及查询参数的 HTTP(S) 地址")
        if self.api_key_env is not None and (not isinstance(self.api_key_env, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env) is None):
            raise MahjongError("CONFIG_INVALID", "LLM api_key_env 必须为环境变量名")
        if (type(self.max_tokens) is not int or self.max_tokens < 1
                or type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 5
                or type(self.request_timeout) not in (int, float) or not math.isfinite(self.request_timeout) or self.request_timeout <= 0
                or self.temperature is not None and (type(self.temperature) not in (int, float) or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2)
                or self.token_limit_field not in ("max_tokens", "max_completion_tokens")):
            raise MahjongError("CONFIG_INVALID", "LLM 采样、重试或时限配置无效")

    def public(self):
        return dict(id="llm:" + self.id, name=self.name, model=self.model)


def load_llm_configs(path=None):
    if path is None:
        return {}
    try:
        entries = json.loads(Path(path).read_text())
        if not isinstance(entries, list):
            raise ValueError("expected list")
        configs = [LLMConfig(**item) for item in entries]
        result = {"llm:" + c.id: c for c in configs}
        if len(result) != len(configs):
            raise ValueError("duplicate id")
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise MahjongError("CONFIG_INVALID", "LLM 配置必须为字段有效、id 唯一的 JSON 数组") from exc


def plain(value):
    if isinstance(value, Atom):
        return value.name
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def decision_prompt(view, *, explain=False):
    """Serialize only the bridge's existing privacy projection, never an environment."""
    from mahjong.tokenizer.codec import EVENT_FIELDS

    sequence = view.sequence
    return dict(
        schema=SUMMARY_PROMPT_VERSION if explain else PROMPT_VERSION,
        observation=plain(sequence),
        event_fields=EVENT_FIELDS,
        legal_actions=[dict(candidate_id=m.execution_ref, **plain(m.action)) for m in view.legal_moves],
    )


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatCompletionClient:
    def __init__(self, config):
        self.config = config
        self._key = None if config.api_key_env is None else os.environ.get(config.api_key_env)
        if config.api_key_env is not None and not self._key:
            raise MahjongError("CONFIG_INVALID", f"请设置 LLM 密钥环境变量 {config.api_key_env}")

    def _request(self, messages, timeout):
        config = self.config
        body = dict(model=config.model, messages=messages, stream=False,
                    **{config.token_limit_field: config.max_tokens})
        if config.temperature is not None:
            body["temperature"] = config.temperature
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key is not None:
            headers["Authorization"] = "Bearer " + self._key
        request = Request(config.base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
                raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise MahjongError("LLM_RESPONSE_INVALID", "LLM 响应过大")
            result = json.loads(raw)
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("missing text")
            usage = result.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("invalid usage")
            def count(key):
                value = usage.get(key, 0)
                return value if type(value) is int and value >= 0 else 0
            return Completion(content, count("prompt_tokens"), count("completion_tokens"))
        except HTTPError as exc:
            code = "LLM_RETRYABLE" if exc.code in (408, 429) or exc.code >= 500 else "LLM_HTTP_FAILED"
            status = exc.code
            exc.close()
            raise MahjongError(code, f"LLM 服务返回 HTTP {status}") from None
        except (TimeoutError, URLError, OSError):
            raise MahjongError("LLM_RETRYABLE", "LLM 服务连接失败或请求超时") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise MahjongError("LLM_RESPONSE_INVALID", "LLM 服务响应格式无效") from None

    def complete(self, messages, *, timeout, cancel=None):
        # A socket's timeout is per read. Bound the entire HTTP operation as well,
        # including slow response streams, DNS, and cancellation. No remote body
        # or exception text reaches the caller/browser.
        deadline = time.monotonic() + timeout
        check_deadline(deadline, cancel)
        result = queue.Queue(maxsize=1)
        def request():
            try:
                result.put((self._request(messages, timeout), None))
            except Exception as exc:
                result.put((None, exc if isinstance(exc, MahjongError) else
                            MahjongError("LLM_HTTP_FAILED", "LLM 请求失败")))
        threading.Thread(target=request, name="llm-http", daemon=True).start()
        while True:
            check_deadline(deadline, cancel)
            try:
                value, error = result.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            check_deadline(deadline, cancel)
            if error is not None:
                raise error
            return value


class LLMPlayer:
    def __init__(self, config, *, client=None, explain=False):
        self.config = config
        self.explain = explain
        self.client = ChatCompletionClient(config) if client is None else client
        self.stats = dict(decisions=0, requests=0, invalid_responses=0, retries=0,
                          prompt_tokens=0, completion_tokens=0, seconds=0.0)

    def choose(self, view, *, deadline, cancel=None):
        check_deadline(deadline, cancel)
        if not view.legal_set_complete or not view.legal_moves:
            raise MahjongError("LEGAL_SET_INCOMPLETE", "LLM 需要完整且非空的合法动作集")
        moves = {m.execution_ref: m for m in view.legal_moves}
        messages = [dict(role="system", content=SUMMARY_SYSTEM_PROMPT if self.explain else SYSTEM_PROMPT),
                    dict(role="user", content=json.dumps(decision_prompt(view, explain=self.explain), ensure_ascii=False))]
        start = time.monotonic()
        try:
            for attempt in range(self.config.max_attempts):
                check_deadline(deadline, cancel)
                if attempt:
                    self.stats["retries"] += 1
                self.stats["requests"] += 1
                try:
                    result = self.client.complete(messages, timeout=min(self.config.request_timeout,
                                                  deadline - time.monotonic()), cancel=cancel)
                    check_deadline(deadline, cancel)
                    self.stats["prompt_tokens"] += result.prompt_tokens
                    self.stats["completion_tokens"] += result.completion_tokens
                    content = result.content.strip()
                    if content.startswith("```json\n") and content.endswith("```"):
                        content = content[8:-3].strip()
                    value = json.loads(content)
                    candidate = value.get("candidate_id") if isinstance(value, dict) else None
                    if not isinstance(candidate, str) or candidate not in moves:
                        raise ValueError("unknown candidate")
                except (ValueError, TypeError):
                    self.stats["invalid_responses"] += 1
                    messages.append(dict(role="user", content='Invalid response. Return only {"candidate_id":"one of the supplied IDs"}.'))
                    continue
                except MahjongError as exc:
                    if exc.code not in ("LLM_RETRYABLE", "LLM_RESPONSE_INVALID"):
                        raise
                    if exc.code == "LLM_RESPONSE_INVALID":
                        self.stats["invalid_responses"] += 1
                    continue
                self.stats["decisions"] += 1
                summary = value.get("reason") if self.explain else None
                summary = summary.strip()[:600] if isinstance(summary, str) else None
                return PlayerChoice(view.key, candidate, summary=summary or None)
            raise MahjongError("LLM_DECISION_FAILED", f"LLM {self.config.name} 在有限重试内未返回合法动作")
        finally:
            self.stats["seconds"] += time.monotonic() - start
