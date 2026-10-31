import json
import time
from types import SimpleNamespace

import pytest

from mahjong.arena.llm import (
    ChatCompletionClient, Completion, LLMConfig, LLMPlayer, decision_prompt, load_llm_configs,
)
from mahjong.contracts import MahjongError
from tests.unit.inference.test_policy import view


def config(**kwargs):
    values = dict(id="test", name="Test LLM", model="test-model", base_url="http://127.0.0.1:8000/v1")
    values.update(kwargs)
    return LLMConfig(**values)


class StubClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, **kwargs):
        self.calls.append(json.loads(json.dumps(messages)))
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return Completion(result, 10, 3)


def choose(player, decision=None, **kwargs):
    return player.choose(view() if decision is None else decision, deadline=time.monotonic() + 5, **kwargs)


@pytest.mark.parametrize("content", ['{"candidate_id":"1"}', '```json\n{"candidate_id":"1"}\n```'])
def test_legal_choice_preserves_exact_view_key_and_execution_ref(content):
    v = view()
    player = LLMPlayer(config(), client=StubClient(content))
    choice = choose(player, v)
    assert choice.key == v.key and choice.execution_ref == "1" and choice.generation is None
    assert player.stats["decisions"] == 1 and player.stats["prompt_tokens"] == 10


@pytest.mark.parametrize("invalid", ['{"candidate_id":"9999"}', '{"candidate_id":1}', 'null', '[]', 'I discard 5m', '{"action":"A_PASS"}'])
def test_invalid_output_corrects_within_budget_and_never_falls_back(invalid):
    client = StubClient(invalid, '{"candidate_id":"0"}')
    player = LLMPlayer(config(), client=client)
    assert choose(player).execution_ref == "0"
    assert player.stats["requests"] == 2 and player.stats["invalid_responses"] == 1 and player.stats["retries"] == 1
    assert len(client.calls[1]) == 3
    assert invalid not in client.calls[1][-1]["content"]
    failing = LLMPlayer(config(), client=StubClient(invalid, invalid))
    with pytest.raises(MahjongError, match="LLM_DECISION_FAILED"):
        choose(failing)
    assert failing.stats["decisions"] == 0


def test_calls_are_stateless_and_no_cross_seat_or_previous_answer_messages_survive():
    client = StubClient('{"candidate_id":"0"}', '{"candidate_id":"1"}')
    player = LLMPlayer(config(), client=client)
    choose(player)
    choose(player, view(revision=1))
    assert len(client.calls[0]) == len(client.calls[1]) == 2
    assert all(m["role"] != "assistant" for call in client.calls for m in call)


def test_incomplete_view_and_cancelled_or_expired_decisions_do_not_call_model():
    client = StubClient()
    player = LLMPlayer(config(), client=client)
    v = SimpleNamespace(legal_set_complete=False, legal_moves=[])
    with pytest.raises(MahjongError, match="LEGAL_SET_INCOMPLETE"):
        choose(player, v)
    with pytest.raises(MahjongError, match="CANCELLED"):
        choose(player, cancel=lambda: True)
    with pytest.raises(MahjongError, match="DEADLINE_EXCEEDED"):
        player.choose(view(), deadline=time.monotonic() - 1)
    assert not client.calls


@pytest.mark.parametrize("kwargs", [dict(max_attempts=0), dict(max_attempts=6), dict(max_tokens=True),
    dict(request_timeout=float("nan")), dict(temperature=float("inf")),
    dict(base_url="https://key:secret@service.invalid/v1"), dict(base_url="https://service.invalid/v1?key=secret"),
    dict(api_key_env="literal-secret!"), dict(id="llm:bad"), dict(token_limit_field="invalid")])
def test_invalid_llm_configuration(kwargs):
    values = dict(id="test", name="Test", model="test", base_url="http://localhost/v1")
    values.update(kwargs)
    with pytest.raises(MahjongError, match="CONFIG_INVALID"):
        LLMConfig(**values)


def test_registry_validates_duplicate_and_unknown_fields_without_exposing_values(tmp_path):
    path = tmp_path / "llms.json"
    entry = dict(id="test", name="Test", model="test", base_url="http://localhost/v1", api_key_env="TEST_LLM_KEY")
    path.write_text(json.dumps([entry]))
    configs = load_llm_configs(path)
    assert configs["llm:test"].public() == dict(id="llm:test", name="Test", model="test")
    path.write_text(json.dumps([entry, entry]))
    with pytest.raises(MahjongError, match="CONFIG_INVALID"):
        load_llm_configs(path)
    path.write_text(json.dumps([dict(entry, api_key="sensitive-value")]))
    with pytest.raises(MahjongError) as error:
        load_llm_configs(path)
    assert "sensitive-value" not in str(error.value)


def test_real_http_request_authentication_payload_and_token_usage(llm_server, monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "test-secret")
    c = LLMConfig(id="test", name="Test", model="served-model", base_url=llm_server.url,
                  api_key_env="TEST_LLM_KEY", token_limit_field="max_completion_tokens")
    player = LLMPlayer(c)
    assert choose(player).execution_ref in {m.execution_ref for m in view().legal_moves}
    call = llm_server.calls[0]
    assert call["path"] == "/v1/chat/completions" and call["authorization"] == "Bearer test-secret"
    assert call["body"]["max_completion_tokens"] == 512 and "temperature" not in call["body"]
    assert player.stats["prompt_tokens"] == 20 and player.stats["completion_tokens"] == 5
    assert "test-secret" not in json.dumps(call["body"])


def test_http_transient_failures_retry_but_auth_errors_fail_without_body_leak(llm_server):
    c = LLMConfig(id="test", name="Test", model="test", base_url=llm_server.url)
    llm_server.replies.append((429, {"error": "private provider body"}))
    player = LLMPlayer(c)
    choose(player)
    assert player.stats["requests"] == 2
    llm_server.replies.append((401, {"error": "secret-key private provider body"}))
    with pytest.raises(MahjongError, match="HTTP 401") as error:
        choose(player)
    assert "private" not in str(error.value) and player.stats["requests"] == 3


def test_whole_http_operation_has_deadline_even_if_transport_hangs(monkeypatch):
    client = ChatCompletionClient(config())
    def slow(*args):
        time.sleep(0.3)
        return Completion('{"candidate_id":"0"}')
    monkeypatch.setattr(client, "_request", slow)
    start = time.monotonic()
    with pytest.raises(MahjongError, match="DEADLINE_EXCEEDED"):
        client.complete([], timeout=0.03)
    assert time.monotonic() - start < 0.2
    with pytest.raises(MahjongError, match="CANCELLED"):
        client.complete([], timeout=1, cancel=lambda: True)


def test_missing_key_is_checked_without_revealing_any_other_environment(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_LLM_KEY", raising=False)
    with pytest.raises(MahjongError, match="MISSING_TEST_LLM_KEY"):
        ChatCompletionClient(config(api_key_env="MISSING_TEST_LLM_KEY"))


def test_prompt_contains_named_candidates_and_no_execution_environment():
    prompt = decision_prompt(view())
    assert prompt["legal_actions"][1]["candidate_id"] == "1"
    assert prompt["event_fields"]["EV_DRAW"] == ("actor", "tile", "draw_kind")
    assert all(word not in json.dumps(prompt) for word in ("env_instance_id", "binding_digest", "prefix_ids", "native_candidates"))


def test_opt_in_summary_is_the_final_short_reason_and_never_provider_internal_reasoning(llm_server):
    llm_server.replies.append((200, dict(choices=[dict(message=dict(
        content=json.dumps(dict(candidate_id="0", reason="先保留可用的顺子组合。")),
        reasoning_content="private internal reasoning must not be displayed"))])))
    c = config(base_url=llm_server.url)
    player = LLMPlayer(c, explain=True)
    assert choose(player).summary == "先保留可用的顺子组合。"
    assert "brief decision summary" in llm_server.calls[0]["body"]["messages"][0]["content"]
    prompt = json.loads(llm_server.calls[0]["body"]["messages"][1]["content"])
    assert prompt["schema"] == "riichi-legal-choice-summary/1"
    # Formal evaluation keeps its original action-only prompt and payload.
    action_only = LLMPlayer(c, client=StubClient('{"candidate_id":"0","reason":"not requested"}'))
    assert choose(action_only).summary is None
    long_reason = LLMPlayer(c, client=StubClient(json.dumps(dict(candidate_id="0", reason="字" * 1000))), explain=True)
    assert len(choose(long_reason).summary) == 600
