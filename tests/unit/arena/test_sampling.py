"""Actual categorical draws, action likelihoods, per-seat RNG and reproducibility."""

import math
import time

import pytest
import torch

from mahjong.arena.players import ModelPlayer
from mahjong.contracts import MahjongError
from mahjong.inference.sampling import DEFAULT_INFERENCE_TEMPERATURE, DEFAULT_INFERENCE_TOP_P
from tests.unit.inference.test_policy import inference, view


def test_model_player_samples_rather_than_argmax_and_has_its_own_rng(inference):
    with torch.no_grad():
        for parameter in inference.model.parameters():
            parameter.zero_()
    player = ModelPlayer(inference, seed=42)
    isolated = ModelPlayer(inference, seed=42)
    other = ModelPlayer(inference, seed=73)
    v = view()
    samples = []
    for _ in range(48):
        result = player.choose(v, deadline=time.monotonic() + 10).generation
        # Interleaving another seat that shares weights cannot advance this RNG.
        other.choose(v, deadline=time.monotonic() + 10)
        expected = isolated.choose(v, deadline=time.monotonic() + 10).generation
        assert result.suffix_ids == expected.suffix_ids
        assert result.sampling == "categorical"
        assert result.behavior_action_logprob == pytest.approx(result.model_action_logprob)
        probability = math.exp(result.behavior_action_logprob)
        action = inference.codec.decode_action_suffix(result.suffix_ids)
        assert probability == pytest.approx(0.5 if action.kind == "A_PASS" else 0.25)
        samples.append(action)
    assert len(set(samples)) == 3
    assert sum(action.kind == "A_PASS" for action in samples) in range(10, 39)
    assert player.generator is not isolated.generator
    assert (player.temperature, player.top_p) == (DEFAULT_INFERENCE_TEMPERATURE, DEFAULT_INFERENCE_TOP_P)


def test_model_player_respects_custom_temperature_and_top_p(inference):
    with torch.no_grad():
        for parameter in inference.model.parameters():
            parameter.zero_()
    greedy_like = ModelPlayer(inference, seed=42, temperature=0.0, top_p=0.5)
    v = view()
    first = greedy_like.choose(v, deadline=time.monotonic() + 10).generation
    for _ in range(10):
        nxt = greedy_like.choose(v, deadline=time.monotonic() + 10).generation
        assert nxt.suffix_ids == first.suffix_ids
        assert nxt.behavior_action_logprob == 0.0
    with pytest.raises(MahjongError, match="CONFIG_INVALID"):
        ModelPlayer(inference, seed=42, temperature=2.5)
    with pytest.raises(MahjongError, match="CONFIG_INVALID"):
        ModelPlayer(inference, seed=42, top_p=0.0)
