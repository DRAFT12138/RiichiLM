"""Actual tiny NTP checkpoint loading, source identity and per-seat generation."""

from dataclasses import replace
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest
import torch
import yaml

from mahjong.app.assembly import build_model
from mahjong.app.config import parse_model
from mahjong.contracts import MahjongError, canonical_json
from mahjong.data.mahjong.corpus import open_policy_reader
from mahjong.data.mahjong.policy import dataset_fingerprint
from mahjong.inference.policy import PolicyInference
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, LEGACY_SCHEMA
from mahjong.training.checkpoint import FileCheckpointStore
from mahjong.web.catalog import Catalog
from mahjong.web.worker import TableEngine, profile_for
from mahjong.arena.llm import ChatCompletionClient, Completion
from mahjong.arena.evaluation import evaluate_cli
from tests.unit.data.test_corpus import corpus
from tests.unit.training.storage.test_checkpoint import (
    make_run,
    make_attempt,
    snapshot_for,
)


@pytest.fixture
def models(tmp_path, corpus):
    torch.set_num_threads(1)
    output = tmp_path / "models"
    for index, schema in enumerate((LEGACY_SCHEMA, CONTEXT_SCHEMA, CONTEXT_SCHEMA)):
        parent = output / str(index)
        parent.mkdir(parents=True)
        context = make_attempt(make_run(parent))
        (context.run_dir / "run.json").write_text(
            json.dumps(dict(run_id="R", experiment=f"tiny-{index}"))
        )
        config = parse_model(
            [
                f"model.input_schema={schema}",
                "model.num_hidden_layers=1",
                "model.hidden_size=8",
                "model.intermediate_size=16",
                "model.num_attention_heads=1",
                "model.num_key_value_heads=1",
                "model.head_dim=8",
                "model.vocab_size=1024",
                "model.max_position_embeddings=8192",
            ]
        )
        config["seed"] = 42 + index
        config["data"] = dict(
            name="majsoul_policy",
            build_dir=str(corpus),
            split_seed=7,
            validation_fraction=0.5,
            shared_context=index == 2,
        )
        model = build_model(config)
        with open_policy_reader(corpus) as reader:
            fingerprint = dataset_fingerprint(
                reader,
                split_seed=7,
                validation_fraction=0.5,
                max_sequence_length=8192,
                input_schema=schema,
            )
        snapshot = snapshot_for(context)
        descriptors = tuple(
            model.descriptor()
            if d.kind == "model"
            else replace(d, data_fingerprint=fingerprint)
            for d in snapshot.descriptors
        )
        snapshot = replace(
            snapshot,
            config=config,
            descriptors=descriptors,
            frozen_config_hash=hashlib.sha256(
                canonical_json(config).encode()
            ).hexdigest(),
            runtime=replace(
                snapshot.runtime, model=dict(model.state_dict()), auxiliary={}
            ),
        )
        FileCheckpointStore().save(snapshot, context)
        (context.run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    # Each fixture run lives one folder deeper; expose three ordinary model runs.
    root = tmp_path / "visible"
    root.mkdir()
    for i, path in enumerate(sorted(output.glob("*/R"))):
        path.rename(root / f"run-{i}")
    catalog = Catalog(root)
    entries = [catalog.resolve(r["checkpoints"][0]["id"]) for r in catalog.data["runs"]]
    return entries


def test_three_distinct_models_mixed_schemas_and_same_model_dedup(models, monkeypatch):
    loaded = []
    original = PolicyInference.from_checkpoint

    def load(cls, path, **kwargs):
        loaded.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(PolicyInference, "from_checkpoint", classmethod(load))
    snapshots = []
    engine = TableEngine(
        dict(mode="tonpuu", opponents=models, seed=42),
        emit=lambda v: snapshots.__setitem__(slice(None), [v]),
    )
    assert len(loaded) == 3 and len({id(m) for _, m in engine.controllers[1:]}) == 3
    assert [m.input_schema for _, m in engine.controllers[1:]] == [
        LEGACY_SCHEMA,
        CONTEXT_SCHEMA,
        CONTEXT_SCHEMA,
    ]
    engine.drive()
    # Delete already loaded files: decisions still use the validated in-memory instances.
    shutil.rmtree(models[0]["path"])
    for _ in range(4):
        snapshot = snapshots[0]["table"]
        candidate = next(
            (c for c in snapshot["candidates"] if c["kind"] == "A_DISCARD"),
            snapshot["candidates"][0],
        )
        engine.action(snapshot["window"], candidate["id"])
    assert engine.timings["last_decision_seconds"] is not None
    assert snapshots[0]["table"]["cross_mode"]
    loaded.clear()
    same = TableEngine(dict(mode="hanchan", opponents=[models[1]] * 3, seed=42))
    assert len(loaded) == 1 and len({id(m) for _, m in same.controllers[1:]}) == 1


def test_choice_identity_and_source_evidence_fail_closed(models):
    changed = {**models[0], "manifest_hash": "0" * 64}
    with pytest.raises(MahjongError, match="实际加载发布物"):
        TableEngine(dict(mode="single", opponents=[changed, None, None], seed=42))
    p = PolicyInference.from_checkpoint(models[0]["path"])
    p.source["training_data_fingerprint"] = "0" * 64
    with pytest.raises(MahjongError, match="fingerprint"):
        p.verify_training_profile(profile_for("hanchan"))
    with pytest.raises(MahjongError):
        PolicyInference.from_checkpoint(models[1]["path"]).new_bridge(
            profile_for("tonpuu")
        )


def test_llm_checkpoint_and_random_share_a_table_without_bypassing_model_binding(models, monkeypatch, tmp_path):
    calls = []
    def complete(self, messages, **kwargs):
        prompt = json.loads(messages[1]["content"])
        calls.append(prompt)
        return Completion(json.dumps(dict(candidate_id=prompt["legal_actions"][0]["candidate_id"])))
    monkeypatch.setattr(ChatCompletionClient, "complete", complete)
    llm = dict(kind="llm", config=dict(id="test", name="LLM", model="test", base_url="http://localhost/v1"))
    latest = []
    engine = TableEngine(dict(mode="single", opponents=[llm, models[2], None], seed=42),
                         emit=lambda v: latest.__setitem__(slice(None), [v["table"]]))
    engine.drive()
    for _ in range(6):
        if engine.status != "playing":
            break
        table = latest[0]
        engine.action(table["window"], table["candidates"][0]["id"])
    assert calls and engine.loop.decisions[2] > 0
    assert [c[0] for c in engine.controllers] == ["human", "llm", "model", "random"]
    config_path = tmp_path / "llms.json"
    config_path.write_text(json.dumps([llm["config"]]))
    output = tmp_path / "evaluation.json"
    options = SimpleNamespace(players=["checkpoint:" + models[2]["path"], "llm:test", "random", "random"],
                              mode="single", device="cpu", llm_config=config_path, seeds=[42], rotate=False,
                              decision_timeout=30, max_windows=1000, output=str(output))
    assert evaluate_cli(options) == 0
    report = json.loads(output.read_text())
    assert report["players"][0]["checkpoint_id"] == models[2]["checkpoint_id"]
    assert report["outcomes"][0]["llm_stats"][1]["decisions"] > 0
    assert report["model_sampling"] == "categorical"
    assert report["outcomes"][0]["model_sampling_seeds"][0] is not None


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='GPU unavailable'))])
def test_four_checkpoint_spectators_share_weights_but_keep_independent_sampling_rng(models, device):
    if device == 'cuda:0' and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 GPU unavailable')
    engine = TableEngine(dict(mode="single", opponents=[models[2]] * 4, spectator=True, seed=42), device=device)
    from mahjong.policy.execution import policy_precision
    assert policy_precision(engine.controllers[0][1].model) == ('bf16-mixed' if device == 'cuda:0' else '32-true')
    assert len({id(policy) for _, policy in engine.controllers}) == 1
    seeds = [player.generator.initial_seed() for player in engine.loop.players]
    assert len(set(seeds)) == 4
    engine.drive()
    snapshot = engine.publish()
    assert all(c["sampling"] == "categorical" for c in snapshot["controllers"])
    note = snapshot["decision_notes"][0]
    assert note["sampling"] == "categorical" and note["summary"] is None
    assert 0 < note["selected_probability"] <= 1
    assert engine.loop.players[0].policy is engine.loop.players[1].policy
