from dataclasses import fields, is_dataclass
import random

import numpy as np
import pytest
import torch

from mahjong import contracts as c
from mahjong.training.runtime import FabricRuntime
from tests.fixtures.contracts.doubles import KnownModel, KnownSource


def assert_equal(left, right):
    if is_dataclass(left):
        for field in fields(left):
            assert_equal(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def build(training=True):
    model = KnownModel()
    model.child = torch.nn.Sequential(torch.nn.Dropout(.1))
    model.register_buffer("known_buffer", torch.tensor([3.]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003) if training else None
    runtime = FabricRuntime()
    runtime.setup(model, optimizer)
    return model, optimizer, runtime


def update(runtime):
    batch = KnownSource().acquire().batch
    runtime.zero_grad()
    result = runtime.compute(batch, phase="train")
    runtime.backward(result.loss)
    runtime.check_gradients()
    runtime.step()
    return result


def test_real_update_roundtrip_and_state_ownership(tmp_path):
    model, optimizer, runtime = build()
    initial = model.weight.detach().clone()
    assert runtime.model.module.model is model
    assert optimizer.param_groups[0]["params"][0] is model.weight
    update(runtime)
    assert not torch.equal(initial, model.weight)
    state = runtime.export_state()
    assert not any(key.startswith("_forward_module") for key in state.model)
    held = c.clone_state(state.model)
    with torch.no_grad():
        model.weight.add_(9)
    assert_equal(state.model, held)
    destination_model, destination_optimizer, destination = build()
    destination.import_state(state)
    loaded = destination.export_state()
    assert_equal(state, loaded)
    state.model["weight"].add_(8)
    state.optimizer["state"][0]["exp_avg"].add_(8)
    state.auxiliary["owned"]["nested"][1][11].add_(8)
    assert_equal(destination.export_state(), loaded)
    before = destination_model.weight.detach().clone()
    update(destination)
    assert not torch.equal(before, destination_model.weight)


@pytest.mark.parametrize("exit_kind", ["normal", "cancelled", "exception"])
def test_evaluation_restores_global_private_rng_modes_grad_and_model(exit_kind):
    model, optimizer, runtime = build()
    update(runtime)
    model.train()
    model.child.eval()  # Deliberately mixed per-module flags.
    modes = [module.training for module in model.modules()]
    before = runtime.export_state()
    rng = runtime.capture_rng()
    grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    batch = KnownSource().acquire().batch
    if exit_kind == "exception":
        with pytest.raises(RuntimeError, match="injected evaluation"):
            with runtime.evaluation_scope(43):
                runtime.compute(batch, phase="eval")
                random.random()
                np.random.rand()
                raise RuntimeError("injected evaluation")
    else:
        with runtime.evaluation_scope(43):
            assert not any(module.training for module in model.modules())
            runtime.compute(batch, phase="eval")
            random.random()
            np.random.rand()
            if exit_kind == "cancelled":
                model.weight.grad = None
    assert_equal(runtime.capture_rng(), rng)
    assert_equal(runtime.export_state(), before)
    assert [module.training for module in model.modules()] == modes
    assert_equal([p.grad for p in model.parameters()], grads)


def test_eval_same_seed_identical_statistics_and_no_optimizer():
    model, _, runtime = build(training=False)
    assert runtime.optimizer is None and runtime.learning_rates() == ()
    batch = KnownSource().acquire().batch
    values = []
    for _ in range(2):
        with runtime.evaluation_scope(43):
            values.append(runtime.compute(batch, phase="eval").metrics)
    assert values[0] == values[1]


def test_rng_is_safe_load_pure_values_and_independent(tmp_path):
    _, _, runtime = build()
    rng = runtime.capture_rng()
    filename = tmp_path / "rng.pt"
    torch.save(c.encode_contract(rng), filename)
    loaded = c.decode_contract(c.RngState, torch.load(filename, weights_only=True))
    assert_equal(rng, loaded)
    runtime.restore_rng(loaded)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    runtime.restore_rng(rng)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected
    saved = runtime.capture_rng()
    loaded.torch_cpu.fill_(0)
    assert_equal(runtime.capture_rng(), saved)


def test_no_gradients_and_nonfinite_gradients_rejected():
    model, _, runtime = build()
    with pytest.raises(c.MahjongError) as error:
        runtime.check_gradients()
    assert error.value.code == "CONTRACT_MISMATCH"
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    with pytest.raises(c.MahjongError) as error:
        runtime.check_gradients()
    assert error.value.code == "NONFINITE_GRADIENT"


def test_cpu_gpu_resources_are_unavailable_with_reason():
    _, _, runtime = build()
    snapshot = runtime.resource_snapshot()
    assert snapshot.device == "cpu"
    assert snapshot.values.keys() == {"allocated", "reserved"}
    assert all(value.value is None and value.status == "unavailable" and value.reason == "cpu_execution"
               for value in snapshot.values.values())


@pytest.mark.parametrize("kwargs", [{"device": "cuda:1"}, {"precision": "16-mixed"},
                                    {"cpu_threads": 0}, {"cpu_threads": True}, {"deterministic": "false"}])
def test_unsupported_configuration_fails_before_setup(kwargs):
    with pytest.raises(c.MahjongError) as error:
        FabricRuntime(**kwargs)
    assert error.value.code == "CONFIG_INVALID" and error.value.exit_code == 2


def test_setup_once_and_strict_state_import():
    model, optimizer, runtime = build()
    with pytest.raises(c.MahjongError):
        runtime.setup(model, optimizer)
    state = runtime.export_state()
    state.model["extra"] = torch.ones(1)
    with pytest.raises(c.MahjongError) as error:
        runtime.import_state(state)
    assert error.value.code == "RESUME_INCOMPATIBLE"
