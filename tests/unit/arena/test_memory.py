from types import SimpleNamespace

import pytest
import torch

from mahjong.app.cli import parser
from mahjong.arena.memory import ModelMemoryBudget, GIB
from mahjong.contracts import MahjongError
from mahjong.web.session import Session


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_invalid_budget_is_rejected_on_cpu_and_web(value, tmp_path):
    with pytest.raises(MahjongError, match='finite positive'):
        ModelMemoryBudget('cpu', value)
    with pytest.raises(MahjongError, match='finite positive'):
        Session(tmp_path, gpu_memory_gib=value)


def test_cli_defaults_and_custom_budget():
    assert parser().parse_args(['web-play']).gpu_memory_gib == 16
    args = ['arena-evaluate', '--players', *['random']*4, '--output', 'out.json']
    assert parser().parse_args(args).gpu_memory_gib == 16
    assert parser().parse_args(args + ['--gpu-memory-gib', '8']).gpu_memory_gib == 8


def test_cpu_budget_never_initializes_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *a: pytest.fail('GPU queried on CPU'))
    budget = ModelMemoryBudget('cpu', 16)
    budget.refresh()
    assert budget.snapshot() == dict(requested_gib=16., effective_gib=None)


def test_dynamic_budget_counts_owned_reservations_and_adapts_to_free_vram(monkeypatch):
    state = dict(free=20*GIB, total=24*GIB, reserved=2*GIB, allocated=GIB)
    fractions = []
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda d: (state['free'], state['total']))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda d: state['reserved'])
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda d: state['allocated'])
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda f,d: fractions.append(f))
    budget = ModelMemoryBudget('cuda:0')
    budget.refresh()
    assert budget.limit_bytes == 16*GIB
    state['free'] = 4*GIB
    budget.refresh()
    assert budget.limit_bytes == 5*GIB  # 4 free + 2 owned - 1 driver margin
    state['free'] = 20*GIB
    budget.refresh()
    assert budget.limit_bytes == 16*GIB
    assert fractions == pytest.approx([16/24, 5/24, 16/24])


def test_memory_pressure_evicts_prefix_caches_in_lru_order(monkeypatch):
    state = dict(allocated=8*GIB)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda d: (12*GIB, 24*GIB))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda d: 8*GIB)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda d: state['allocated'])
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    class Device:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    monkeypatch.setattr(torch.cuda, 'device', lambda d: Device())
    calls = []
    class Cache:
        def __init__(self, age): self.last_used = age
        def clear(self, **kwargs):
            calls.append(self.last_used)
            state['allocated'] -= 2*GIB
    a, b = Cache(1), Cache(2)
    budget = ModelMemoryBudget('cuda:0', 5)
    budget.register(b); budget.register(a)
    budget.refresh()
    assert calls == [1, 2] and state['allocated'] == 4*GIB
    state['allocated'] = 9*GIB
    budget = ModelMemoryBudget('cuda:0', 5)
    with pytest.raises(MahjongError, match='exceeds'):
        budget.refresh()
