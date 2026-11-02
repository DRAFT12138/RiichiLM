import json

import pytest
import torch

from mahjong.contracts import MahjongError
from mahjong.web.catalog import Catalog
from mahjong.web.worker import TableEngine
from tests.unit.inference.test_weights import weights


def test_custom_catalog_lists_without_loading_and_detects_edits(tmp_path, monkeypatch):
    root = tmp_path / 'weights'
    root.mkdir()
    path = root / 'mine.pt'
    path.write_bytes(b'listing never opens torch files')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('loaded during listing'))
    c = Catalog(tmp_path / 'no-training-runs', weights_root=root)
    row, = c.data['runs']
    item, = row['checkpoints']
    assert row['experiment'] == 'mine' and item['selectable']
    assert c.resolve(item['id'])['kind'] == 'weights'
    path.with_suffix('.json').write_text(json.dumps({'max_position_embeddings': 256}))
    with pytest.raises(MahjongError):
        c.resolve(item['id'])
    c.refresh()
    new = c.data['runs'][0]['checkpoints'][0]
    path.unlink()
    with pytest.raises(MahjongError):
        c.resolve(new['id'])


def test_custom_directory_does_not_follow_outside_weights_or_sidecars(tmp_path):
    root = tmp_path / 'weights'
    root.mkdir()
    outside = tmp_path / 'outside.pt'
    outside.write_bytes(b'outside')
    (root / 'escape.pt').symlink_to(outside)
    (root / 'local.pt').write_bytes(b'local')
    (root / 'local.json').symlink_to(outside)
    c = Catalog(tmp_path / 'outputs', weights_root=root)
    assert not c.data['runs'] and len(c.data['errors']) == 3


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='GPU unavailable'))])
def test_custom_weights_play_without_training_data_and_share_model_per_table(weights, tmp_path, device):
    if device == 'cuda:0' and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 GPU unavailable')
    path, _, _ = weights
    # GPU execution must also handle a custom file requesting eager attention.
    if device == 'cuda:0':
        path.with_suffix('.json').write_text(json.dumps({'attn_implementation': 'eager'}))
    catalog = Catalog(tmp_path / 'outputs', weights_root=path.parent)
    cp = catalog.data['runs'][0]['checkpoints'][0]
    entry = catalog.resolve(cp['id'])
    engine = TableEngine(dict(mode='single', opponents=[entry] * 4,
                             spectator=True, seed=42), device=device)
    policy = engine.controllers[0][1]
    from mahjong.policy.execution import policy_precision
    assert policy_precision(policy.model) == ('bf16-mixed' if device == 'cuda:0' else '32-true')
    assert policy.lm.config._attn_implementation == 'sdpa'
    assert all(p.dtype == torch.float32 for p in policy.model.parameters())
    if device == 'cuda:0':
        from mahjong.inference.policy import PolicyInference
        fp32 = PolicyInference.from_weights(path, device=device)
        assert fp32.binding_digest != policy.binding_digest
    assert len({id(p) for _, p in engine.controllers}) == 1
    assert len({p.generator.initial_seed() for p in engine.loop.players}) == 4
    for _ in range(4):
        engine.drive(force=True)
    table = engine.publish()
    assert table['state'] == 'playing'
    assert sum(engine.loop.decisions) > 0
    assert all(c['sampling'] == 'categorical' for c in table['controllers'])
