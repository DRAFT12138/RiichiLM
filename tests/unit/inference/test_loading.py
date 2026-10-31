from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from mahjong.app.config import parse_model
from mahjong.app.assembly import build_model
from mahjong.contracts import MahjongError, canonical_json
from mahjong.inference import PolicyInference
from mahjong.models.qwen3 import Qwen3PolicyAdapter
from mahjong.training.checkpoint import FileCheckpointStore
from tests.unit.training.storage.test_checkpoint import make_run, make_attempt, snapshot_for


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    config=parse_model(['model.input_schema=majang-policy-ntp/v1',
        'model.num_hidden_layers=1','model.hidden_size=32','model.intermediate_size=64',
        'model.num_attention_heads=4','model.num_key_value_heads=2','model.head_dim=8',
        'model.vocab_size=1024','model.max_position_embeddings=256'])
    model=build_model(config)
    context=make_attempt(make_run(tmp_path))
    snapshot=snapshot_for(context)
    snapshot=replace(snapshot, config=config,
        frozen_config_hash=hashlib.sha256(canonical_json(config).encode()).hexdigest(),
        descriptors=tuple(model.descriptor() if d.kind=='model' else d for d in snapshot.descriptors),
        runtime=replace(snapshot.runtime, model=dict(model.state_dict()), auxiliary={}))
    return context,snapshot,model


def save(checkpoint, mutate=lambda s:s):
    context,snapshot,_=checkpoint
    snapshot=mutate(snapshot)
    ref=FileCheckpointStore().save(snapshot,context)
    return context.run_dir/ref.path


def test_strict_offline_weight_only_load_does_not_use_training_components(checkpoint, monkeypatch):
    path=save(checkpoint)
    import mahjong.app.assembly
    monkeypatch.setattr(mahjong.app.assembly,'build_train_source',lambda *_:pytest.fail('training source built'))
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',lambda *_:pytest.fail('network accessed'))
    inference=PolicyInference.from_checkpoint(path)
    assert not inference.model.training
    for key,value in checkpoint[2].state_dict().items():
        torch.testing.assert_close(inference.model.state_dict()[key],value,rtol=0,atol=0)
    assert inference.lm.config.use_cache is False
    assert inference.lm.config.eos_token_id is None
    assert inference.lm.get_input_embeddings().weight is inference.lm.get_output_embeddings().weight


@pytest.mark.parametrize('kind',['old_classifier','schema','fingerprint','context','missing_head','dtype','shared_head'])
def test_incompatible_binding_or_state_rejected(checkpoint,kind):
    def mutate(s):
        descriptors=list(s.descriptors)
        i=next(i for i,d in enumerate(descriptors) if d.kind=='model')
        d=descriptors[i]
        if kind=='old_classifier':descriptors[i]=replace(d,implementation_id='majang.qwen3_policy/v1')
        elif kind=='schema':descriptors[i]=replace(d,input_schema='majang-policy/v1')
        elif kind=='fingerprint':descriptors[i]=replace(d,config_fingerprint='0'*64)
        elif kind=='context':
            config={**s.config,'model':{**s.config['model'],'max_position_embeddings':257}}
            return replace(s,config=config)
        else:
            state=dict(s.runtime.model)
            if kind=='missing_head':state.pop('causal_lm.lm_head.weight')
            elif kind=='dtype':state['causal_lm.lm_head.weight']=state['causal_lm.lm_head.weight'].half()
            else:state['causal_lm.lm_head.weight']=state['causal_lm.lm_head.weight'].clone()+1
            return replace(s,runtime=replace(s.runtime,model=state))
        return replace(s,descriptors=tuple(descriptors))
    with pytest.raises(MahjongError,match='CHECKPOINT_INVALID'):
        PolicyInference.from_checkpoint(save(checkpoint,mutate))


def test_corruption_bare_hf_path_and_unavailable_device_are_explicit(checkpoint):
    path=save(checkpoint)
    state=path/'state.pt'
    with state.open('r+b') as f:
        f.seek(100);f.write(b'corrupt')
    with pytest.raises(MahjongError,match='SHA256'):
        PolicyInference.from_checkpoint(path)
    with pytest.raises(MahjongError,match='CHECKPOINT_INVALID'):
        PolicyInference.from_checkpoint(path.parent)
    with pytest.raises(MahjongError,match='DEVICE_UNAVAILABLE'):
        PolicyInference.from_checkpoint(path,device='cuda:17')


@pytest.mark.parametrize('previous_rounds', [3, 4])
def test_cross_round_checkpoint_preserves_the_bound_history_window(checkpoint, monkeypatch, previous_rounds):
    from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, CONTEXT_BINDING
    context, snapshot, base_model = checkpoint
    config = {**snapshot.config, 'model': {**snapshot.config['model'], 'input_schema': CONTEXT_SCHEMA}}
    model = Qwen3PolicyAdapter(base_model.causal_lm, input_schema=CONTEXT_SCHEMA)
    with monkeypatch.context() as patch:
        patch.setitem(CONTEXT_BINDING, 'previous_rounds', previous_rounds)
        descriptor = model.descriptor()
    snapshot = replace(snapshot, config=config,
        frozen_config_hash=hashlib.sha256(canonical_json(config).encode()).hexdigest(),
        descriptors=tuple(descriptor if d.kind=='model' else d for d in snapshot.descriptors))
    path = save((context, snapshot, model))
    if previous_rounds == 4:
        with pytest.raises(MahjongError, match='CHECKPOINT_INVALID'):
            PolicyInference.from_checkpoint(path)
    else:
        loaded = PolicyInference.from_checkpoint(path)
        assert loaded.model.descriptor() == descriptor


@pytest.mark.parametrize('invalid_binding',[False,True])
def test_shared_checkpoint_formal_load_preserves_weights_and_requires_shared_binding(checkpoint,invalid_binding):
    from mahjong.models.qwen3_shared import Qwen3SharedPolicyAdapter
    from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
    context,snapshot,base=checkpoint
    config={**snapshot.config,'model':{**snapshot.config['model'],'input_schema':CONTEXT_SCHEMA},
            'data':{'shared_context':not invalid_binding}}
    model=Qwen3SharedPolicyAdapter(base.causal_lm)
    snapshot=replace(snapshot,config=config,
        frozen_config_hash=hashlib.sha256(canonical_json(config).encode()).hexdigest(),
        descriptors=tuple(model.descriptor() if d.kind=='model' else d for d in snapshot.descriptors))
    path=save((context,snapshot,model))
    if invalid_binding:
        with pytest.raises(MahjongError,match='shared training binding'):
            PolicyInference.from_checkpoint(path)
    else:
        loaded=PolicyInference.from_checkpoint(path)
        assert loaded.model.descriptor()==model.descriptor()
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,loaded.model.state_dict()[name],rtol=0,atol=0)
