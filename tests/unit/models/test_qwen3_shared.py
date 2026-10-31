"""Compare the delivered shared graph to independent Qwen3 and legal NTP."""
import copy
from pathlib import Path

import pytest
import torch

from mahjong.app.assembly import build_model, build_runtime
from mahjong.app.config import parse_new
from mahjong.contracts import Batch, MahjongError
from mahjong.data.mahjong.policy import _collate
from mahjong.data.mahjong.shared_policy import collate_shared_group
from mahjong.models.qwen3 import Qwen3PolicyAdapter
from mahjong.policy.shared_context import INDEPENDENT_LAYOUT
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA


def config(device='cpu', precision='32-true'):
    return parse_new(['model.num_hidden_layers=2','model.hidden_size=32',
        'model.intermediate_size=64','model.num_attention_heads=4',
        'model.num_key_value_heads=2','model.head_dim=8',
        'model.max_position_embeddings=128','data.shared_max_tokens=128',
        f'runtime.device={device}',f'runtime.precision={precision}'],
        Path.cwd(),config_name='pretrain')


def samples():
    trunk = [11,12,13,14,15,16,17]
    actions = [[[1,4,6],[1,5,6],[3,6]], [[1,6],[3,6]], [[1,4,5,6],[3,6]]]
    return [dict(input_ids=trunk[:cut]+[31,40+i,35], cut_length=cut,
        candidates=legal, target_index=0, sample_id=f'decision-{i}',
        group_key=('source',0,0),window_id=i)
        for i,(cut,legal) in enumerate(zip([4,4,7],actions))]


def selected(model,batch):
    p = model.prepare_batch(batch,torch.device('cpu')).payload
    return model.causal_lm.lm_head(model._hidden(p)[0,p['query_nodes']])


@pytest.mark.parametrize('size',[1,2,3])
def test_logits_loss_all_parameter_gradients_and_adam_match_independent(size):
    torch.set_num_threads(1)
    shared = build_model(config())
    independent = Qwen3PolicyAdapter(copy.deepcopy(shared.causal_lm),input_schema=CONTEXT_SCHEMA)
    group=samples()[:size]
    batch=collate_shared_group(group)
    prepared=shared.prepare_batch(batch,torch.device('cpu')).payload
    logits=shared.causal_lm.lm_head(shared._hidden(prepared)[0,prepared['query_nodes']])
    reference=[]
    for sample in group:
        action=sample['candidates'][sample['target_index']]
        prefix=sample['input_ids']
        ids=torch.tensor([prefix+action])
        reference.append(independent.causal_lm(ids,use_cache=False).logits[0,
            len(prefix)-1:len(prefix)+len(action)-1])
    torch.testing.assert_close(logits,torch.cat(reference),atol=1e-6,rtol=1e-5)
    a=shared(batch,phase='train'); b=independent(_collate(group,CONTEXT_SCHEMA),phase='train')
    torch.testing.assert_close(a.loss,b.loss,atol=1e-6,rtol=1e-5)
    assert a.metrics['ntp_loss'].denominator==sum(len(s['candidates'][0]) for s in group)
    assert a.metrics['token_accuracy']==b.metrics['token_accuracy']
    a.loss.backward(); b.loss.backward()
    sq=norm=0.0
    for name,param in shared.named_parameters():
        other=dict(independent.named_parameters())[name]
        assert param.grad is not None and other.grad is not None
        assert torch.isfinite(param.grad).all()
        torch.testing.assert_close(param.grad,other.grad,atol=1e-6,rtol=1e-4)
        sq+=float((param.grad-other.grad).square().sum())
        norm+=float(other.grad.square().sum())
    assert (sq/norm)**0.5<1e-5
    for model in (shared,independent):
        torch.optim.Adam(model.parameters(),lr=1e-4,foreach=False,fused=False).step()
    for name,param in shared.named_parameters():
        torch.testing.assert_close(param,dict(independent.named_parameters())[name],atol=2e-6,rtol=1e-5)
    assert shared.causal_lm.lm_head.weight is shared.causal_lm.model.embed_tokens.weight


def test_segmented_future_and_other_branch_isolation_and_wrong_positions():
    model=build_model(config())
    batch=collate_shared_group(samples())
    base=selected(model,batch)
    future={**batch.payload,'trunk_ids':batch.payload['trunk_ids'].clone()}
    future['trunk_ids'][-1]=25
    actual=selected(model,Batch(future,batch.sample_ids,batch.sample_count,batch.input_schema))
    torch.testing.assert_close(actual[:5],base[:5],rtol=0,atol=0)
    other={**batch.payload,'private_ids':batch.payload['private_ids'].clone()}
    other['private_ids'][1]=55
    actual=selected(model,Batch(other,batch.sample_ids,batch.sample_count,batch.input_schema))
    torch.testing.assert_close(actual[3:],base[3:],rtol=0,atol=0)
    p=model.prepare_batch(batch,torch.device('cpu')).payload
    wrong={**p,'position_ids':torch.arange(p['input_ids'].shape[1])[None]}
    incorrect=model.causal_lm.lm_head(model._hidden(wrong)[0,p['query_nodes']])
    assert float((incorrect-base).abs().max().detach())>1e-5


def test_segmented_wrong_triangular_mask_is_a_detected_negative_control(monkeypatch):
    from torch.nn import functional as F
    model=build_model(config()); batch=collate_shared_group(samples())
    base=selected(model,batch)
    original=F.scaled_dot_product_attention
    def wrong(q,k,v,**kwargs):
        if not kwargs['is_causal']:
            kwargs['attn_mask']=torch.ones_like(kwargs['attn_mask']).tril()
        return original(q,k,v,**kwargs)
    monkeypatch.setattr(F,'scaled_dot_product_attention',wrong)
    altered=selected(model,batch)
    assert float((base-altered).abs().max().detach())>1e-5


def test_explicit_independent_evaluation_preserves_metrics_and_rejects_inferred_layout():
    model=build_model(config())
    independent=_collate(samples(),CONTEXT_SCHEMA)
    with pytest.raises(MahjongError,match='layout'):
        model(independent,phase='eval')
    explicit=Batch({**independent.payload,'layout':INDEPENDENT_LAYOUT},
        independent.sample_ids,independent.sample_count,independent.input_schema)
    a=model(explicit,phase='eval')
    b=model(collate_shared_group(samples()),phase='eval')
    torch.testing.assert_close(a.loss,b.loss,atol=1e-6,rtol=1e-5)
    assert a.metrics['token_accuracy']==b.metrics['token_accuracy']
    assert a.metrics['ntp_loss'].denominator==9


@pytest.mark.parametrize('field,value',[
    ('cut_lengths',torch.tensor([0,4,7])),
    ('branch_offsets',torch.tensor([0,4,3,9])),
    ('target_indices',torch.tensor([99,0,0])),
    ('private_ids',torch.tensor([9999]*9)),
])
def test_malformed_structural_payload_is_rejected(field,value):
    model=build_model(config()); batch=collate_shared_group(samples())
    bad=Batch({**batch.payload,field:value},batch.sample_ids,batch.sample_count,batch.input_schema)
    with pytest.raises(MahjongError,match='CONTRACT_MISMATCH'):
        model.prepare_batch(bad,torch.device('cpu'))


DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('precision',['32-true','bf16-mixed'])
def test_fabric_update_optimizer_rng_and_next_update_roundtrip(device,precision):
    cfg=config(device,precision)
    runtime=build_runtime(cfg,training=True)
    batch=collate_shared_group(samples())
    runtime.seed_training(123)
    def update(target):
        target.zero_grad(); result=target.compute(batch,phase='train')
        target.backward(result.loss); target.check_gradients(); target.step()
        return result
    result=update(runtime)
    assert result.metrics['ntp_loss'].denominator==9
    state,rng=runtime.export_state(),runtime.capture_rng()
    restored=build_runtime(cfg,training=True)
    restored.import_state(state); restored.restore_rng(rng)
    update(runtime); update(restored)
    for name,param in runtime.raw_model.named_parameters():
        torch.testing.assert_close(param,dict(restored.raw_model.named_parameters())[name],rtol=0,atol=0)
    a,b=runtime.export_state().optimizer,restored.export_state().optimizer
    for key,values in a['state'].items():
        for field,value in values.items():
            torch.testing.assert_close(value,b['state'][key][field],rtol=0,atol=0)
