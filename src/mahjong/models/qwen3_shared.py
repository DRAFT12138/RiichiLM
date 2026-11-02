"""Differentiable shared-trunk Qwen3 execution within one optimizer update."""
from dataclasses import replace
import hashlib
import math

import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from mahjong.contracts import Batch, MahjongError, MetricStat, StepResult, canonical_json
from mahjong.policy.constraints import build_trie
from mahjong.policy.shared_context import (
    INDEPENDENT_LAYOUT, MODEL_IMPLEMENTATION, PREPARED_LAYOUT, SHARED_LAYOUT,
)
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
from .qwen3 import Qwen3PolicyAdapter

# Conservative cold-start amortization for the measured default-model regions.
MIN_COMPILE_UPDATES = 1000


class Qwen3SharedPolicyAdapter(Qwen3PolicyAdapter):
    def __init__(self, causal_lm, *, input_schema=CONTEXT_SCHEMA):
        super().__init__(causal_lm, input_schema=input_schema)
        config = causal_lm.config
        if (input_schema != CONTEXT_SCHEMA or config._attn_implementation != 'sdpa' or config.attention_dropout != 0 or
                config.rope_parameters.get('rope_type') != 'default' or
                any(t != 'full_attention' for t in config.layer_types)):
            raise MahjongError('CONFIG_INVALID', 'shared policy requires v2, SDPA, zero dropout, default RoPE and full attention')
        self._compile_training = False
        self._training_kernels = None

    def configure_training_execution(self, remaining_updates):
        self._compile_training = remaining_updates >= MIN_COMPILE_UPDATES

    def supports_cpu_lookahead(self):
        if (type(self) is not Qwen3SharedPolicyAdapter or not self.training or
                any(getattr(getattr(self, name), '__func__', getattr(self, name)) is not original
                    for name, original in _CPU_PREPARATION_METHODS.items())):
            return False
        from .shared_kernels import _has_hooks
        return not any(_has_hooks(module) for module in self.modules())

    @staticmethod
    def transfer_prepared_batch(batch, device):
        """Move model-owned prepared tensors without rebuilding the CPU layout."""
        return Batch({k:v.to(device) if isinstance(v,torch.Tensor) else v
                      for k,v in batch.payload.items()}, batch.sample_ids,
                     batch.sample_count, batch.input_schema)

    def descriptor(self):
        base = super().descriptor()
        fingerprint = hashlib.sha256(canonical_json({'independent':base.config_fingerprint,
            'layout':SHARED_LAYOUT, 'execution':'segmented-sdpa/1',
            'eval_layout':INDEPENDENT_LAYOUT}).encode()).hexdigest()
        return replace(base, implementation_id=MODEL_IMPLEMENTATION, config_fingerprint=fingerprint)

    @staticmethod
    def _invalid(reason):
        raise MahjongError('CONTRACT_MISMATCH', f'Qwen3 shared policy payload: {reason}', 'training')

    def _independent(self, batch):
        if (type(batch.payload) is not dict or batch.payload.get('layout') != INDEPENDENT_LAYOUT):
            self._invalid('explicit independent layout required')
        return Batch({k:v for k,v in batch.payload.items() if k != 'layout'}, batch.sample_ids,
                     batch.sample_count, batch.input_schema)

    def prepare_batch(self, batch, device):
        from .nanikiru import is_nanikiru, prepare_nanikiru
        if is_nanikiru(batch):
            return prepare_nanikiru(self, batch)
        if type(batch.payload) is not dict or batch.input_schema != self.input_schema:
            self._invalid('expected v2 mapping')
        layout = batch.payload.get('layout')
        if layout == INDEPENDENT_LAYOUT:
            prepared = super().prepare_batch(self._independent(batch), device)
            return Batch({**prepared.payload, 'layout':INDEPENDENT_LAYOUT}, prepared.sample_ids,
                         prepared.sample_count, prepared.input_schema)
        if layout != SHARED_LAYOUT:
            self._invalid('expected explicit shared or independent layout')
        p = batch.payload
        expected = {'layout','trunk_ids','private_ids','branch_offsets','cut_lengths',
                    'candidate_ids','target_indices'}
        if p.keys() != expected:
            self._invalid('unexpected shared fields')
        values = {key:value.detach().cpu() if isinstance(value, torch.Tensor) else value
                  for key,value in p.items() if key != 'layout'}
        for key in ('trunk_ids','private_ids','branch_offsets','cut_lengths'):
            value = values[key]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 1:
                self._invalid(f'{key} must be int64 [N]')
        trunk, private = values['trunk_ids'].tolist(), values['private_ids'].tolist()
        offsets, cuts = values['branch_offsets'].tolist(), values['cut_lengths'].tolist()
        m, P = batch.sample_count, len(trunk)
        if (not P or len(cuts) != m or len(offsets) != m+1 or offsets[0] != 0 or
                offsets[-1] != len(private) or any(a >= b for a,b in zip(offsets, offsets[1:])) or
                any(not 0 < c <= P for c in cuts) or max(cuts) != P):
            self._invalid('invalid branch offsets or history cuts')
        prefixes = [trunk[:cut] + private[lo:hi] for cut,lo,hi in zip(cuts,offsets,offsets[1:])]
        width = max(map(len, prefixes))
        ids = torch.zeros((m,width), dtype=torch.int64)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row,prefix in enumerate(prefixes):
            ids[row,:len(prefix)] = torch.tensor(prefix, dtype=torch.int64)
            mask[row,:len(prefix)] = True
        independent = Batch({'input_ids':ids, 'attention_mask':mask,
            'candidate_ids':values['candidate_ids'], 'target_indices':values['target_indices']},
            batch.sample_ids,m,batch.input_schema)
        payload, active = self._payload(independent)
        # The existing validator owns candidate padding, IDs and target bounds.
        candidates = payload['candidate_ids'].tolist()
        lengths = active.sum(-1).tolist()
        targets = payload['target_indices'].tolist()
        flat, positions, branch_starts = list(trunk), list(range(P)), [P]
        queries, labels, allowed = [], [], []
        for row,(cut,lo,hi) in enumerate(zip(cuts,offsets,offsets[1:])):
            actions = [a[:n] for a,n in zip(candidates[row],lengths[row]) if n]
            if len(prefixes[row]) + max(map(len,actions)) > self.causal_lm.config.max_position_embeddings:
                self._invalid('prefix + longest legal action exceeds context')
            action = actions[targets[row]]
            node = build_trie(actions)
            start = len(flat)
            suffix = private[lo:hi] + action
            flat.extend(suffix); positions.extend(range(cut,cut+len(suffix)))
            for j,token in enumerate(action):
                queries.append(start + hi-lo-1+j)
                labels.append(token); allowed.append(tuple(node)); node = node[token]
            branch_starts.append(len(flat))
        if len(flat) > self.causal_lm.config.max_position_embeddings:
            self._invalid('physical token budget exceeds configured context')
        legal_mask = torch.zeros((len(labels),self.causal_lm.config.vocab_size), dtype=torch.bool)
        for row,legal in enumerate(allowed):
            legal_mask[row,list(legal)] = True
        tensor_payload = {'layout':PREPARED_LAYOUT,
            'input_ids':torch.tensor([flat],dtype=torch.int64),
            'position_ids':torch.tensor([positions],dtype=torch.int64),
            'branch_offsets':torch.tensor(branch_starts,dtype=torch.int64),
            'cut_lengths':torch.tensor(cuts,dtype=torch.int64),
            'query_nodes':torch.tensor(queries,dtype=torch.int64),
            'target_ids':torch.tensor(labels,dtype=torch.int64), 'legal_mask':legal_mask,
            'trunk_length':P, 'logical_tokens':sum(map(len,prefixes))+len(labels)}
        return Batch({k:v.to(device) if isinstance(v,torch.Tensor) else v
                      for k,v in tensor_payload.items()}, batch.sample_ids,m,batch.input_schema)

    def _hidden(self, payload, *, fused=False):
        """Two attention calls per layer, with no persistent cache or detach."""
        backbone = self.causal_lm.model
        kernels = None
        if fused:
            if self._training_kernels is None:
                from .shared_kernels import SharedTrainingKernels
                self._training_kernels = SharedTrainingKernels(backbone.embed_tokens.weight.device)
            kernels = self._training_kernels
        P = payload['trunk_length']
        hidden = backbone.embed_tokens(payload['input_ids'])
        positions = payload['position_ids']
        cos, sin = backbone.rotary_emb(hidden, positions)
        starts, cuts = payload['branch_offsets'], payload['cut_lengths']
        branch_positions = torch.arange(hidden.shape[1]-P, device=hidden.device)
        # Bucketize with right=True assigns a boundary query to its next branch.
        owners = torch.bucketize(branch_positions+P, starts[1:-1], right=True)
        key_positions = torch.arange(hidden.shape[1], device=hidden.device)
        query_positions = branch_positions+P
        allowed = ((key_positions[None,:] < cuts[owners,None]) |
            ((key_positions[None,:] >= starts[owners,None]) &
             (key_positions[None,:] <= query_positions[:,None])))
        for layer in backbone.layers:
            if kernels and kernels.supports_layer(layer):
                hidden = kernels.layer(layer, hidden, cos, sin, P, allowed)
                continue
            residual = hidden
            normalized = (kernels.norm(layer.input_layernorm, hidden) if kernels
                          else layer.input_layernorm(hidden))
            attention = layer.self_attn
            shape = (*normalized.shape[:-1], -1, attention.head_dim)
            q = attention.q_proj(normalized).view(shape)
            k = attention.k_proj(normalized).view(shape)
            q = (kernels.norm(attention.q_norm, q) if kernels else attention.q_norm(q)).transpose(1,2)
            k = (kernels.norm(attention.k_norm, k) if kernels else attention.k_norm(k)).transpose(1,2)
            v = attention.v_proj(normalized).view(shape).transpose(1,2)
            q,k = kernels.rotary(q,k,cos,sin) if kernels else apply_rotary_pos_emb(q,k,cos,sin)
            k,v = (repeat_kv(x,attention.num_key_value_groups) for x in (k,v))
            common = F.scaled_dot_product_attention(q[:,:,:P], k[:,:,:P], v[:,:,:P],
                is_causal=True, dropout_p=0.0, scale=attention.scaling)
            branches = F.scaled_dot_product_attention(q[:,:,P:], k,v,
                attn_mask=allowed[None,None], is_causal=False, dropout_p=0.0, scale=attention.scaling)
            combined = torch.cat((common,branches),dim=2).transpose(1,2).contiguous()
            combined = combined.reshape(*hidden.shape[:-1],-1)
            hidden = residual + attention.o_proj(combined)
            normalized = (kernels.norm(layer.post_attention_layernorm, hidden) if kernels
                          else layer.post_attention_layernorm(hidden))
            hidden = hidden + (kernels.mlp(layer.mlp, normalized) if kernels else layer.mlp(normalized))
        return (kernels.norm(backbone.norm, hidden) if kernels and not kernels.layer_enabled
                else backbone.norm(hidden))

    def forward(self, batch, *, phase):
        from .nanikiru import evaluate_nanikiru, is_nanikiru
        if is_nanikiru(batch):
            return evaluate_nanikiru(self, batch, phase=phase)
        if phase not in ('train','eval'):
            self._invalid('phase must be train/eval')
        if type(batch.payload) is not dict:
            self._invalid('expected mapping')
        layout = batch.payload.get('layout')
        if layout == INDEPENDENT_LAYOUT:
            return super().forward(self._independent(batch), phase=phase)
        if layout == SHARED_LAYOUT:
            batch = self.prepare_batch(batch, self.causal_lm.model.embed_tokens.weight.device)
        elif layout != PREPARED_LAYOUT:
            self._invalid('unsupported execution layout')
        p = batch.payload
        expected = {'layout','input_ids','position_ids','branch_offsets','cut_lengths','query_nodes',
                    'target_ids','legal_mask','trunk_length','logical_tokens'}
        if batch.input_schema != self.input_schema or p.keys() != expected:
            self._invalid('invalid prepared layout fields/schema')
        if (not isinstance(p['input_ids'],torch.Tensor) or p['input_ids'].dtype != torch.int64 or
                p['input_ids'].ndim != 2 or p['input_ids'].shape[0] != 1 or
                p['position_ids'].shape != p['input_ids'].shape or
                type(p['trunk_length']) is not int or not 0 < p['trunk_length'] < p['input_ids'].shape[1] or
                p['input_ids'].shape[1] > self.causal_lm.config.max_position_embeddings or
                p['branch_offsets'].shape != (batch.sample_count+1,) or
                p['cut_lengths'].shape != (batch.sample_count,) or
                p['query_nodes'].shape != p['target_ids'].shape or
                p['legal_mask'].shape != (p['target_ids'].numel(),self.causal_lm.config.vocab_size)):
            self._invalid('invalid prepared tensor shapes')
        fused = (self._compile_training and phase == 'train' and self.training and
                 p['input_ids'].device.type == 'cuda' and torch.is_grad_enabled() and
                 torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16)
        hidden = self._hidden(p, fused=fused)
        logits = self.causal_lm.lm_head(hidden[0,p['query_nodes']]).float()
        logits = logits.masked_fill(~p['legal_mask'], -torch.inf)
        loss = F.cross_entropy(logits,p['target_ids'])
        correct = (logits.argmax(-1)==p['target_ids']).sum()
        if loss.device.type == 'cuda':
            # One readback for both metrics and the model's finite-loss check.
            loss_value, correct = torch.stack((loss.detach(), correct.to(loss.dtype))).cpu().tolist()
        else:
            loss_value, correct = float(loss.detach()), int(correct)
        if not math.isfinite(loss_value):
            raise MahjongError('NONFINITE_LOSS', 'shared Qwen3 produced nonfinite loss', 'training')
        count = p['target_ids'].numel()
        metric = MetricStat(loss_value*count,count,'nats')
        return StepResult(loss, {'loss':metric, 'ntp_loss':metric,
            'token_accuracy':MetricStat(int(correct),count,'ratio'),
            'context_reuse':MetricStat(p['logical_tokens'],p['input_ids'].numel(),'ratio')})


_CPU_PREPARATION_METHODS = {name: getattr(Qwen3SharedPolicyAdapter, name) for name in
                          ('prepare_batch', '_payload', '_independent', 'forward',
                           'transfer_prepared_batch')}
