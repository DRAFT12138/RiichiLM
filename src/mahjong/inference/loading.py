"""Read-only inference loading through the project's checkpoint protocol."""
from pathlib import Path
from dataclasses import replace

import torch

from mahjong.contracts import MahjongError
from mahjong.models.qwen3 import (
    POLICY_INPUT_SCHEMAS, SETTINGS_FIELDS, Qwen3PolicyAdapter, build_qwen3,
)
from mahjong.training.checkpoint import FileCheckpointStore
from mahjong.policy.shared_context import MODEL_IMPLEMENTATION


def load_policy_checkpoint(path, *, device='cpu'):
    if device not in ('cpu', 'cuda:0') or (device == 'cuda:0' and not torch.cuda.is_available()):
        raise MahjongError('DEVICE_UNAVAILABLE', f'FP32 inference device unavailable: {device}')
    path = Path(path).resolve()
    store = FileCheckpointStore()
    ref = store.resolve(path)
    snapshot = store.load(ref, run_dir=path.parent.parent)
    descriptors = [d for d in snapshot.descriptors if d.kind == 'model']
    if (len(descriptors) != 1 or descriptors[0].implementation_id not in ('majang.qwen3_legal_ntp/v1', MODEL_IMPLEMENTATION)
            or descriptors[0].input_schema not in POLICY_INPUT_SCHEMAS):
        raise MahjongError('CHECKPOINT_INVALID', 'checkpoint is not a legal-action NTP model')
    try:
        settings = {k: snapshot.config['model'][k] for k in SETTINGS_FIELDS}
        if settings['name'] != 'qwen3' or settings['input_schema'] != descriptors[0].input_schema:
            raise ValueError('model configuration task mismatch')
        adapter = Qwen3PolicyAdapter
        if descriptors[0].implementation_id == MODEL_IMPLEMENTATION:
            from mahjong.models.qwen3_shared import Qwen3SharedPolicyAdapter
            if not snapshot.config.get('data',{}).get('shared_context',False):
                raise ValueError('shared model requires an explicit shared training binding')
            adapter = Qwen3SharedPolicyAdapter
        elif snapshot.config.get('data',{}).get('shared_context',False):
            raise ValueError('independent model conflicts with shared training binding')
        model = adapter(build_qwen3(settings, init_seed=0), input_schema=settings['input_schema'])
        expected_descriptor = model.descriptor()
        # Older FP32 checkpoints may advertise a subset of today's execution
        # modes; task/config/objective fingerprints must still match exactly.
        if (not descriptors[0].supported_execution
                or not set(descriptors[0].supported_execution) <= set(expected_descriptor.supported_execution)
                or replace(expected_descriptor, supported_execution=descriptors[0].supported_execution) != descriptors[0]):
            raise ValueError('model/codec/objective/config descriptor mismatch')
        for field, expected in [('pad_token_id', 0), ('bos_token_id', 1), ('eos_token_id', 2)]:
            if settings[field] not in (None, expected):
                raise ValueError(f'incompatible {field}')
        state, expected = snapshot.runtime.model, model.state_dict()
        if state.keys() != expected.keys():
            raise ValueError('model state keys differ')
        for key, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape
                    or tensor.dtype != expected[key].dtype or not torch.isfinite(tensor).all()):
                raise ValueError(f'model state shape/dtype/finite mismatch: {key}')
        if settings['tie_word_embeddings'] and not torch.equal(
                state['causal_lm.model.embed_tokens.weight'], state['causal_lm.lm_head.weight']):
            raise ValueError('shared embedding/head values differ')
        model.load_state_dict(state, strict=True)
        model.load_auxiliary_state_dict(snapshot.runtime.auxiliary)
        source = {'checkpoint_id': ref.checkpoint_id, 'manifest_hash': ref.manifest_hash,
                  'run_id': ref.run_id, 'attempt_id': ref.attempt_id, 'global_step': ref.global_step, 'training_data': snapshot.config.get('data'),
                  'training_data_fingerprint': next((d.data_fingerprint for d in snapshot.descriptors if d.kind=='data'), None),
                  'training_rules_status': 'unverified: checkpoint does not embed an environment registry; verify the frozen source build'}
        del state, snapshot
        return model.to(device).eval(), source
    except (KeyError, ValueError, RuntimeError) as exc:
        raise MahjongError('CHECKPOINT_INVALID', f'incompatible NTP checkpoint: {exc}') from exc
