"""Self-contained pure Actor publication and strict read-only loading."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid
import torch

from mahjong.contracts import MahjongError
from mahjong.inference.policy import PolicyInference, digest, EVENT_ADAPTER, ACTION_ADAPTER
from mahjong.models.qwen3 import Qwen3PolicyAdapter, build_qwen3
from mahjong.tokenizer import MahjongTokenizer
from mahjong.tokenizer.codec import BINDING
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, CONTEXT_BINDING
from mahjong.envs.policy_bridge import PROFILE_DIGEST
from mahjong.training.checkpoint import _stream_digest, _write_bytes, _fsync_directory
from mahjong.policy.execution import policy_precision, set_policy_precision

SCHEMA = 'mahjong-actor-snapshot/v1'


def compatibility(actor, profile):
    if digest(profile) != PROFILE_DIGEST or actor.input_schema != CONTEXT_SCHEMA or actor.causal_lm.config.max_position_embeddings != 8192:
        raise MahjongError('CONTRACT_MISMATCH', 'RL requires frozen profile, context schema and 8192 context')
    # Shared/independent training objectives have the same inference weights.
    normalized = Qwen3PolicyAdapter(actor.causal_lm, input_schema=actor.input_schema)
    return json.loads(json.dumps(dict(descriptor=asdict(normalized.descriptor()), profile=profile,
                codec={**BINDING, 'codec_spec_sha256': MahjongTokenizer().codec_spec_sha256},
                context=CONTEXT_BINDING, event_adapter=EVENT_ADAPTER, action_adapter=ACTION_ADAPTER,
                precision=policy_precision(actor))))


def check_state(actor, state):
    expected = actor.state_dict()
    if state.keys() != expected.keys():
        raise MahjongError('CHECKPOINT_INVALID', 'Actor keys differ')
    for key, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype
                or not bool(torch.isfinite(tensor).all())):
            raise MahjongError('CHECKPOINT_INVALID', f'Actor tensor invalid: {key}')
    if actor.causal_lm.config.tie_word_embeddings and not torch.equal(state['causal_lm.model.embed_tokens.weight'], state['causal_lm.lm_head.weight']):
        raise MahjongError('CHECKPOINT_INVALID', 'shared head is inconsistent')


def publish_actor(root, actor, settings, profile, provenance, *, max_files=18, byte_budget=8000000000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    check_state(actor, state)
    evidence = compatibility(actor, profile)
    # Content identity independent of torch.save archive metadata.
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
    artifact_id = h.hexdigest()
    final = root / artifact_id
    if final.exists():
        loaded = load_actor(final)
        if loaded.source['compatibility'] != evidence:
            raise MahjongError('CHECKPOINT_INVALID', 'Actor content has conflicting compatibility')
        return loaded.source['actor_ref']
    existing = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')]
    size = sum(p.stat().st_size for d in existing for p in d.iterdir() if p.is_file())
    estimate = sum(v.numel() * v.element_size() for v in state.values())
    if len(existing) >= max_files or size + estimate > byte_budget:
        raise MahjongError('CHECKPOINT_WRITE_FAILED', 'pure Actor capacity exceeded')
    staging = root / ('.tmp-' + uuid.uuid4().hex)
    staging.mkdir()
    with (staging / 'actor.pt').open('xb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    with (staging / 'actor.pt').open('rb') as f:
        length, sha = _stream_digest(f)
    manifest = dict(schema=SCHEMA, artifact_id=artifact_id, weights_sha256=sha, weights_bytes=length,
                    settings=settings, compatibility=evidence, provenance=provenance)
    encoded = (json.dumps(manifest, sort_keys=True, allow_nan=False) + '\n').encode()
    if size + length + len(encoded) > byte_budget:
        import shutil
        shutil.rmtree(staging)
        raise MahjongError('CHECKPOINT_WRITE_FAILED', 'serialized Actor byte budget exceeded')
    _write_bytes(staging / 'manifest.json', encoded)
    _fsync_directory(staging)
    os.rename(staging, final); _fsync_directory(root)
    loaded = load_actor(final)
    if any(not torch.equal(v, loaded.model.state_dict()[k].cpu()) for k, v in state.items()):
        raise MahjongError('CHECKPOINT_INVALID', 'published Actor tensor readback differs')
    return loaded.source['actor_ref']


def load_actor(path, *, expected=None, device='cpu'):
    path = Path(path)
    if path.parent.name == 'checkpoints':
        return load_checkpoint_actor(path, expected=expected, device=device)
    if path.name.startswith('.'):
        raise MahjongError('CHECKPOINT_INVALID', 'unpublished Actor')
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['schema'] != SCHEMA or path.name != manifest['artifact_id']:
            raise ValueError('Actor identity mismatch')
        with (path / 'actor.pt').open('rb') as f:
            length, sha = _stream_digest(f)
        if (length, sha) != (manifest['weights_bytes'], manifest['weights_sha256']):
            raise ValueError('Actor weights digest mismatch')
        ref = dict(artifact_id=manifest['artifact_id'], manifest_hash=digest(manifest),
                   weights_sha256=sha, weights_bytes=length, compatibility_hash=digest(manifest['compatibility']))
        if expected is not None and expected != ref:
            raise ValueError('ActorRef mismatch')
        settings = manifest['settings']
        actor = Qwen3PolicyAdapter(build_qwen3(settings, init_seed=0), input_schema=settings['input_schema'])
        set_policy_precision(actor, manifest['compatibility']['precision'])
        if compatibility(actor, manifest['compatibility']['profile']) != manifest['compatibility']:
            raise ValueError('Actor compatibility mismatch')
        state = torch.load(path / 'actor.pt', map_location='cpu', weights_only=True)
        check_state(actor, state)
        actor.load_state_dict(state)
        actor.requires_grad_(False)
        return PolicyInference(actor.to(device), source=dict(actor_ref=ref, compatibility=manifest['compatibility'],
                                                            provenance=manifest['provenance']))
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, MahjongError):
            raise
        raise MahjongError('CHECKPOINT_INVALID', f'Actor load failed: {exc}') from exc


def new_actor_bridge(actors):
    from mahjong.inference.context_bridge import ContextDecisionBridge
    bridge = ContextDecisionBridge(actors[0].source['compatibility']['profile'], model_binding_digest=actors[0].binding_digest)
    bridge.bind_actors(actors)
    return bridge


def load_checkpoint_actor(path, *, expected=None, device='cpu'):
    """Explicit Actor-only extraction; PREPARED and READY both hold accepted weights."""
    from mahjong.training.checkpoint import FileCheckpointStore
    store = FileCheckpointStore()
    path = Path(path).resolve()
    snapshot = store.load(store.resolve(path), run_dir=path.parent.parent)
    if not any(d.implementation_id == 'mahjong.qwen3_rl/v1' for d in snapshot.descriptors):
        raise MahjongError('CHECKPOINT_INVALID', 'not a complete RL checkpoint')
    ref = snapshot.data['pool']['latest']
    if expected is not None and ref != expected:
        raise MahjongError('CHECKPOINT_INVALID', 'checkpoint ActorRef mismatch')
    loaded = load_actor(path.parent.parent/'actors'/ref['artifact_id'], expected=ref, device=device)
    state = {k.removeprefix('actor.'):v for k,v in snapshot.runtime.model.items() if k.startswith('actor.')}
    check_state(loaded.model, state)
    if any(not torch.equal(v.cpu(), loaded.model.state_dict()[k].cpu()) for k,v in state.items()):
        raise MahjongError('CHECKPOINT_INVALID', 'checkpoint and pool Actor weights differ')
    return loaded
