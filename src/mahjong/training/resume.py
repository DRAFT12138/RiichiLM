"""Pure compatibility checks for complete resume and weight initialization."""

from collections.abc import Mapping
from dataclasses import fields

import torch

from mahjong.app.config import RESUME_ALLOWED, flatten, frozen_config, validate
from mahjong.app.runs import config_hash
from mahjong.contracts import (
    CheckpointRef, ComponentDescriptor, ConfigChange, JsonValue, MahjongError,
    ResolvedConfig, ResumePlan, Snapshot, State, canonical_json, clone_state,
    encode_json_contract, validate_contract,
)


def _incompatible(path: str, message: str, saved=None, current=None) -> MahjongError:
    return MahjongError('RESUME_INCOMPATIBLE', f'{path}: {message}', 'restoring',
                       {'path': path, 'saved': saved, 'current': current})


def _same(left, right) -> bool:
    return canonical_json(encode_json_contract(left)) == canonical_json(encode_json_contract(right))


def _descriptors(values: tuple[ComponentDescriptor, ...], label: str) -> dict[str, ComponentDescriptor]:
    result = {}
    for descriptor in values:
        try:
            validate_contract(descriptor)
        except MahjongError as exc:
            raise _incompatible(f'{label}.{descriptor.kind}', str(exc)) from exc
        if descriptor.kind in result:
            raise _incompatible(label, f'Duplicate {descriptor.kind} descriptor')
        if not descriptor.supported_execution:
            raise _incompatible(f'{label}.{descriptor.kind}.supported_execution', 'No execution mode declared')
        result[descriptor.kind] = descriptor
    if set(result) != {'data', 'model'}:
        raise _incompatible(label, 'Complete resume requires one data and one model descriptor')
    return result


def prepare_resume(*, source: CheckpointRef, snapshot: Snapshot,
                   effective_config: ResolvedConfig, changes: tuple[ConfigChange, ...],
                   descriptors: tuple[ComponentDescriptor, ...], code_fingerprint: str,
                   dependency_fingerprint: str) -> ResumePlan:
    """Validate a loaded Snapshot without importing state or consuming RNG.

    The app merges the checkpoint configuration with its existing whitelist;
    the Runner imports the accepted plan and restores RNG last.
    Source and dependency fingerprints are provenance recorded by the app,
    not state compatibility requirements. Arguments remain for existing callers.
    """
    try:
        validate_contract(snapshot)
        validate_contract(source)
        saved_config = validate(snapshot.config)
        current_config = validate(effective_config)
    except MahjongError as exc:
        raise _incompatible('snapshot/config', str(exc)) from exc
    identity = snapshot.identity
    for name in ('run_id', 'attempt_id', 'checkpoint_id', 'global_step', 'commit_seq'):
        old, new = getattr(identity, name), getattr(source, name)
        if old != new:
            raise _incompatible(f'source.{name}', 'Reference differs from Snapshot identity', old, new)
    if source.path != f'checkpoints/{source.checkpoint_id}':
        raise _incompatible('source.path', 'Reference path differs from checkpoint identity', source.path)
    saved_frozen = frozen_config(saved_config)
    current_frozen = frozen_config(current_config)
    if config_hash(saved_frozen) != snapshot.frozen_config_hash:
        raise _incompatible('frozen_config_hash', 'Saved configuration does not match its frozen hash',
                            snapshot.frozen_config_hash, config_hash(saved_frozen))
    for path in sorted(saved_frozen.keys() | current_frozen.keys()):
        if path in RESUME_ALLOWED:
            continue  # Legacy hash includes cadence; declared overrides below own it.
        if path not in saved_frozen or path not in current_frozen or not _same(saved_frozen[path], current_frozen[path]):
            raise _incompatible(path, 'Frozen configuration changed', saved_frozen.get(path), current_frozen.get(path))
    target = current_config['training']['max_updates']
    original_target = saved_config['training']['max_updates']
    if snapshot.target_updates != original_target:
        raise _incompatible('target_updates', 'Snapshot target differs from saved configuration',
                            snapshot.target_updates, original_target)
    if target < original_target or target < snapshot.progress.global_step:
        raise _incompatible('training.max_updates', 'Cannot reduce the original target or go behind saved progress',
                            original_target, target)
    saved_values, current_values = flatten(saved_config), flatten(current_config)
    actual_changes = {path for path in saved_values if not _same(saved_values[path], current_values[path])}
    declared = set()
    for change in changes:
        validate_contract(change)
        if change.path not in RESUME_ALLOWED or change.path in declared or change.path not in actual_changes:
            raise _incompatible(change.path, 'Change is not a unique, effective allowed override')
        if not _same(change.old, saved_values[change.path]) or not _same(change.new, current_values[change.path]) or not change.reason.strip():
            raise _incompatible(change.path, 'Change record differs from effective configuration or lacks a reason')
        declared.add(change.path)
    if declared != actual_changes:
        raise _incompatible('changes', 'Allowed configuration differences must all have accurate change records',
                            sorted(actual_changes), sorted(declared))
    old_descriptors = _descriptors(snapshot.descriptors, 'snapshot.descriptors')
    new_descriptors = _descriptors(descriptors, 'descriptors')
    for kind, old in old_descriptors.items():
        new = new_descriptors[kind]
        for field in fields(ComponentDescriptor):
            old_value, new_value = getattr(old, field.name), getattr(new, field.name)
            if not _same(old_value, new_value):
                raise _incompatible(f'descriptors.{kind}.{field.name}', 'Component descriptor changed',
                                    encode_json_contract(old_value), encode_json_contract(new_value))
    # Source and environment hashes are provenance, not resume gates. Component,
    # configuration and state compatibility are checked independently.
    progress = snapshot.progress
    if progress.global_step != identity.global_step or progress.global_step > original_target:
        raise _incompatible('progress.global_step', 'Saved progress differs from identity or exceeds the original target')
    if snapshot.observation.cutoff.attempt_id != identity.attempt_id:
        raise _incompatible('observation.cutoff.attempt_id', 'Log cutoff belongs to another attempt')
    if snapshot.runtime.optimizer is None:
        raise _incompatible('runtime.optimizer', 'Complete resume requires optimizer state')
    pending = progress.pending_validation
    if pending is not None:
        if pending.global_step != progress.global_step:
            raise _incompatible('progress.pending_validation.global_step', 'Pending validation must refer to saved progress')
        if not saved_config['validation']['enabled']:
            raise _incompatible('progress.pending_validation', 'Disabled validation cannot have a pending request')
        if (pending.purpose == 'initial') == progress.initial_validation_done:
            raise _incompatible('progress.pending_validation', 'Initial handling and pending purpose are inconsistent')
    return ResumePlan(source, snapshot, clone_state(current_config), changes)


def validate_weights(*, snapshot: Snapshot, model_descriptor: ComponentDescriptor,
                     model_state: State, model_config: Mapping[str, JsonValue]) -> None:
    """Check model compatibility only; do not import optimizer, data or RNG.

    The app imports model_only=True after this check and resets auxiliary RNG
    from the new run seed. Different dropout is allowed for the new run.
    """
    try:
        validate_contract(model_descriptor)
    except MahjongError as exc:
        raise _incompatible('model_descriptor', str(exc)) from exc
    old_models = [descriptor for descriptor in snapshot.descriptors if descriptor.kind == 'model']
    if len(old_models) != 1 or model_descriptor.kind != 'model':
        raise _incompatible('model_descriptor.kind', 'Weights require exactly one source and current model descriptor')
    old = old_models[0]
    try:
        validate_contract(old)
    except MahjongError as exc:
        raise _incompatible('snapshot.model_descriptor', str(exc)) from exc
    for name in ('implementation_id', 'state_version', 'input_schema', 'metric_schema_version', 'recovery_capability'):
        old_value, new_value = getattr(old, name), getattr(model_descriptor, name)
        if old_value != new_value:
            raise _incompatible(f'model_descriptor.{name}', 'Model structure or semantics changed', old_value, new_value)
    if not old.supported_execution or not model_descriptor.supported_execution or model_descriptor.recovery_capability != 'state_dict':
        raise _incompatible('model_descriptor', 'Weights require a declared execution mode and state_dict recovery')
    previous_config = snapshot.config.get('model')
    if not isinstance(previous_config, Mapping):
        raise _incompatible('snapshot.config.model', 'Saved model semantics are required for weight checks')
    for name in ('input_schema', 'input_dim', 'hidden_dim', 'target_rule'):
        if name not in model_config or name not in previous_config or not _same(previous_config[name], model_config[name]):
            raise _incompatible(f'model.{name}', 'Weights require identical input, parameter structure and target semantics',
                                previous_config.get(name), model_config.get(name))
    saved_state = snapshot.runtime.model
    if saved_state.keys() != model_state.keys():
        raise _incompatible('model_state.keys', 'Strict parameter/buffer keys differ',
                            sorted(map(str, saved_state)), sorted(map(str, model_state)))
    for name, old_value in saved_state.items():
        new_value = model_state[name]
        if isinstance(old_value, torch.Tensor) and isinstance(new_value, torch.Tensor):
            if old_value.shape != new_value.shape or old_value.dtype != new_value.dtype:
                raise _incompatible(f'model_state.{name}', 'Parameter/buffer shape or dtype differs',
                                    {'shape': list(old_value.shape), 'dtype': str(old_value.dtype)},
                                    {'shape': list(new_value.shape), 'dtype': str(new_value.dtype)})
        elif type(old_value) is not type(new_value) or not _same(old_value, new_value):
            raise _incompatible(f'model_state.{name}', 'Non-tensor persistent model values differ')
