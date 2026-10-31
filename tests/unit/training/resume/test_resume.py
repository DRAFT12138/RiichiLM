from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import random

import pytest
import torch

from mahjong.app.config import DEFAULTS, RESUME_ALLOWED, flatten, frozen_config, merge_resume
from mahjong.app.runs import config_hash
from mahjong.contracts import (
    CheckpointRef, ConfigChange, MahjongError, PendingValidation, RecordIdentity,
    RunContext, RunMetadata, clone_state, encode_contract,
)
from mahjong.training.resume import prepare_resume, validate_weights
from tests.fixtures.contracts import (
    KnownEvalSource, KnownEvaluator, KnownModel, KnownRuntime, KnownSource,
    MemoryStore, MutableStopToken, RecordingObserver, make_snapshot,
)


FROZEN_FIELDS = sorted(flatten(DEFAULTS).keys() - RESUME_ALLOWED)


def set_field(config, path, value):
    target = config
    pieces = path.split('.')
    for piece in pieces[:-1]:
        target = target[piece]
    target[pieces[-1]] = value


def fixture():
    config = deepcopy(DEFAULTS)
    config['training']['max_updates'] = 4
    snapshot = make_snapshot()
    snapshot = replace(snapshot, config=config, frozen_config_hash=config_hash(frozen_config(config)),
                       target_updates=4, progress=replace(snapshot.progress, global_step=2, initial_validation_done=True),
                       identity=replace(snapshot.identity, global_step=2))
    ref = CheckpointRef(snapshot.identity.checkpoint_id, snapshot.identity.run_id, snapshot.identity.attempt_id,
                        2, 'checkpoints/' + snapshot.identity.checkpoint_id, 'fixture-manifest', snapshot.identity.commit_seq)
    kwargs = {'source':ref, 'snapshot':snapshot, 'effective_config':deepcopy(config), 'changes':(),
              'descriptors':snapshot.descriptors, 'code_fingerprint':snapshot.identity.code_fingerprint,
              'dependency_fingerprint':snapshot.identity.dependency_fingerprint}
    return kwargs


def assert_incompatible(call, path=None):
    with pytest.raises(MahjongError) as error:
        call()
    assert error.value.code == 'RESUME_INCOMPATIBLE' and error.value.exit_code == 2
    if path is not None:
        assert path in error.value.message
    return error.value


def test_v037_unchanged_and_whitelisted_resume_are_pure_and_owned():
    kwargs = fixture()
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    plan = prepare_resume(**kwargs)
    assert plan.snapshot is kwargs['snapshot'] and plan.source == kwargs['source']
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    effective, changes = merge_resume(kwargs['snapshot'].config, [
        'training.max_updates=6', 'observability.log_level=DEBUG',
        'observability.heartbeat_seconds=1.0', 'observability.local_flush_seconds=0.5',
        'observability.resource_seconds=1.0', 'observability.tensorboard_enabled=false',
        'observability.tensorboard_flush_seconds=1.0',
        'checkpoint.every_updates=5000', 'checkpoint.every_seconds=900',
        'validation.every_updates=5000'])
    assert {change.path for change in changes} == RESUME_ALLOWED
    updated = prepare_resume(**dict(kwargs, effective_config=effective, changes=changes))
    assert updated.effective_config['training']['max_updates'] == 6
    effective['observability']['heartbeat_seconds'] = 999.
    assert updated.effective_config['observability']['heartbeat_seconds'] == 1.


@pytest.mark.parametrize('path', FROZEN_FIELDS)
def test_v037_every_frozen_configuration_leaf_is_rejected(path):
    kwargs = fixture()
    current = flatten(kwargs['effective_config'])[path]
    if type(current) is bool:
        different = not current
    elif type(current) is int:
        different = current + 1
    elif type(current) is float:
        different = current + .01
    elif type(current) is str:
        different = current + '/changed'
    elif type(current) is list:
        different = [0.8, 0.99]
    else:
        different = 'changed'
    set_field(kwargs['effective_config'], path, different)
    error = assert_incompatible(lambda: prepare_resume(**kwargs))
    assert path in error.message or path in str(error.details)


@pytest.mark.parametrize('field,new', [
    ('implementation_id', 'changed/v1'), ('input_schema', 'changed/v1'),
    ('config_fingerprint', 'changed-config'), ('data_fingerprint', 'changed-data'),
    ('metric_schema_version', None), ('recovery_capability', 'committed_batch'),
    ('supported_execution', ('other',)),
])
def test_v037_model_descriptor_fields_are_frozen(field, new):
    kwargs = fixture()
    model = next(descriptor for descriptor in kwargs['descriptors'] if descriptor.kind == 'model')
    modified = replace(model, **{field:new})
    kwargs['descriptors'] = tuple(modified if descriptor.kind == 'model' else descriptor for descriptor in kwargs['descriptors'])
    assert_incompatible(lambda: prepare_resume(**kwargs), f'descriptors.model.{field}')


def test_v037_state_versions_descriptor_count_and_execution_declarations():
    for kind in ('data', 'model'):
        kwargs = fixture()
        modified = deepcopy(next(descriptor for descriptor in kwargs['descriptors'] if descriptor.kind == kind))
        object.__setattr__(modified, 'state_version', 2)
        kwargs['descriptors'] = tuple(modified if descriptor.kind == kind else descriptor for descriptor in kwargs['descriptors'])
        assert_incompatible(lambda: prepare_resume(**kwargs), 'state_version')
    for version_field in ('format_version', 'contract_version'):
        kwargs = fixture()
        object.__setattr__(kwargs['snapshot'], version_field, 2)
        assert_incompatible(lambda: prepare_resume(**kwargs), version_field)
    for descriptor_change in ('missing', 'duplicate', 'empty_execution'):
        kwargs = fixture()
        descriptors = kwargs['descriptors']
        if descriptor_change == 'missing':
            descriptors = descriptors[:1]
        elif descriptor_change == 'duplicate':
            descriptors = descriptors + descriptors[:1]
        else:
            descriptors = (replace(descriptors[0], supported_execution=()), descriptors[1])
        assert_incompatible(lambda: prepare_resume(**dict(kwargs, descriptors=descriptors)), 'descriptors')


@pytest.mark.parametrize('fingerprint', ['changed', '', 'not_provided'])
def test_dependency_fingerprint_is_provenance_only(fingerprint):
    kwargs = fixture()
    plan = prepare_resume(**dict(kwargs, dependency_fingerprint=fingerprint))
    assert plan.snapshot is kwargs['snapshot']
    assert plan.snapshot.runtime is kwargs['snapshot'].runtime
    assert plan.snapshot.data == kwargs['snapshot'].data


@pytest.mark.parametrize('fingerprint', ['changed', '', 'not_provided'])
def test_source_fingerprint_is_provenance_only(fingerprint):
    kwargs = fixture()
    plan = prepare_resume(**dict(kwargs, code_fingerprint=fingerprint))
    assert plan.source == kwargs['source']
    assert plan.snapshot is kwargs['snapshot']
    assert plan.snapshot.runtime is kwargs['snapshot'].runtime
    assert plan.snapshot.data == kwargs['snapshot'].data


CONTROLLED_FILES = ['src/mahjong/example.py', 'configs/train.yaml', '.python-version',
                    'pyproject.toml', 'requirements-rocm.txt', 'requirements-cuda.txt',
                    'src/mahjong/models/nanikiru.py', 'src/mahjong/evaluation/runner.py',
                    'data/evaluation/nanikiru/manifest.json', 'data/evaluation/nanikiru/questions.jsonl']


def controlled_tree(tmp_path, monkeypatch):
    import mahjong.app.assembly as assembly
    root = tmp_path/'source-tree'
    for name in CONTROLLED_FILES + ['docs/review.md']:
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('initial controlled file content\n')
    monkeypatch.setattr(assembly,'ROOT',root)
    return assembly,root


def test_v037_document_only_changes_do_not_change_controlled_source_fingerprint(tmp_path,monkeypatch):
    # Mutate a document inside the same root used by the actual app algorithm.
    assembly,root = controlled_tree(tmp_path,monkeypatch)
    before, files = assembly.source_fingerprint()
    (root/'docs/review.md').write_text('Documentation-only review changed\n')
    after, changed_files = assembly.source_fingerprint()
    assert before == after and files == changed_files
    assert all(not path.startswith('docs/') for path in files)
    kwargs = fixture()
    kwargs['snapshot'] = replace(kwargs['snapshot'], identity=replace(kwargs['snapshot'].identity, code_fingerprint=before))
    assert prepare_resume(**dict(kwargs, code_fingerprint=after)).source == kwargs['source']


@pytest.mark.parametrize('path',CONTROLLED_FILES)
def test_controlled_file_changes_are_recorded_without_blocking_compatible_resume(tmp_path,monkeypatch,path):
    assembly,root = controlled_tree(tmp_path,monkeypatch)
    before,files = assembly.source_fingerprint()
    (root/path).write_text('changed controlled file\n')
    after,changed_files = assembly.source_fingerprint()
    assert before != after and files[path] != changed_files[path]
    kwargs = fixture()
    kwargs['snapshot'] = replace(kwargs['snapshot'],identity=replace(kwargs['snapshot'].identity,code_fingerprint=before))
    plan = prepare_resume(**dict(kwargs,code_fingerprint=after))
    assert plan.snapshot.identity.code_fingerprint == before
    assert plan.source == kwargs['source']


@pytest.mark.parametrize('target', [0, 1, 2, 3])
def test_v037_target_cannot_reduce_original_total_or_go_behind_progress(target):
    kwargs = fixture()
    kwargs['effective_config']['training']['max_updates'] = target
    kwargs['changes'] = (ConfigChange('training.max_updates', 4, target, 'requested'),)
    assert_incompatible(lambda: prepare_resume(**kwargs), 'training.max_updates')


def test_v037_change_records_must_match_all_and_only_allowed_differences():
    kwargs = fixture()
    effective, changes = merge_resume(kwargs['snapshot'].config, ['training.max_updates=6'])
    for bad in [(), changes+changes, (replace(changes[0], old=3),), (replace(changes[0], new=7),),
                (replace(changes[0], reason=''),), (ConfigChange('optimizer.lr', .003, .006, 'disallowed'),)]:
        assert_incompatible(lambda: prepare_resume(**dict(kwargs, effective_config=effective, changes=bad)))
    assert_incompatible(lambda: prepare_resume(**dict(kwargs, changes=(ConfigChange('training.max_updates',4,4,'unchanged'),))))


def test_v037_identity_frozen_hash_optimizer_and_pending_relationship_checks():
    kwargs = fixture()
    assert_incompatible(lambda: prepare_resume(**dict(kwargs, source=replace(kwargs['source'], run_id='other'))), 'source.run_id')
    assert_incompatible(lambda: prepare_resume(**dict(kwargs, source=replace(kwargs['source'], path='../C'))), 'source.path')
    broken = replace(kwargs['snapshot'], frozen_config_hash='different')
    assert_incompatible(lambda: prepare_resume(**dict(kwargs, snapshot=broken)), 'frozen_config_hash')
    broken = replace(kwargs['snapshot'], runtime=replace(kwargs['snapshot'].runtime, optimizer=None))
    assert_incompatible(lambda: prepare_resume(**dict(kwargs, snapshot=broken)), 'runtime.optimizer')
    for pending, done in [(PendingValidation('periodic', 2), False), (PendingValidation('final', 2), False),
                          (PendingValidation('initial', 2), True), (PendingValidation('periodic', 1), True)]:
        broken = replace(kwargs['snapshot'], progress=replace(kwargs['snapshot'].progress,
                         pending_validation=pending, initial_validation_done=done))
        assert_incompatible(lambda: prepare_resume(**dict(kwargs, snapshot=broken)), 'pending_validation')
    for pending, done in [(PendingValidation('periodic', 2), True), (PendingValidation('final', 2), True),
                          (PendingValidation('initial', 2), False)]:
        preserved = replace(kwargs['snapshot'], progress=replace(kwargs['snapshot'].progress,
                            pending_validation=pending, initial_validation_done=done))
        assert prepare_resume(**dict(kwargs, snapshot=preserved)).snapshot.progress.pending_validation == pending


def weights_fixture():
    kwargs = fixture()
    snapshot = kwargs['snapshot']
    model = next(descriptor for descriptor in snapshot.descriptors if descriptor.kind == 'model')
    return {'snapshot':snapshot, 'model_descriptor':model,
            'model_state':clone_state(snapshot.runtime.model), 'model_config':deepcopy(snapshot.config['model'])}


def test_v038_weights_allow_dropout_and_other_new_run_settings_without_importing_state():
    kwargs = weights_fixture()
    kwargs['model_config']['dropout'] = .1
    kwargs['model_descriptor'] = replace(kwargs['model_descriptor'], config_fingerprint='new-dropout-config')
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    before = kwargs['model_state']['weight'].clone()
    validate_weights(**kwargs)
    assert torch.equal(before, kwargs['model_state']['weight'])
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()


@pytest.mark.parametrize('mutation', ['missing_key','extra_key','shape','dtype','input_schema','target_rule','input_dim','hidden_dim',
                                      'implementation_id','state_version','empty_execution'])
def test_v038_weights_keys_shapes_dtypes_and_semantics_are_strict(mutation):
    kwargs = weights_fixture()
    if mutation == 'missing_key':
        kwargs['model_state'].pop('weight')
    elif mutation == 'extra_key':
        kwargs['model_state']['unexpected'] = torch.tensor(1.)
    elif mutation == 'shape':
        kwargs['model_state']['weight'] = torch.ones(2)
    elif mutation == 'dtype':
        kwargs['model_state']['weight'] = kwargs['model_state']['weight'].double()
    elif mutation in {'input_schema','target_rule','input_dim','hidden_dim'}:
        kwargs['model_config'][mutation] = 'changed' if mutation in {'input_schema','target_rule'} else 999
    elif mutation == 'empty_execution':
        kwargs['model_descriptor'] = replace(kwargs['model_descriptor'], supported_execution=())
    elif mutation == 'state_version':
        object.__setattr__(kwargs['model_descriptor'], 'state_version', 2)
    else:
        kwargs['model_descriptor'] = replace(kwargs['model_descriptor'], implementation_id='different/v1')
    assert_incompatible(lambda: validate_weights(**kwargs))


def test_v038_real_fabric_model_only_import_preserves_new_optimizer_data_and_private_seed():
    from mahjong.training.runtime import FabricRuntime
    model = KnownModel()
    old_runtime = FabricRuntime()
    old_runtime.setup(model, torch.optim.Adam(model.parameters(),lr=.1))
    old_source = KnownSource()
    lease = old_source.acquire()
    loss = old_runtime.compute(lease.batch,phase='train').loss
    old_runtime.backward(loss)
    old_runtime.check_gradients()
    old_runtime.step()
    old_source.commit(lease.token)
    old_runtime.zero_grad()
    snapshot = make_snapshot(old_runtime,old_source)
    new_model = KnownModel()
    new_runtime = FabricRuntime()
    new_runtime.setup(new_model,torch.optim.Adam(new_model.parameters(),lr=.005))
    new_source = KnownSource()
    new_state = new_runtime.export_state()
    config = deepcopy(DEFAULTS['model'])
    snapshot = replace(snapshot,config=deepcopy(DEFAULTS))
    validate_weights(snapshot=snapshot,model_descriptor=new_runtime.descriptor(),model_state=new_state.model,model_config=config)
    new_runtime.import_state(snapshot.runtime,model_only=True)
    new_runtime.reset_auxiliary_rng(123)
    exported = new_runtime.export_state()
    assert torch.equal(exported.model['weight'],snapshot.runtime.model['weight'])
    assert exported.optimizer['state'] == {} and exported.optimizer['param_groups'][0]['lr'] == .005
    assert new_source.state_dict()['index'] == 0
    assert torch.equal(exported.auxiliary['rng'],torch.Generator().manual_seed(123).get_state())
    assert not torch.equal(exported.auxiliary['rng'],snapshot.runtime.auxiliary['rng'])
    previous = exported.model['weight'].clone()
    loss = new_runtime.compute(new_source.acquire().batch,phase='train').loss
    new_runtime.backward(loss)
    new_runtime.check_gradients()
    new_runtime.step()
    assert not torch.equal(previous,new_runtime.export_state().model['weight'])


def test_v038_source_descriptor_version_execution_and_model_semantics_are_required():
    for mutation in ('version','empty_execution','missing_model'):
        kwargs = weights_fixture()
        snapshot = kwargs['snapshot']
        if mutation == 'missing_model':
            snapshot.config.pop('model')
        else:
            old_model = deepcopy(next(value for value in snapshot.descriptors if value.kind == 'model'))
            if mutation == 'version':
                object.__setattr__(old_model,'state_version',2)
            else:
                object.__setattr__(old_model,'supported_execution',())
            kwargs['snapshot'] = replace(snapshot,descriptors=tuple(old_model if value.kind == 'model' else value
                                             for value in snapshot.descriptors)) if mutation != 'version' else snapshot
            if mutation == 'version':
                object.__setattr__(snapshot,'descriptors',tuple(old_model if value.kind == 'model' else value for value in snapshot.descriptors))
        assert_incompatible(lambda:validate_weights(**kwargs))


def test_v034_real_runner_imports_runtime_data_window_then_rng_before_updates(tmp_path):
    from mahjong.training.loop import Trainer
    order = []
    class RuntimeSpy(KnownRuntime):
        def import_state(self,state,*,model_only=False):
            order.append('runtime')
            return super().import_state(state,model_only=model_only)
        def restore_rng(self,state):
            order.append('rng')
            return super().restore_rng(state)
        def compute(self,batch,*,phase):
            order.append(f'compute:{phase}')
            return super().compute(batch,phase=phase)
    class SourceSpy(KnownSource):
        def load_state_dict(self,state):
            order.append('data')
            return super().load_state_dict(state)
    class ObserverSpy(RecordingObserver):
        def load_state_dict(self,state,source):
            order.append('window')
            return super().load_state_dict(state,source)
    kwargs = fixture()
    plan = prepare_resume(**kwargs)
    runtime = RuntimeSpy()
    model = KnownModel()
    runtime.setup(model,torch.optim.Adam(model.parameters(),lr=.1))
    context = RunContext('R','A2',tmp_path,tmp_path/'attempts/A2','resume',plan.source)
    observer = ObserverSpy(RecordIdentity('training','R','A2',None))
    metadata = RunMetadata(kwargs['code_fingerprint'],kwargs['dependency_fingerprint'],{},
                           kwargs['descriptors'],kwargs['snapshot'].frozen_config_hash,{'train_seed':1})
    completion = Trainer().run(context=context,metadata=metadata,config=plan.effective_config,
        source=SourceSpy(),validation_source=KnownEvalSource(),runtime=runtime,
        evaluator=KnownEvaluator(),observer=observer,store=MemoryStore(),stop=MutableStopToken(),resume=plan)
    assert order[:5] == ['runtime','data','window','rng','compute:train']
    assert completion.global_step == 4 and completion.exit_code == 0
    assert len([name for name in order if name == 'compute:train']) == 2


def test_validation_cadence_override_retains_legacy_hash_and_requires_a_change_record():
    kwargs=fixture();snapshot=kwargs['snapshot']
    # The prior frozen hash included validation.every_updates. Keep that exact
    # historical basis, so no checkpoint rewriting or alternate hash is needed.
    legacy={p:v for p,v in flatten(snapshot.config).items()
            if p not in (RESUME_ALLOWED-{'validation.every_updates'})}
    assert config_hash(legacy)==snapshot.frozen_config_hash
    effective,changes=merge_resume(snapshot.config,['validation.every_updates=5000'])
    accepted=prepare_resume(**dict(kwargs,effective_config=effective,changes=changes))
    assert accepted.effective_config['validation']['every_updates']==5000
    assert snapshot.config['validation']['every_updates']==50
    assert_incompatible(lambda:prepare_resume(**dict(kwargs,effective_config=effective,changes=())), 'changes')
