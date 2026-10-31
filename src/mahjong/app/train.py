"""Training entry points: identities and assembly, without training algorithms."""

from dataclasses import replace
from pathlib import Path

from mahjong.contracts import MahjongError
from . import assembly, config as configuration, runs
from .checkpoint_source import checkpoint_source
from .lifecycle import announce, bind_stop, finish


def train(overrides, *, cwd, stop, argv, init_weights=None, pretraining=False):
    from mahjong.training.checkpoint import FileCheckpointStore
    from mahjong.training.loop import Trainer
    from mahjong.evaluation.runner import Evaluator
    settings = configuration.parse_new(overrides, cwd,
                                       config_name='pretrain' if pretraining else 'train')
    runs.preflight(settings)
    store = FileCheckpointStore()
    weights = None
    weights_ref = None
    if init_weights is not None:
        weights_run, weights_ref = checkpoint_source(store, init_weights)
        weights = store.load(weights_ref, run_dir=weights_run)
    context = runs.create_run(settings)
    if weights is not None:
        context = replace(context, mode='weights')
        runs.record_weights_source(context, weights_ref)
    with runs.lock_run(context.run_dir):
        observer = assembly.build_observer(context, settings)
        bind_stop(observer, stop)
        announce(context, settings)

        def execute():
            observer.phase('initializing', None)
            observer.phase('preparing_data', None)
            source = assembly.build_train_source(settings)
            validation = None
            try:
                validation = assembly.build_eval_source(settings)
                runtime = assembly.build_runtime(settings, training=True)
                metadata = assembly.collect_metadata(settings, runtime, source)
                runs.publish_metadata(context, settings, metadata, argv=argv, init_from=weights_ref)
                if weights is not None:
                    from mahjong.training.resume import validate_weights
                    validate_weights(snapshot=weights, model_descriptor=runtime.descriptor(),
                                     model_state=runtime.export_state().model, model_config=settings['model'])
                    runtime.import_state(weights.runtime, model_only=True)
                    runtime.reset_auxiliary_rng(metadata.derived_seeds['model_private_seed'])
                runs.activate(context)
                return Trainer().run(context=context, metadata=metadata, config=settings, source=source,
                                     validation_source=validation, runtime=runtime, evaluator=Evaluator(),
                                     observer=observer, store=store, stop=stop)
            finally:
                for component in (source, validation):
                    close = getattr(component, 'close', None)
                    if close is not None:
                        close()

        return finish(observer, execute)


def resume(overrides, *, cwd, stop, argv, checkpoint, run=None):
    from mahjong.training.checkpoint import FileCheckpointStore
    from mahjong.training.loop import Trainer
    from mahjong.training.resume import prepare_resume
    from mahjong.evaluation.runner import Evaluator
    store = FileCheckpointStore()
    # Acquire ownership before resolving latest or loading source state.
    if checkpoint == 'latest':
        if run is None:
            raise MahjongError('CONFIG_INVALID', '--checkpoint latest requires --run')
        run_dir = Path(run).resolve()
    else:
        candidate = Path(checkpoint).resolve()
        if candidate.parent.name != 'checkpoints':
            raise MahjongError('CHECKPOINT_INVALID', 'Expected run/checkpoints/<checkpoint_id>')
        run_dir = candidate.parent.parent
        if run is not None and Path(run).resolve() != run_dir:
            raise MahjongError('CONFIG_INVALID', '--run disagrees with checkpoint')
    with runs.lock_run(run_dir):
        run_dir, reference = checkpoint_source(store, checkpoint, run)
        snapshot = store.load(reference, run_dir=run_dir)
        settings, changes = configuration.merge_resume(snapshot.config, overrides)
        runs.preflight(settings, output=False)
        context = runs.create_attempt(run_dir, reference, settings)
        observer = assembly.build_observer(context, settings)
        bind_stop(observer, stop)
        announce(context, settings)

        def execute():
            observer.phase('initializing', None)
            observer.phase('preparing_data', None)
            source = assembly.build_train_source(settings)
            validation = None
            try:
                validation = assembly.build_eval_source(settings)
                runtime = assembly.build_runtime(settings, training=True)
                metadata = assembly.collect_metadata(settings, runtime, source)
                runs.publish_metadata(context, settings, metadata, argv=argv, changes=changes,
                                      observation=snapshot.observation)
                plan = prepare_resume(source=reference, snapshot=snapshot, effective_config=settings,
                                      changes=changes, descriptors=metadata.descriptors,
                                      code_fingerprint=metadata.code_fingerprint,
                                      dependency_fingerprint=metadata.dependency_fingerprint)
                runs.activate(context)
                return Trainer().run(context=context, metadata=metadata, config=settings, source=source,
                                     validation_source=validation, runtime=runtime, evaluator=Evaluator(),
                                     observer=observer, store=store, stop=stop, resume=plan)
            finally:
                for component in (source, validation):
                    close = getattr(component, 'close', None)
                    if close is not None:
                        close()

        return finish(observer, execute, initial_checkpoint=reference)
