"""Read a checkpoint and produce one isolated evaluation transaction."""

from mahjong.contracts import Completion, EvaluationRequest, ModelSource, encode_json_contract
from . import assembly, config as configuration, runs
from .lifecycle import announce, bind_stop, finish
from .checkpoint_source import checkpoint_source


def evaluate(overrides, *, cwd, stop, argv, checkpoint):
    from mahjong.training.checkpoint import FileCheckpointStore
    from mahjong.evaluation.runner import Evaluator
    store = FileCheckpointStore()
    run_dir, reference = checkpoint_source(store, checkpoint)
    snapshot = store.load(reference, run_dir=run_dir)
    settings = configuration.merge_evaluate(snapshot.config, overrides, cwd)
    runs.preflight(settings, output=False)
    context = runs.create_evaluation(run_dir, reference)
    runs.write_config(context.evaluation_dir / 'resolved_config.yaml', settings)
    runs.atomic_json(context.evaluation_dir / 'manifest.json', {
        'schema_version': 1, 'evaluation_id': context.evaluation_id,
        'source_checkpoint': encode_json_contract(reference), 'config_hash': runs.config_hash(settings),
        'metadata': None, 'argv': list(argv), 'started_at': runs.utc_now(),
    })
    observer = assembly.build_observer(context, settings)
    bind_stop(observer, stop)
    announce(context, settings)

    def execute():
        observer.phase('initializing', None)
        observer.phase('preparing_data', None)
        source = assembly.build_eval_source(settings)
        runtime = assembly.build_runtime(settings, training=False)
        from mahjong.training.resume import validate_weights
        validate_weights(snapshot=snapshot, model_descriptor=runtime.descriptor(),
                         model_state=runtime.export_state().model, model_config=settings['model'])
        runtime.import_state(snapshot.runtime, model_only=True)
        metadata = assembly.collect_metadata(settings, runtime, source)
        runs.atomic_json(context.evaluation_dir / 'manifest.json', {
            'schema_version': 1, 'evaluation_id': context.evaluation_id,
            'source_checkpoint': encode_json_contract(reference), 'config_hash': runs.config_hash(settings),
            'metadata': encode_json_contract(metadata), 'argv': list(argv),
        })
        request = EvaluationRequest(
            context.evaluation_id, 'standalone',
            ModelSource(reference.run_id, reference.attempt_id, reference.global_step, reference.checkpoint_id),
            settings['evaluation']['seed'], settings['evaluation']['batch_size'], settings['runtime']['precision'],
            context.evaluation_dir / 'report.json', runs.config_hash(settings),
            f'evaluations/{context.evaluation_id}/manifest.json',
        )
        report = Evaluator().run(runtime, source, request, observer, stop)
        if report.status == 'cancelled':
            outcome, code = 'interrupted', 128 + (stop.signal_number() or 2)
        elif report.status == 'not_evaluable':
            outcome, code = 'not_evaluable', 3
        elif report.status == 'ok':
            outcome, code = 'completed', 0
        else:
            outcome, code = 'failed', 1
        return Completion(outcome, code, reference.global_step, reference, report.report_ref, report.reason)

    return finish(observer, execute, initial_checkpoint=reference)
