"""RL command assembly, strict NTP conversion and full checkpoint recovery."""
from copy import deepcopy
from pathlib import Path
import json
import torch
from mahjong import contracts as c
from mahjong.app import assembly, runs
from mahjong.app.config import parse_new
from mahjong.app.lifecycle import finish
from mahjong.app.checkpoint_source import checkpoint_source
from mahjong.inference.policy import PolicyInference, digest
from mahjong.inference.actor_loading import publish_actor, load_actor, compatibility
from mahjong.models.qwen3 import Qwen3PolicyAdapter, build_qwen3, SETTINGS_FIELDS
from mahjong.models.qwen3_rl import Qwen3RLAdapter
from mahjong.rl.config import resolve, validate
from mahjong.rl.source import RLTrainSource
from mahjong.rl.pool import initial_pool
from mahjong.rl.controller import RLController
from mahjong.rl.recovery import reconcile
from mahjong.training.runtime import FabricRuntime
from mahjong.training.checkpoint import FileCheckpointStore
from mahjong.policy.execution import set_policy_precision


def model_settings(rl):
    return {**rl['ppo'], 'oracle_width':rl['critic']['oracle_width'], 'value_width':rl['critic']['value_width'], 'use_oracle':rl['critic']['use_oracle']}


def make_runtime(actor, config):
    set_policy_precision(actor, config['runtime']['precision'])
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(c.derive_seed(config['rl']['seed'], 'mahjong-rl/critic-init/v1'))
        model = Qwen3RLAdapter(actor, model_settings(config['rl']))
    settings = config['rl']['ppo']
    groups = [dict(params=params, lr=settings[name+'_lr']) for name,params in model.parameter_groups()]
    optimizer = torch.optim.Adam(groups, foreach=False, fused=False)
    runtime = FabricRuntime(**config['runtime'])
    runtime.setup(model, optimizer)
    return runtime


def create_settings(rl, actor_settings, origin, output, device):
    base = parse_new([], Path.cwd())
    base.update(rl=rl, rl_origin=origin, model=actor_settings, seed=rl['seed'])
    base['runtime'].update(device=device, precision='bf16-mixed' if device=='cuda:0' else '32-true', cpu_threads=1)
    base['output']['root'] = str(Path(output).resolve())
    base['experiment']['name'] = 'mahjong-rl'
    base['training']['max_updates'] = rl['rounds']*rl['ppo']['max_updates']
    base['checkpoint'].update(every_updates=None, every_seconds=None)
    base['observability']['tensorboard_enabled'] = False
    return base


def train(options, *, stop, argv):
    rl = resolve(options.config, options.overrides)
    torch.set_num_threads(1)
    # Explicit NTP path: verifies original source once, then embeds its evidence.
    inference = PolicyInference.from_checkpoint(options.init_checkpoint, device='cpu')
    profile = json.loads((Path(__file__).resolve().parents[3]/'configs'/'inference-profile.json').read_text())
    inference.new_bridge(profile)
    compatibility(inference.model, profile)
    store = FileCheckpointStore()
    ntp_run, ntp_ref = checkpoint_source(store, options.init_checkpoint)
    snapshot = store.load(ntp_ref, run_dir=ntp_run)
    settings = {k:snapshot.config['model'][k] for k in SETTINGS_FIELDS}
    manifest = json.loads((ntp_run/ntp_ref.path/'manifest.json').read_text())
    origin = dict(checkpoint=c.encode_json_contract(ntp_ref), state_sha256=manifest['state_sha256'],
                  profile=profile, source_evidence=inference.source,
                  limitations='Initialization provenance does not establish BC strength or repaired legacy candidates.')
    config = create_settings(rl, settings, origin, options.output, options.device)
    actor = Qwen3PolicyAdapter(inference.lm, input_schema=settings['input_schema'])
    del snapshot, inference
    context = runs.create_run(config)
    with runs.lock_run(context.run_dir):
        observer = assembly.build_observer(context, config)
        observer.bind_stop(stop)
        print(f'run={context.run_dir}', flush=True)
        def execute():
            source = RLTrainSource(context.run_dir, rl, origin)
            runtime = make_runtime(actor, config)
            metadata = assembly.collect_metadata(config, runtime, source)
            runs.publish_metadata(context, config, metadata, argv=argv, init_from=ntp_ref)
            runs.activate(context)
            bc = publish_actor(context.run_dir/'actors', actor, settings, profile, origin,
                max_files=rl['pool']['physical_capacity'], byte_budget=rl['pool']['byte_budget'])
            source.state['pool'] = initial_pool(bc)
            return RLController(context=context,config=config,metadata=metadata,runtime=runtime,source=source,
                                observer=observer,store=store,stop=stop).run()
        return finish(observer, execute)


def resume(options, *, stop, argv):
    store = FileCheckpointStore()
    run_dir = Path(options.run).resolve() if options.checkpoint=='latest' and options.run else Path(options.checkpoint).resolve().parent.parent
    with runs.lock_run(run_dir):
        run_dir, ref = checkpoint_source(store, options.checkpoint, options.run)
        snapshot = store.load(ref, run_dir=run_dir)
        if 'rl' not in snapshot.config:
            raise c.MahjongError('RESUME_INCOMPATIBLE', 'rl resume requires a complete RL checkpoint')
        config = snapshot.config
        validate(config['rl'])
        latest = reconcile(store, run_dir, config['rl']['collection'], verify=True)
        if latest != ref:
            raise c.MahjongError('RESUME_INCOMPATIBLE', 'RL resume requires latest authoritative checkpoint')
        context = runs.create_attempt(run_dir, ref, config)
        observer = assembly.build_observer(context, config)
        observer.bind_stop(stop)
        def execute():
            settings = config['model']
            actor = Qwen3PolicyAdapter(build_qwen3(settings, init_seed=0), input_schema=settings['input_schema'])
            source = RLTrainSource(run_dir, config['rl'], config['rl_origin'])
            runtime = make_runtime(actor, config)
            metadata = assembly.collect_metadata(config, runtime, source)
            if (metadata.descriptors != snapshot.descriptors
                    or metadata.frozen_config_hash != snapshot.frozen_config_hash):
                raise c.MahjongError('RESUME_INCOMPATIBLE', 'RL configuration or components differ')
            runs.publish_metadata(context,config,metadata,argv=argv,observation=snapshot.observation)
            runs.activate(context)
            return RLController(context=context,config=config,metadata=metadata,runtime=runtime,source=source,
                observer=observer,store=store,stop=stop).run(snapshot)
        return finish(observer, execute, initial_checkpoint=ref)


def evaluate(options, *, stop):
    from mahjong.rl.evaluation import paired_evaluate
    rl = resolve(options.config, options.overrides)
    torch.set_num_threads(1)
    candidate = load_actor(options.actor, device=options.device)
    baseline = load_actor(options.baseline, device=options.device)
    opponent = load_actor(options.opponent or options.baseline, device=options.device)
    if any(a.source['compatibility'] != candidate.source['compatibility'] for a in (baseline,opponent)):
        raise c.MahjongError('CONTRACT_MISMATCH', 'evaluation actors are incompatible')
    directory = Path(options.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = paired_evaluate(directory,candidate,baseline,opponent,model=None,config=rl,iteration=0,
        cancel=stop.requested,seed_plan=options.seed_plan,strategy=options.strategy)
    runs.atomic_json(directory/'report.json', report)
    print(f'report={directory / "report.json"}', flush=True)
    return 0 if report['accepted'] else 1
