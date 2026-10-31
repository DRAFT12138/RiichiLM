"""Construct concrete components; algorithm choices belong only here."""

import hashlib
from pathlib import Path

import torch

from mahjong.contracts import MahjongError, RunMetadata, canonical_json, derive_seed
from mahjong.environment import collect_environment
from .config import ROOT, frozen_config
from .runs import config_hash


def seeds(config):
    root = config['seed']
    return {name: derive_seed(root, namespace) for name, namespace in {
        'data_seed': 'data/v1', 'sampler_seed': 'sampler/v1',
        'model_init_seed': 'model/init/v1', 'model_private_seed': 'model/private/v1',
        'train_seed': 'runtime/train/v1',
    }.items()}


def build_model(config):
    """Assemble an existing project Model from resolved model settings."""
    derived = seeds(config)
    settings = config['model']
    if settings['name'] == 'qwen3':
        from mahjong.models.qwen3 import (POLICY_INPUT_SCHEMAS, SETTINGS_FIELDS,
                                         Qwen3Adapter, Qwen3PolicyAdapter, build_qwen3)
        qwen_settings = {key: settings[key] for key in SETTINGS_FIELDS}
        causal_lm = build_qwen3(qwen_settings, init_seed=derived['model_init_seed'])
        if settings['input_schema'] in POLICY_INPUT_SCHEMAS:
            if config.get('data',{}).get('shared_context', False):
                from mahjong.models.qwen3_shared import Qwen3SharedPolicyAdapter
                return Qwen3SharedPolicyAdapter(causal_lm, input_schema=settings['input_schema'])
            return Qwen3PolicyAdapter(causal_lm, input_schema=settings['input_schema'])
        return Qwen3Adapter(causal_lm)
    if settings['name'] != 'mock_mlp':
        raise MahjongError('CONFIG_INVALID', 'model.name: supported models are mock_mlp and qwen3')
    from mahjong.models import MockMLP
    return MockMLP(
        input_dim=settings['input_dim'], hidden_dim=settings['hidden_dim'],
        dropout=settings['dropout'], target_rule=settings['target_rule'],
        input_schema=settings['input_schema'], init_seed=derived['model_init_seed'],
        private_seed=derived['model_private_seed'],
    )


def build_runtime(config, *, training):
    from mahjong.training.runtime import FabricRuntime
    model = build_model(config)
    optimizer = None
    if training:
        settings = config['optimizer']
        optimizer = torch.optim.Adam(
            model.parameters(), lr=settings['lr'], betas=tuple(settings['betas']),
            eps=settings['eps'], weight_decay=settings['weight_decay'],
            # GPU NTP uses the measured fused path for new runs and full resume;
            # import_state retains saved moments/steps and migrates execution
            # flags before loading, including step-counter device placement.
            foreach=False, fused=config['runtime']['device'] == 'cuda:0',
        )
    runtime = FabricRuntime(**config['runtime'])
    runtime.setup(model, optimizer)
    return runtime


def build_train_source(config):
    if config['data']['name'] == 'majsoul_policy':
        from mahjong.data.mahjong.policy import MahjongPolicyTrainSource
        derived = seeds(config)
        data = config['data']
        source_class, options = MahjongPolicyTrainSource, {}
        if data['shared_context']:
            from mahjong.data.mahjong.shared_policy import MahjongSharedPolicyTrainSource
            source_class = MahjongSharedPolicyTrainSource
            options = {'max_tokens':data['shared_max_tokens'],
                       'max_branch_cells':data['shared_max_branch_cells']}
        return source_class(**options,
            build_dir=data['build_dir'], seed=derived['sampler_seed'],
            batch_size=data['batch_size'], validation_fraction=data['validation_fraction'],
            split_seed=data['split_seed'], max_sequence_length=config['model']['max_position_embeddings'],
            shuffle=data['shuffle'], input_schema=data['input_schema'])
    from mahjong.data import SyntheticTrainSource
    derived = seeds(config)
    settings = {key: config['data'][key] for key in (
        'train_size', 'val_size', 'input_dim', 'batch_size', 'shuffle',
        'drop_last', 'num_workers', 'input_schema',
    )}
    return SyntheticTrainSource(seed=derived['data_seed'], sampler_seed=derived['sampler_seed'], **settings)


def build_eval_source(config):
    if config['data']['name'] == 'majsoul_policy':
        from mahjong.data.mahjong.policy import MahjongPolicyEvalSource
        from mahjong.policy.shared_context import INDEPENDENT_LAYOUT
        data = config['data']
        corpus = MahjongPolicyEvalSource(
            build_dir=data['build_dir'], batch_size=config['evaluation']['batch_size'],
            validation_fraction=data['validation_fraction'], split_seed=data['split_seed'],
            max_sequence_length=config['model']['max_position_embeddings'],
            max_samples=data['validation_max_samples'], input_schema=data['input_schema'],
            layout=INDEPENDENT_LAYOUT if data['shared_context'] else None)
        from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
        directory = config['evaluation'].get('nanikiru_dir', 'data/evaluation/nanikiru')
        if data['input_schema'] != CONTEXT_SCHEMA or directory is None:
            return corpus
        from mahjong.data.mahjong.nanikiru import NanikiruEvalSource, PolicyEvalSuite
        try:
            benchmark = NanikiruEvalSource(ROOT / directory,
                max_sequence_length=config['model']['max_position_embeddings'])
            return PolicyEvalSuite(corpus, benchmark)
        except Exception:
            corpus.close()
            raise
    from mahjong.data import SyntheticEvalSource
    settings = config['data']
    return SyntheticEvalSource(
        seed=seeds(config)['data_seed'], train_size=settings['train_size'],
        val_size=settings['val_size'], input_dim=settings['input_dim'],
        batch_size=config['evaluation']['batch_size'], num_workers=0,
        input_schema=settings['input_schema'],
    )


def build_observer(context, config):
    from mahjong.observability.events import LocalObserver
    return LocalObserver(context, config)


def source_fingerprint():
    paths = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'configs').rglob('*.yaml'))
    paths += [ROOT / name for name in (
        '.python-version', 'pyproject.toml', 'requirements-rocm.txt',
        'requirements-cuda.txt',
    )]
    paths += [ROOT / name for name in (
        'data/evaluation/nanikiru/manifest.json', 'data/evaluation/nanikiru/questions.jsonl',
    ) if (ROOT / name).is_file()]
    digest_map = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if '__pycache__' not in path.parts
    }
    return hashlib.sha256(canonical_json(digest_map).encode()).hexdigest(), digest_map


def collect_metadata(config, runtime, source):
    environment = collect_environment(config['output']['root'])
    code_fingerprint, files = source_fingerprint()
    environment['controlled_source_sha256'] = files
    dependency_map = {
        item['name']: item['version'] for item in environment['dependencies']
        if item['name'] != 'mahjong'
    }
    dependency_fingerprint = hashlib.sha256(canonical_json({
        'python': environment['python'], 'versions': dependency_map,
        'lock_sha256': {name: files[name] for name in
                        ('requirements-rocm.txt', 'requirements-cuda.txt')},
    }).encode()).hexdigest()
    return RunMetadata(
        code_fingerprint, dependency_fingerprint, environment,
        (source.descriptor(), runtime.descriptor()), config_hash(frozen_config(config)),
        seeds(config),
    )
