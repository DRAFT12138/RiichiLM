"""Strict phase 1 configuration and fixed-root Hydra composition."""

from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    name: str = 'mock'
@dataclass
class RuntimeConfig:
    device: str = 'cpu'
    precision: str = '32-true'
    deterministic: bool = True
    cpu_threads: int = 1
@dataclass
class DataConfig:
    name: str = 'synthetic'
    input_schema: str = 'synthetic-regression/v1'
    train_size: int = 1024
    val_size: int = 256
    input_dim: int = 8
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0
    build_dir: str = ''
    validation_fraction: float = 0.02
    validation_max_samples: int | None = None
    split_seed: int = 20261018
    shared_context: bool = False
    shared_max_tokens: int = 8192
    shared_max_branch_cells: int = 4194304
@dataclass
class ModelConfig:
    name: str = 'mock_mlp'
    input_schema: str = 'synthetic-regression/v1'
    input_dim: int = 8
    hidden_dim: int = 32
    dropout: float = 0.0
    target_rule: str = 'linear-teacher/v1'
    num_hidden_layers: int = 16
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 1024
    max_position_embeddings: int = 2048
    attn_implementation: str = 'sdpa'
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    use_cache: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
@dataclass
class OptimizerConfig:
    name: str = 'adam'
    lr: float = 0.003
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.0
@dataclass
class TrainingConfig:
    max_updates: int = 200
    accumulate_grad_batches: int = 1
    scheduler: str | None = None
    max_consecutive_skips: int = 32
@dataclass
class ValidationConfig:
    enabled: bool = True
    at_start: bool = True
    every_updates: int | None = 50
@dataclass
class EvaluationConfig:
    seed: int = 20261012
    batch_size: int = 32
    nanikiru_dir: str | None = 'data/evaluation/nanikiru'
@dataclass
class CheckpointConfig:
    every_updates: int | None = None
    every_seconds: float | None = 1800.0
@dataclass
class ObservabilityConfig:
    log_level: str = 'INFO'
    metrics_every_updates: int = 10
    heartbeat_seconds: float = 2.0
    local_flush_seconds: float = 1.0
    tensorboard_enabled: bool = True
    tensorboard_flush_seconds: float = 2.0
    resource_seconds: float = 2.0
@dataclass
class OutputConfig:
    root: str = 'outputs'
@dataclass
class AppConfig:
    seed: int = 20261011
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import re

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf
import yaml

from mahjong.contracts import ConfigChange, MahjongError, ResolvedConfig

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / 'configs'
DEFAULTS = asdict(AppConfig())
RESUME_ALLOWED = frozenset({
    'training.max_updates', 'observability.log_level',
    'checkpoint.every_updates', 'checkpoint.every_seconds',
    'validation.every_updates',
    'observability.heartbeat_seconds', 'observability.local_flush_seconds',
    'observability.resource_seconds', 'observability.tensorboard_enabled',
    'observability.tensorboard_flush_seconds',
})
EVALUATE_ALLOWED = (RESUME_ALLOWED - {'training.max_updates', 'checkpoint.every_updates', 'checkpoint.every_seconds',
                                    'validation.every_updates'}) | {
    'evaluation.seed', 'evaluation.batch_size', 'data.val_size',
    'runtime.device', 'runtime.deterministic', 'runtime.cpu_threads',
}


def _invalid(path, reason):
    raise MahjongError('CONFIG_INVALID', f'{path}: {reason}')


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if type(item) is dict:
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


LEAVES = flatten(DEFAULTS)


def _set(config, path, value):
    parts = path.split('.')
    target = config
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = value


def _check_tree(value, expected, prefix=''):
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if key not in expected:
            _invalid(path, 'unknown field')
        if type(expected[key]) is dict:
            if type(item) is not dict:
                _invalid(path, 'expected mapping')
            _check_tree(item, expected[key], path)


def _normalize(path, value):
    if path not in LEAVES:
        _invalid(path, 'unknown field')
    if isinstance(value, str) and '${' in value:
        _invalid(path, 'interpolation is unsupported')
    default = LEAVES[path]
    if path == 'evaluation.nanikiru_dir':
        if value is not None and (type(value) is not str or not value.strip()):
            _invalid(path, 'expected a nonempty directory path or null')
        return value
    if path == 'checkpoint.every_seconds':
        if value is None:
            return None
        if type(value) not in (int, float) or not math.isfinite(value):
            _invalid(path, 'expected a finite number or null')
        return float(value)
    if path in {'model.bos_token_id','model.eos_token_id','model.pad_token_id'}:
        if value is not None and type(value) is not int:
            _invalid(path, 'expected integer or null')
        return value
    if path == 'data.validation_fraction':
        if type(value) not in (int,float) or not math.isfinite(value):
            _invalid(path, 'expected a finite number')
        return float(value)
    if path in {'validation.every_updates', 'checkpoint.every_updates', 'data.validation_max_samples'}:
        if value is None:
            return None
        if type(value) is not int:
            _invalid(path, 'expected integer or null')
    elif path == 'training.scheduler':
        if value is not None:
            _invalid(path, 'only null is supported')
    elif path == 'optimizer.betas':
        if type(value) is not list or len(value) != 2:
            _invalid(path, 'expected two finite numeric betas')
        if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
            _invalid(path, 'expected two finite numeric betas')
        return [float(item) for item in value]
    elif type(default) is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            _invalid(path, 'expected a finite number')
        return float(value)
    elif type(value) is not type(default):
        _invalid(path, f'expected {type(default).__name__}')
    return value


def validate(config: ResolvedConfig) -> ResolvedConfig:
    """Check raw types before OmegaConf can convert them, then support/ranges."""
    if type(config) is not dict:
        _invalid('config', 'expected mapping')
    _check_tree(config, DEFAULTS)
    values = flatten(config)
    # Older checkpoints predate this optional evaluation input. Keep their
    # saved tree and integrity hash intact; assembly supplies the old default.
    missing = LEAVES.keys() - values.keys() - {'evaluation.nanikiru_dir'}
    if missing:
        _invalid(sorted(missing)[0], 'required field missing')
    result = deepcopy(config)
    for path, value in values.items():
        _set(result, path, _normalize(path, value))
    values = flatten(result)
    choices = {
        'runtime.device': {'cpu', 'cuda:0'}, 'runtime.precision': {'32-true', 'bf16-mixed'},
        'data.name': {'synthetic','majsoul_policy'},
        'data.input_schema': {'synthetic-regression/v1','majang-policy-ntp/v1','majang-policy-ntp/v2'},
        'model.name': {'mock_mlp','qwen3'},
        'model.input_schema': {'synthetic-regression/v1','token-causallm/v1','majang-policy-ntp/v1','majang-policy-ntp/v2'},
        'model.target_rule': {'linear-teacher/v1'}, 'optimizer.name': {'adam'},
        'observability.log_level': {'DEBUG', 'INFO', 'WARNING', 'ERROR'},
    }
    for path, options in choices.items():
        if values[path] not in options:
            _invalid(path, f'supported values: {sorted(options)}')
    positive = {
        'runtime.cpu_threads', 'data.train_size', 'data.input_dim', 'data.batch_size',
        'data.shared_max_tokens', 'data.shared_max_branch_cells',
        'model.input_dim', 'model.hidden_dim', 'optimizer.lr', 'optimizer.eps',
        'training.max_consecutive_skips', 'evaluation.batch_size',
        'observability.metrics_every_updates', 'observability.heartbeat_seconds',
        'observability.local_flush_seconds', 'observability.tensorboard_flush_seconds',
        'observability.resource_seconds',
    }
    for path in positive:
        if values[path] <= 0:
            _invalid(path, 'must be positive')
    if not 0 < values['data.validation_fraction'] < 1:
        _invalid('data.validation_fraction', 'must be between 0 and 1')
    if not 0 <= values['data.split_seed'] < 2**63:
        _invalid('data.split_seed', 'must be in [0,2**63)')
    for path in ('seed', 'evaluation.seed'):
        if not 0 <= values[path] < 2**63:
            _invalid(path, 'must be in [0, 2**63)')
    for path in ('data.val_size', 'training.max_updates', 'optimizer.weight_decay'):
        if values[path] < 0:
            _invalid(path, 'must be nonnegative')
    if not 0 <= values['model.dropout'] < 1:
        _invalid('model.dropout', 'must be in [0, 1)')
    if not all(0 <= beta < 1 for beta in values['optimizer.betas']):
        _invalid('optimizer.betas', 'each beta must be in [0, 1)')
    for path in ('validation.every_updates', 'checkpoint.every_updates', 'checkpoint.every_seconds', 'data.validation_max_samples'):
        if values[path] is not None and values[path] <= 0:
            _invalid(path, 'must be positive or null')
    for path, expected in [('data.num_workers', 0), ('training.accumulate_grad_batches', 1)]:
        if values[path] != expected:
            _invalid(path, f'only {expected} is supported')
    for path in ('experiment.name', 'output.root'):
        if not values[path].strip():
            _invalid(path, 'must be nonempty')
    if values['data.drop_last'] and values['data.train_size'] < values['data.batch_size']:
        _invalid('data.drop_last', 'would leave no training batches')
    if values['data.name'] == 'synthetic':
        if values['data.shared_context']:
            _invalid('data.shared_context', 'requires majsoul_policy data')
        if values['runtime.precision'] != '32-true':
            _invalid('runtime.precision', 'synthetic mock_mlp supports 32-true; BF16 is supported by Qwen3')
        if values['model.name'] != 'mock_mlp' or values['data.input_schema'] != 'synthetic-regression/v1':
            _invalid('data.name/model.name', 'synthetic training requires mock_mlp and synthetic-regression/v1')
        if values['model.input_dim'] != values['data.input_dim']:
            _invalid('model.input_dim', 'must match data.input_dim')
        if values['model.input_schema'] != values['data.input_schema']:
            _invalid('model.input_schema', 'must match data.input_schema')
    else:
        if not values['data.build_dir'].strip():
            _invalid('data.build_dir', 'required for majsoul_policy')
        if (values['model.name'] != 'qwen3' or
                values['data.input_schema'] not in ('majang-policy-ntp/v1', 'majang-policy-ntp/v2') or
                values['model.input_schema'] != values['data.input_schema']):
            _invalid('model/data input_schema', 'majsoul_policy requires Qwen3 policy candidate schema')
        from mahjong.models.qwen3 import SETTINGS_FIELDS, validate_settings
        validate_settings({key: values[f'model.{key}'] for key in SETTINGS_FIELDS})
        if values['data.shared_context']:
            if values['data.input_schema'] != 'majang-policy-ntp/v2':
                _invalid('data.shared_context', 'requires cross-round v2 input schema')
            if values['model.attn_implementation'] != 'sdpa':
                _invalid('model.attn_implementation', 'shared policy requires sdpa')
            if values['data.shared_max_tokens'] > values['model.max_position_embeddings']:
                _invalid('data.shared_max_tokens', 'must not exceed model.max_position_embeddings')
            if values['data.drop_last']:
                _invalid('data.drop_last', 'shared policy retains every tail group')
    return result


def parse_overrides(tokens, *, allowed=None, experiment=False):
    """Accept only documented single literal assignments, without coercion."""
    parser = OverridesParser.create()
    assignments = {}
    for token in tokens:
        if '${' in token or '=' not in token:
            _invalid(token, 'expected literal key=value; interpolation is unsupported')
        raw_value = token.split('=', 1)[1].lstrip()
        if raw_value[:1] not in {'"', "'"} and re.search(r'(^|[\[,])\s*[A-Za-z_]\w*\s*\(', raw_value):
            _invalid(token, 'functions are unsupported')
        try:
            parsed = parser.parse_override(token)
            if parsed.is_add() or parsed.is_force_add() or parsed.is_delete() or parsed.is_sweep_override() or parsed.package is not None:
                _invalid(token, 'only ordinary single assignments are supported')
            path, value = parsed.key_or_group, parsed.value()
        except MahjongError:
            raise
        except Exception as exc:
            _invalid(token, f'invalid override: {exc}')
        if path in assignments:
            _invalid(path, 'duplicate override')
        if path == 'experiment' and experiment:
            if value not in ('mock','mahjong_policy'):
                _invalid(path, 'unsupported experiment config')
        else:
            if allowed is not None and path not in allowed:
                _invalid(path, 'explicit override is not allowed for this operation')
            value = _normalize(path, value)
        assignments[path] = value
    return assignments


def _read_yaml(path):
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        _invalid(str(path), f'cannot read configuration: {exc}')
    if type(value) is not dict:
        _invalid(str(path), 'expected YAML mapping')
    return value


def _check_raw_yaml(root, config_name='train'):
    if config_name not in ('train','pretrain'):
        _invalid('config', 'unsupported training config entry')
    root_value = _read_yaml(root / f'{config_name}.yaml')
    selections = (
        [('runtime','default'),('data','synthetic'),('model','mock_mlp')]
        if config_name == 'train' else
        [('runtime','bf16'),('data','majsoul_policy'),('model','qwen3_policy')]
    )
    experiment_choice = 'mock' if config_name == 'train' else 'mahjong_policy'
    expected_defaults = ['phase1_schema', '_self_'] + [
        {name: choice} for name, choice in [
            *selections,
            ('optimizer', 'adam'), ('training', 'default'), ('validation', 'default'),
            ('evaluation', 'default'), ('checkpoint', 'default'),
            ('observability', 'local'), ('output', 'local'), ('experiment', experiment_choice),
        ]
    ]
    if root_value.get('defaults') != expected_defaults:
        _invalid('defaults', 'must match the phase 1 composition order')
    for key, value in flatten({k: v for k, v in root_value.items() if k != 'defaults'}).items():
        _normalize(key, value)
    _check_tree({k: v for k, v in root_value.items() if k != 'defaults'}, DEFAULTS)
    for item in expected_defaults[2:]:
        group, choice = next(iter(item.items()))
        path = root / group / f'{choice}.yaml'
        value = _read_yaml(path)
        prefix = '' if group == 'experiment' else group
        _check_tree(value, DEFAULTS if group == 'experiment' else DEFAULTS[group], prefix)
        for key, leaf in flatten(value, prefix).items():
            _normalize(key, leaf)


def parse_new(overrides: list[str], cwd: Path, *, config_name='train') -> ResolvedConfig:
    parse_overrides(overrides, experiment=True)
    _check_raw_yaml(CONFIG_ROOT, config_name)
    ConfigStore.instance().store(name='phase1_schema', node=AppConfig)
    try:
        with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base='1.3'):
            cfg = compose(config_name=config_name, overrides=overrides)
        result = OmegaConf.to_container(cfg, resolve=False)
    except Exception as exc:
        _invalid('config', f'composition failed: {exc}')
    # Validate once more after composition; raw YAML and CLI were checked first.
    result = validate(result)
    result['output']['root'] = str((cwd / result['output']['root']).resolve())
    return result


def merge_resume(saved: ResolvedConfig, overrides: list[str]):
    result = deepcopy(saved)
    assignments = parse_overrides(overrides, allowed=RESUME_ALLOWED)
    changes = []
    old_values = flatten(saved)
    for path, value in assignments.items():
        _set(result, path, value)
        if old_values[path] != value:
            changes.append(ConfigChange(path, old_values[path], value, 'explicit allowed override'))
    result = validate(result)
    if result['training']['max_updates'] < saved['training']['max_updates']:
        _invalid('training.max_updates', 'cannot reduce the saved total target')
    return result, tuple(changes)


def merge_evaluate(saved: ResolvedConfig, overrides: list[str], cwd: Path):
    # The evaluate config is sparse and contains no train defaults.
    sparse = _read_yaml(CONFIG_ROOT / 'evaluate.yaml')
    if sparse.get('defaults') != ['_self_']:
        _invalid('defaults', 'evaluate config must contain only _self_')
    _check_tree({key: value for key, value in sparse.items() if key != 'defaults'}, DEFAULTS)
    values = flatten({key: value for key, value in sparse.items() if key != 'defaults'})
    result = deepcopy(saved)
    for path, value in values.items():
        if path not in EVALUATE_ALLOWED:
            _invalid(path, 'not allowed in sparse evaluate config')
        _normalize(path, value)
    try:
        with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base='1.3'):
            composed = compose(config_name='evaluate')
        sparse_values = OmegaConf.to_container(composed, resolve=False)
    except Exception as exc:
        _invalid('config', f'evaluate composition failed: {exc}')
    for path, value in flatten(sparse_values).items():
        _set(result, path, _normalize(path, value))
    for path, value in parse_overrides(overrides, allowed=EVALUATE_ALLOWED).items():
        _set(result, path, value)
    return validate(result)


def frozen_config(config):
    # Preserve the integrity-hash basis of existing checkpoints. Cadence is now
    # resumable through an audited ConfigChange, but stays in this legacy hash.
    return {path: value for path, value in flatten(config).items()
            if path not in RESUME_ALLOWED or path == 'validation.every_updates'}



def parse_model(overrides: list[str] | None = None) -> ResolvedConfig:
    """Compose the model-only Hydra entry without relaxing the training schema."""
    from mahjong.models.qwen3 import SETTINGS_FIELDS, validate_settings
    tokens = overrides or []
    parser = OverridesParser.create()
    seen = set()
    allowed = {'seed', 'model'} | {f'model.{field}' for field in SETTINGS_FIELDS}
    for token in tokens:
        try:
            item = parser.parse_override(token)
            path = item.key_or_group
            if (path not in allowed or path in seen or item.is_add() or item.is_force_add()
                    or item.is_delete() or item.is_sweep_override() or item.package is not None
                    or '${' in token or re.search(r'\b\w+\s*\(', token)):
                _invalid(token, 'expected unique literal model field override')
            seen.add(path)
            if path == 'model' and item.value() != 'qwen3':
                _invalid(path, 'model-only entry supports qwen3')
        except MahjongError:
            raise
        except Exception as exc:
            _invalid(token, f'invalid override: {exc}')
    try:
        with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base='1.3'):
            composed = compose(config_name='model_only', overrides=tokens)
        result = OmegaConf.to_container(composed, resolve=False)
    except Exception as exc:
        _invalid('model_only', f'composition failed: {exc}')
    if result.keys() != {'model', 'seed'}:
        _invalid('model_only', 'expected model and seed only')
    if type(result['seed']) is not int or not 0 <= result['seed'] < 2**63:
        _invalid('seed', 'expected integer in [0,2**63)')
    validate_settings(result['model'])
    return result
