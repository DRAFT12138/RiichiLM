"""Strict RL configuration, independent of the supervised Hydra schema."""
from copy import deepcopy
import math
from pathlib import Path
from omegaconf import OmegaConf
from mahjong.contracts import MahjongError

DEFAULT_PATH = Path(__file__).resolve().parents[3] / 'configs' / 'rl.yaml'


def resolve(path=None, overrides=()):
    default = OmegaConf.to_container(OmegaConf.load(DEFAULT_PATH), resolve=True)
    actual = deepcopy(default)
    if path:
        supplied = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        def merge(a, b):
            if not isinstance(b, dict): raise MahjongError('CONFIG_INVALID', 'RL configuration group must be a mapping')
            for k, v in b.items():
                if k not in a: raise MahjongError('CONFIG_INVALID', f'unknown RL field {k}')
                if isinstance(a[k], dict): merge(a[k], v)
                else: a[k] = v
        merge(actual, supplied)
    for override in overrides:
        try:
            key, value = override.split('=', 1)
            parts, node = key.removeprefix('rl.').split('.'), actual
            for part in parts[:-1]: node = node[part]
            if parts[-1] not in node: raise KeyError(parts[-1])
            # YAML scalar/list parsing, rather than retaining numeric strings.
            import yaml
            node[parts[-1]] = yaml.safe_load(value)
        except (KeyError, ValueError, TypeError) as exc:
            raise MahjongError('CONFIG_INVALID', f'invalid RL override {override}') from exc
    validate(actual, default)
    return actual


def validate(actual, template=None):
    if template is None: template = OmegaConf.to_container(OmegaConf.load(DEFAULT_PATH), resolve=True)
    def visit(a, b, path='rl'):
        if not isinstance(a, dict) or a.keys() != b.keys():
            raise MahjongError('CONFIG_INVALID', f'{path}: missing or unknown fields')
        for k, expected in b.items():
            value, key = a[k], path+'.'+k
            if isinstance(expected, dict): visit(value, expected, key)
            elif isinstance(expected, bool):
                if type(value) is not bool: raise MahjongError('CONFIG_INVALID', key+' requires bool')
            elif isinstance(expected, int):
                if type(value) is not int or value < 0: raise MahjongError('CONFIG_INVALID', key+' requires nonnegative int')
            elif isinstance(expected, float):
                if type(value) not in (int, float) or not math.isfinite(value): raise MahjongError('CONFIG_INVALID', key+' requires finite number')
            elif isinstance(expected, list):
                if not isinstance(value, list) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value):
                    raise MahjongError('CONFIG_INVALID', key+' requires nonnegative numeric list')
    visit(actual, template)
    p, c, v, pool, e = (actual[k] for k in ('ppo', 'control', 'critic', 'pool', 'evaluation'))
    positive = [actual['rounds'], *[n for k, n in actual['collection'].items() if k != 'max_failed_fraction'],
                p['epochs'], p['batch_size'], p['max_updates'], p['actor_lr'], p['critic_lr'],
                c['actor_grad_norm'], c['critic_grad_norm'], c['probe_size'], c['min_probe_samples'], c['probe_interval'], c['score_tolerance'],
                v['oracle_width'], v['value_width'], v['warmup_rounds'], v['min_matches'], v['consecutive_windows'],
                v['max_rmse'], pool['history_interval'], pool['byte_budget'], e['seed_groups'], e['bootstrap_samples']]
    if (any(n <= 0 for n in positive) or not 0 < p['clip'] < 1 or not 0 <= p['lam'] <= 1
            or min(p['beta'], p['c_v'], p['c_e'], v['max_bias'], e['regression_tolerance']) < 0
            or not 0 < c['soft_kl'] < c['hard_kl'] or c['max_ratio'] <= 1
            or c['probe_size'] < c['min_probe_samples'] or pool['active_capacity'] < 2
            or pool['physical_capacity'] != pool['active_capacity'] + 2
            or len(pool['table_weights']) != 3 or not math.isclose(sum(pool['table_weights']), 1)
            or not 0 < e['confidence'] < 1 or e['monitor_seed'] == e['heldout_seed']
            or not 0 <= actual['collection']['max_failed_fraction'] < 1 or not 0 <= e['max_failed_fraction'] < 1
            or actual['collection']['max_attempts'] < actual['collection']['train_matches']+actual['collection']['validation_matches']):
        raise MahjongError('CONFIG_INVALID', 'inconsistent RL budget, threshold or sampling configuration')
    for field in ('parameter_check_rounds', 'full_readback_rounds'):
        if any(type(x) is not int for x in c[field]):
            raise MahjongError('CONFIG_INVALID', 'check rounds must be integers')
    return actual
