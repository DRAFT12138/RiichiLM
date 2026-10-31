"""Whole-episode value statistics and executable policy safety checks."""
import math
from mahjong.contracts import MahjongError


def value_statistics(values, returns):
    if not values or len(values) != len(returns) or not all(math.isfinite(v) for v in [*values, *returns]):
        raise MahjongError('CONTRACT_MISMATCH', 'invalid value diagnostics')
    n = len(values)
    mean = math.fsum(returns)/n
    errors = [v-r for v, r in zip(values, returns)]
    bias = math.fsum(errors)/n
    var = math.fsum((r-mean)**2 for r in returns)/n
    error_var = math.fsum((e-bias)**2 for e in errors)/n
    mse = math.fsum(e*e for e in errors)/n
    value_mean = math.fsum(values)/n
    return dict(samples=n, ev=1-error_var/var if var > 1e-12 else None,
                reason=None if var>1e-12 else 'zero_return_variance', rmse=math.sqrt(mse), bias=bias,
                return_mean=mean, return_std=math.sqrt(var), value_mean=value_mean,
                value_std=math.sqrt(math.fsum((v-value_mean)**2 for v in values)/n),
                constant_baseline_rmse=math.sqrt(var))


def value_gate(stats, matches, config):
    return (matches >= config['min_matches'] and stats['ev'] is not None and stats['ev'] >= config['min_ev']
            and stats['rmse'] <= config['max_rmse'] and abs(stats['bias']) <= config['max_bias'])


def check_ratios(result, config):
    for name, stat in result.metrics.items():
        if not math.isfinite(stat.numerator):
            raise MahjongError('NONFINITE_LOSS', 'nonfinite RL diagnostic '+name)
    hi, lo = result.metrics.get('ratio_max'), result.metrics.get('ratio_min')
    if hi and (hi.numerator > config['max_ratio'] or lo.numerator < 1/config['max_ratio']):
        raise MahjongError('CONTRACT_MISMATCH', 'observed action ratio exceeds hard budget')


def probe_decision(kl, count, config):
    if count < config['min_probe_samples']:
        raise MahjongError('EVALUATION_FAILED', 'fixed policy probe has insufficient nonforced samples')
    if not math.isfinite(kl) or kl > config['hard_kl']:
        raise MahjongError('EVALUATION_FAILED', 'fixed policy probe exceeds hard KL budget')
    return kl > config['soft_kl']
