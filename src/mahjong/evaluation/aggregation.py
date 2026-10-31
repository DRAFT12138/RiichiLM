"""One weighted-metric implementation shared by evaluation and observation."""
from collections.abc import Mapping
import math

from mahjong.contracts import (AggregatedMetric, MahjongError, MetricStat, State,
                              clone_state, validate_contract)


class MetricAccumulator:
    def __init__(self):
        self._metrics: dict[str, dict] = {}

    def add(self, stats: Mapping[str, MetricStat]) -> None:
        changes = {}
        for name, stat in stats.items():
            if type(name) is not str or not name:
                raise MahjongError('CONTRACT_MISMATCH', 'Metric name must be nonempty')
            validate_contract(stat)
            prior = self._metrics.get(name)
            if prior is not None and prior['unit'] != stat.unit:
                raise MahjongError('CONTRACT_MISMATCH', f'Metric {name}: unit changed')
            numerator = float(stat.numerator) + (0.0 if prior is None else prior['numerator'])
            denominator = stat.denominator + (0 if prior is None else prior['denominator'])
            if not math.isfinite(numerator):
                raise MahjongError('CONTRACT_MISMATCH', f'Metric {name}: nonfinite sum')
            changes[name] = {'numerator': numerator, 'denominator': denominator, 'unit': stat.unit}
        self._metrics.update(changes)

    def result(self) -> Mapping[str, AggregatedMetric]:
        return {name: AggregatedMetric(stat['numerator'], stat['denominator'],
                                       stat['numerator']/stat['denominator'] if stat['denominator'] else None,
                                       stat['unit'], None if stat['denominator'] else 'no_valid_statistics')
                for name, stat in self._metrics.items()}

    def state_dict(self) -> State:
        return clone_state({'schema_version': 1, 'metrics': self._metrics})

    def load_state_dict(self, state: State) -> None:
        if type(state) is not dict or set(state) != {'schema_version', 'metrics'} or type(state['schema_version']) is not int or state['schema_version'] != 1:
            raise MahjongError('CHECKPOINT_INVALID', 'MetricAccumulator state version/fields invalid', 'restoring')
        if type(state['metrics']) is not dict:
            raise MahjongError('CHECKPOINT_INVALID', 'MetricAccumulator metrics must be a dict', 'restoring')
        restored = MetricAccumulator()
        try:
            stats = {}
            for name, value in state['metrics'].items():
                if type(value) is not dict or set(value) != {'numerator', 'denominator', 'unit'}:
                    raise MahjongError('CONTRACT_MISMATCH', 'Metric accumulator fields invalid')
                stats[name] = MetricStat(value['numerator'], value['denominator'], value['unit'])
            restored.add(stats)
        except MahjongError as exc:
            raise MahjongError('CHECKPOINT_INVALID', str(exc), 'restoring') from exc
        self._metrics = clone_state(restored._metrics)

    def reset(self) -> None:
        self._metrics.clear()
