"""Model-owned full-action benchmark metrics using the production decoder."""
from mahjong.contracts import MahjongError, MetricStat, StepResult
from mahjong.inference.policy import DecisionKey, DecisionView, LegalMove, PolicyInference, WindowKey
from mahjong.policy.shared_context import NANIKIRU_LAYOUT
from mahjong.tokenizer import MahjongTokenizer
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, PolicyContextCodec


def is_nanikiru(batch):
    return type(batch.payload) is dict and batch.payload.get('layout') == NANIKIRU_LAYOUT


def prepare_nanikiru(model, batch):
    if (model.input_schema != CONTEXT_SCHEMA or batch.input_schema != CONTEXT_SCHEMA or
            batch.sample_count != 1 or batch.payload.keys() != {
                'layout', 'input_ids', 'candidates', 'target_indices', 'answer_kind', 'sample_id', 'coverage'}):
        raise MahjongError('CONTRACT_MISMATCH', 'Nanikiru requires one v2 evaluation question', 'evaluating')
    return batch


def evaluate_nanikiru(model, batch, *, phase):
    if phase != 'eval':
        raise MahjongError('CONTRACT_MISMATCH', 'Nanikiru is evaluation-only', 'training')
    prepare_nanikiru(model, batch)
    p = batch.payload
    codec = MahjongTokenizer()
    key = DecisionKey(WindowKey('nanikiru', p['sample_id'], 0, 0, 0), 0)
    view = DecisionView(key, PolicyContextCodec().decode_policy_prefix(p['input_ids']),
        tuple(LegalMove(codec.decode_action_suffix(s), str(i)) for i, s in enumerate(p['candidates'])),
        NANIKIRU_LAYOUT, True)
    # PolicyInference sets eval mode. Preserve every module flag even when this
    # adapter is called directly, outside Runtime's protected evaluation scope.
    modes = [(module, module.training) for module in model.modules()]
    try:
        inference = PolicyInference(model)
        generated = inference.generate(inference.prepare(view))
    finally:
        for module, training in modes:
            module.training = training
    # prepare() sorts candidates. Compare complete suffixes, never its reordered
    # candidate index; multiple targets include alternative tiles/tsumogiri.
    accepted = {p['candidates'][i] for i in p['target_indices']}
    correct = float(generated.suffix_ids in accepted)
    metrics = {'nanikiru/action_accuracy': MetricStat(correct, 1, 'ratio'),
               f"nanikiru/{p['answer_kind']}_accuracy": MetricStat(correct, 1, 'ratio')}
    if p['coverage'] is not None:
        ready, total = p['coverage']
        metrics['nanikiru/coverage'] = MetricStat(ready, total, 'ratio')
    return StepResult(None, metrics)
