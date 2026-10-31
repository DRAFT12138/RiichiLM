"""Shared backward/step boundary; optional RL group clipping and checks."""
import torch
from mahjong import contracts as c


def execute_update(runtime, result, *, groups=None, limits=None, before_step=None,
                   after_backward=None):
    c.validate_step_result(result, phase='train')
    if result.skip_reason is not None:
        return result
    if before_step is not None:
        before_step(result)
    runtime.backward(result.loss)
    if after_backward is not None:
        after_backward()
    runtime.check_gradients()
    extra = {}
    if groups is not None:
        for name, parameters in groups:
            parameters = [p for p in parameters if p.grad is not None]
            if parameters:
                norm = torch.nn.utils.clip_grad_norm_(parameters, limits[name], error_if_nonfinite=True)
                extra[f'{name}_grad_norm'] = c.MetricStat(float(norm), 1, 'norm')
                extra[f'{name}_clipped_grad_norm'] = c.MetricStat(min(float(norm), limits[name]), 1, 'norm')
    runtime.step()
    if groups is not None and any(not bool(torch.isfinite(p).all()) for p in runtime.raw_model.parameters()):
        raise c.MahjongError('NONFINITE_GRADIENT', 'updated RL parameters are nonfinite')
    return c.StepResult(result.loss, {**result.metrics, **extra}, result.skip_reason)
