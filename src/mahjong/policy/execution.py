"""One execution definition for RL sampling, rescoring and Actor snapshots."""
from contextlib import contextmanager
import torch
from mahjong.contracts import MahjongError


def policy_precision(actor):
    return getattr(actor, '_policy_precision', '32-true')


def set_policy_precision(actor, precision):
    if precision not in ('32-true', 'bf16-mixed'):
        raise MahjongError('CONTRACT_MISMATCH', 'unsupported policy execution precision')
    actor._policy_precision = precision


def sampling_definition(precision, strategy='categorical'):
    if precision not in ('32-true', 'bf16-mixed'):
        raise MahjongError('CONTRACT_MISMATCH', 'unsupported sampling precision')
    return dict(strategy=strategy, temperature=1, precision=precision, trie='complete')


@contextmanager
def policy_execution(actor):
    precision = policy_precision(actor)
    device = next(actor.parameters()).device
    if precision == '32-true':
        yield
        return
    if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise MahjongError('DEVICE_UNAVAILABLE', 'BF16 RL policy execution requires a supported GPU')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    # Never silently select the quadratic math backend for long GPU sequences.
    with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        yield
