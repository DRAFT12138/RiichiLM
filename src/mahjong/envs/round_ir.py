"""Canonical round IR conversions, independent of source orchestration."""
from copy import deepcopy

from mahjong.tokenizer import Atom
from .mjai_ir import UNKNOWN


def abs_seat(actor):
    return Atom('ABS_P'+str(actor))


def tile(value):
    return Atom('TILE_'+value)


def furiten_values(values):
    return {k:UNKNOWN if v is None else v for k,v in values.items()}


def canonical_settlement(value):
    """Translate environment transfers, merging/order already fixed by rules."""
    from mahjong.tokenizer import MahjongTokenizer
    result=deepcopy(value)
    for payment in result['payments']:
        for name in ('payer','payee'):
            payment[name]=None if payment[name] is None else abs_seat(payment[name])
        payment['payment_kind']=Atom('PAYMENT_'+payment['payment_kind'])
    codec=MahjongTokenizer()
    result['payments'].sort(key=lambda p:tuple(codec.token_id('NULL' if p[k] is None else p[k].name)
                                              for k in ('payer','payee','payment_kind')))
    return result
