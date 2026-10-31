"""Explicit declaration and committed-call policy; native owns ippatsu state."""
from dataclasses import dataclass

from .offline_mjai import check


@dataclass(frozen=True)
class RiichiPolicy:
    deduct_at: str
    allow_furiten: bool
    allow_open: bool
    ippatsu_cancel_on_call: bool
    ippatsu_cancel_on_ankan: bool

    def __post_init__(self):
        check(self.deduct_at=='accept','UNSUPPORTED_PROFILE','riichi deduct_at')
        for name in ('allow_furiten','allow_open','ippatsu_cancel_on_call','ippatsu_cancel_on_ankan'):
            check(type(getattr(self,name)) is bool,'TYPE_MISMATCH',name)
        check(not self.allow_open,'UNSUPPORTED_PROFILE','open riichi')

    @classmethod
    def from_dict(cls,value):
        check(type(value) is dict and value.keys()==cls.__dataclass_fields__.keys(),
              'TYPE_MISMATCH','riichi_policy requires all five parameters')
        return cls(**value)

    def cancel_on(self,kind):
        return self.ippatsu_cancel_on_ankan if kind=='ankan' else self.ippatsu_cancel_on_call
