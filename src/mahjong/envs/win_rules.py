"""Explicit4P scoring options; decomposition and yaku remain native."""
from dataclasses import dataclass

from .offline_mjai import check


@dataclass(frozen=True)
class WinRules:
    kuitan: bool
    atozuke: bool
    min_yaku_han: int
    double_wind_pair_fu: int
    pinfu_tsumo_fu: int
    chiitoitsu_fu: int

    def __post_init__(self):
        for name in ('kuitan','atozuke'):
            check(type(getattr(self,name)) is bool,'TYPE_MISMATCH',name)
        check(self.atozuke,'UNSUPPORTED_PROFILE','atozuke=false requires prior opportunity semantics')
        check(type(self.min_yaku_han) is int and self.min_yaku_han>=1,'TYPE_MISMATCH','min_yaku_han')
        check(self.min_yaku_han<=13,'UNSUPPORTED_PROFILE','min_yaku_han above13')
        for name,values in (('double_wind_pair_fu',(2,4)),('pinfu_tsumo_fu',(20,)),('chiitoitsu_fu',(25,))):
            value=getattr(self,name)
            check(type(value) is int and value in values,'UNSUPPORTED_PROFILE',name)

    @classmethod
    def from_dict(cls,value):
        check(type(value) is dict and value.keys()==cls.__dataclass_fields__.keys(),
              'TYPE_MISMATCH','win_rules requires all six parameters')
        return cls(**value)

    def conditions(self):
        return dict(kuitan=self.kuitan,min_yaku_han=self.min_yaku_han,
                    double_wind_pair_fu=self.double_wind_pair_fu)
