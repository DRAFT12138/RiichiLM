"""Enabled native abort predicates and explicit resolution priority."""
from dataclasses import dataclass

from .offline_mjai import check

NATIVE_ABORTS={'sufuurenta':('four_winds','suufon'),
               'suucha_riichi':('four_riichi','suucha_riichi'),
               'suukansansen':('four_kan','suukaikan')}
PRIORITIES=frozenset(('ron','four_kan','four_riichi','four_winds','exhaustive'))


@dataclass(frozen=True)
class AbortRules:
    kyuushu: bool
    suufon: bool
    suucha_riichi: bool
    suukaikan: bool
    priority: list
    four_kan_abort: str

    def __post_init__(self):
        for name in ('kyuushu','suufon','suucha_riichi','suukaikan'):
            check(type(getattr(self,name)) is bool,'TYPE_MISMATCH',name)
        check(type(self.priority) is list and len(self.priority)==5 and
              all(type(v) is str for v in self.priority) and set(self.priority)==PRIORITIES,
              'TYPE_MISMATCH','abort priority permutation')
        check(self.four_kan_abort=='after_discard_response_if_multiple_owners',
              'UNSUPPORTED_PROFILE','immediate four-kan abort is not implemented')

    @classmethod
    def from_dict(cls,value):
        check(type(value) is dict and value.keys()==cls.__dataclass_fields__.keys(),
              'TYPE_MISMATCH','abort_rules requires all six parameters')
        return cls(**value)

    def enabled(self,native):
        return [kind for code,(kind,flag) in NATIVE_ABORTS.items() if code in native and getattr(self,flag)]

    def check_choice(self,choice,available):
        check(choice in available and choice==min(available,key=self.priority.index),
              'RESULT_DETAILS_MISMATCH','result disagrees with explicit abort priority')
