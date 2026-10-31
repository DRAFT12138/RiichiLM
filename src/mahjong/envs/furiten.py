"""Offline response knowledge; hand/role eligibility stays in the native engine.

None is unknown, never false. This ledger does not infer or emit action labels.
"""
from dataclasses import dataclass

from .offline_mjai import check


def known_or(left,right):
    if left is True or right is True:return True
    if left is None or right is None:return None
    return False


@dataclass(frozen=True)
class FuritenRules:
    miss_ron_on_call: bool
    temp_clear: str
    riichi_clear: str
    response_collection: str

    def __post_init__(self):
        check(type(self.miss_ron_on_call) is bool,'TYPE_MISMATCH','miss_ron_on_call')
        check(self.temp_clear=='own_draw','UNSUPPORTED_PROFILE','furiten temp clear')
        check(self.riichi_clear=='round_start','UNSUPPORTED_PROFILE','furiten riichi clear')
        check(self.response_collection in ('all_before_resolution','priority_short_circuit'),
              'UNSUPPORTED_PROFILE','furiten response collection')

    @classmethod
    def from_dict(cls,value):
        check(type(value) is dict and value.keys()=={'miss_ron_on_call','temp_clear','riichi_clear','response_collection'},
              'TYPE_MISMATCH','furiten_rules requires all four explicit parameters')
        return cls(**value)


class FuritenKnowledge:
    def __init__(self,rules):
        self.rules=rules
        self.reset()

    def reset(self):
        self.temp=[False]*4
        self.riichi=[False]*4

    def own_draw(self,actor):
        self.temp[actor]=False

    def observe(self,actor,*,riichi,ron_available,choice):
        """Consume proven RON/other response, proven no-ron, or no observation.

        NO_RON is a public-resolution proof under an explicit full-collection
        policy. It is not a fabricated PASS or a private intention.
        """
        check(type(actor) is int and 0<=actor<4,'TYPE_MISMATCH','response actor')
        check(riichi is None or type(riichi) is bool,'TYPE_MISMATCH','response riichi knowledge')
        check(ron_available is None or type(ron_available) is bool,'TYPE_MISMATCH','ron availability')
        check(choice in (None,'PASS','RON','CHI','PON','DAIMINKAN','NO_RON'),
              'TYPE_MISMATCH','response choice')
        if ron_available is False or choice=='RON':return False
        if choice in ('CHI','PON','DAIMINKAN') and not self.rules.miss_ron_on_call:return False
        missed=True if ron_available is True and choice is not None else None
        if riichi is None:
            self.temp[actor]=known_or(self.temp[actor],None)
            self.riichi[actor]=known_or(self.riichi[actor],None)
        else:
            states=self.riichi if riichi else self.temp
            states[actor]=known_or(states[actor],missed)
        return missed
