"""Select native scoring interpretations with explicit payment/display policy.

The caller supplies canonical yaku token keys; no tokenizer or hand solver is
implemented here. Basic receipts exclude honba, kyotaku and final adjustments.
"""
from .offline_mjai import check


def select_win_values(values,keys,*,dealer,payment_rules=None):
    check(bool(values) and len(values)==len(keys),'TYPE_MISMATCH','scoring interpretations')
    receipts=[]
    for value in values:
        if payment_rules is not None:
            amount=payment_rules.win([0]*4,dealer=dealer,honba=0,kyotaku=0,claims=[value])['score_delta'][value.actor]
        else:
            # Fixed gimite standard score semantics. This does not supply a
            # missing payment breakdown or admit an unknown historical profile.
            from riichienv import calculate_score
            han=13 if value.yakuman else min(value.han,13)
            amount=calculate_score(han,0 if value.yakuman else value.fu,value.actor==dealer,value.target is None,0).total
            if value.yakuman:amount*=sum(c['yakuman_units'] for c in value.yakuman)
        receipts.append(amount)
    def key(index):
        value=values[index]
        return (-receipts[index],-sum(c['yakuman_units'] for c in value.yakuman),
                -(value.han or 0),-(value.fu or 0),tuple(keys[index]))
    selected=min(range(len(values)),key=key)
    return dict(selected=selected,receipts=receipts,
                maximal=[i for i,p in enumerate(receipts) if p==receipts[selected]])
