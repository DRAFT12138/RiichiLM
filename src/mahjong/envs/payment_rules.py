"""Payments from explicit result facts, independent of hand/yaku adjudication.

Callers must supply verified win values and liability, or verified draw facts.
This component never derives missing facts from a recorded net score delta.
Seat numbers and payment kinds are environment values, not tokenizer objects.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

from .offline_mjai import check

NAGASHI_NO_HONBA_CARRY_POOL = 'no_honba_carry_pool'


def natural(value, name):
    check(type(value) is int and value >= 0, 'TYPE_MISMATCH', name)


def seat(value, name):
    check(type(value) is int and 0 <= value < 4, 'TYPE_MISMATCH', name)


def balances(scores, kyotaku):
    check(type(scores) is list and len(scores) == 4 and all(type(s) is int for s in scores),
          'TYPE_MISMATCH', 'scores')
    natural(kyotaku, 'kyotaku')


class Transfers:
    def __init__(self):
        self.amounts = defaultdict(int)

    def add(self, payer, payee, amount, kind):
        natural(amount, 'payment amount')
        check(payer != payee, 'INVALID_REFERENCE', 'self payment')
        if amount:
            self.amounts[payer, payee, kind] += amount

    def settlement(self, scores, kyotaku):
        balances(scores, kyotaku)
        delta, pool = [0] * 4, 1000 * kyotaku
        payments = []
        # Preserve reciprocal payments; only identical directed keys merge.
        order = lambda item: tuple(-1 if v is None else v for v in item[:2]) + (item[2],)
        for payer, payee, kind in sorted(self.amounts, key=order):
            amount = self.amounts[payer, payee, kind]
            if payer is None:
                pool -= amount
            else:
                delta[payer] -= amount
            if payee is None:
                pool += amount
            else:
                delta[payee] += amount
            payments.append(dict(payer=payer, payee=payee, amount=amount, payment_kind=kind))
        check(pool >= 0 and pool % 1000 == 0, 'SETTLEMENT_MISMATCH', 'pool balance')
        return dict(scores_before=list(scores), scores_after=[s+d for s,d in zip(scores, delta)],
                    score_delta=delta, kyotaku=kyotaku, next_kyotaku=pool // 1000, payments=payments)


@dataclass(frozen=True)
class WinValue:
    actor: int
    target: int | None  # None is tsumo, otherwise the discarder/kan declarer.
    han: int | None = None
    fu: int | None = None
    # Each explicit component: name, yakuman_units, pao_player (None = no liability).
    yakuman: tuple = ()
    # Verified source-policy fact, required for nonzero liability honba.
    # None leaves that allocation unknown; it is not inferred from base shares.
    honba_payer: int | None = None


class PaymentRules:
    def __init__(self, value):
        from .rule_profile import YAKUMAN, object_fields, boolean, choice, integer
        object_fields(value, 'policy multiple_ron kazoe kiriage yakuman_stack yakuman_multipliers', 'payment_rules')
        policy = value['policy']
        object_fields(policy, 'honba_ron_each honba_tsumo_each multi_ron_honba multi_ron_kyotaku '
                      'pao_scope pao_ron pao_tsumo noten_total nagashi nagashi_noten final_kyotaku', 'payment policy')
        for name in ('honba_ron_each', 'honba_tsumo_each', 'noten_total'):
            integer(policy[name], name, minimum=0)
        for name, values in {
            'multi_ron_honba': ('each', 'nearest'), 'multi_ron_kyotaku': ('nearest',),
            'pao_scope': ('liable_units', 'all_yakuman'), 'pao_ron': ('half_with_discarder',),
            'pao_tsumo': ('liable_pays_all',), 'nagashi': ('independent_tsumo', 'disabled'),
            'final_kyotaku': ('first_rank', 'carry'),
        }.items():
            choice(policy[name], values, name)
        boolean(policy['nagashi_noten'], 'nagashi_noten')
        choice(value['multiple_ron'], ('MULTIRON_ALL', 'MULTIRON_HEAD_BUMP', 'MULTIRON_TRIPLE_ABORT'), 'multiple_ron')
        choice(value['kazoe'], ('KAZOE_YAKUMAN', 'KAZOE_SANBAIMAN', 'KAZOE_NONE'), 'kazoe')
        check(value['kazoe'] != 'KAZOE_NONE', 'UNSUPPORTED_PROFILE', 'uncapped kazoe scoring is not implemented')
        for name in ('kiriage', 'yakuman_stack'):
            boolean(value[name], name)
        multipliers = value['yakuman_multipliers']
        check(type(multipliers) is list and len(multipliers) == len(YAKUMAN), 'TYPE_MISMATCH', 'yakuman multipliers')
        for item in multipliers:
            object_fields(item, 'name count', 'multiplier')
            choice(item['name'], YAKUMAN, 'multiplier name')
            integer(item['count'], 'multiplier count', minimum=1)
        check([m['name'] for m in multipliers] == list(YAKUMAN), 'NONCANONICAL', 'multiplier order')
        self.value = deepcopy(value)
        self.policy = self.value['policy']
        self.multipliers = {m['name']: m['count'] for m in multipliers}

    def _validate_win(self, win):
        check(type(win) is WinValue, 'TYPE_MISMATCH', 'WinValue')
        seat(win.actor, 'winner')
        if win.target is not None:
            seat(win.target, 'target')
            check(win.target != win.actor, 'INVALID_REFERENCE', 'ron target is winner')
        check(type(win.yakuman) is tuple, 'TYPE_MISMATCH', 'yakuman components')
        if not win.yakuman:
            check(type(win.han) is int and win.han > 0, 'TYPE_MISMATCH', 'positive known han')
            check(type(win.fu) is int and (win.fu in (20, 25) or 30 <= win.fu <= 250 and win.fu % 10 == 0),
                  'TYPE_MISMATCH', 'known rounded fu')
        else:
            check(win.han is None and win.fu is None, 'TYPE_MISMATCH', 'yakuman han/fu must be inapplicable')
            names = []
            for item in win.yakuman:
                check(type(item) is dict and item.keys() == {'name', 'yakuman_units', 'pao_player'},
                      'TYPE_MISMATCH', 'yakuman payment component')
                name, units, liable = item['name'], item['yakuman_units'], item['pao_player']
                check(type(name) is str and name in self.multipliers, 'TYPE_MISMATCH', 'yakuman name')
                check(type(units) is int and units == self.multipliers[name], 'SETTLEMENT_MISMATCH', 'configured yakuman units')
                if liable is not None:
                    seat(liable, 'pao player')
                    check(liable != win.actor, 'INVALID_REFERENCE', 'winner cannot be liable')
                names.append(name)
            check(len(names) == len(set(names)), 'INVALID_REFERENCE', 'duplicate yakuman')
            check(self.value['yakuman_stack'] or len(names) == 1, 'UNSUPPORTED_PAYMENT_FACTS',
                  'non-stacking result requires a selected single yakuman component')
        if win.honba_payer is not None:
            seat(win.honba_payer, 'honba payer')
            check(win.honba_payer in {item['pao_player'] for item in win.yakuman},
                  'INVALID_REFERENCE', 'honba payer must be a verified responsible seat')

    def _ordinary(self, win, dealer):
        from riichienv import calculate_score
        han = min(win.han, 13 if self.value['kazoe'] == 'KAZOE_YAKUMAN' else 12)
        if self.value['kiriage'] and (han, win.fu) in ((3, 60), (4, 30)):
            han = 5
        # Native >=26 means multiple actual yakuman; ordinary kazoe is capped above.
        return calculate_score(han, win.fu, win.actor == dealer, win.target is None, 0)

    def ordinary_value(self,win,dealer):
        self._validate_win(win)
        seat(dealer,'dealer')
        check(not win.yakuman,'TYPE_MISMATCH','ordinary value required')
        han,fu=win.han,win.fu
        if han>=13:
            limit='LIMIT_KAZOE_YAKUMAN' if self.value['kazoe']=='KAZOE_YAKUMAN' else 'LIMIT_SANBAIMAN'
        elif han>=11:
            limit='LIMIT_SANBAIMAN'
        elif han>=8:
            limit='LIMIT_BAIMAN'
        elif han>=6:
            limit='LIMIT_HANEMAN'
        elif han>=5 or fu*2**(han+2)>=2000 or self.value['kiriage'] and (han,fu) in ((3,60),(4,30)):
            limit='LIMIT_MANGAN'
        else:
            limit='LIMIT_NONE'
        return self._ordinary(win,dealer).total,limit

    @staticmethod
    def _base(transfers, actor, target, dealer, score, units=1, kind='BASE'):
        if target is not None:
            transfers.add(target, actor, score.pay_ron * units, kind)
        else:
            for payer in range(4):
                if payer != actor:
                    amount = score.pay_tsumo_oya if payer == dealer else score.pay_tsumo_ko
                    transfers.add(payer, actor, amount * units, kind)

    def _win_base(self, transfers, win, dealer):
        if not win.yakuman:
            self._base(transfers, win.actor, win.target, dealer, self._ordinary(win, dealer))
            return
        from riichienv import calculate_score
        units = defaultdict(int)
        for component in win.yakuman:
            units[component['pao_player']] += component['yakuman_units']
        liable = [p for p in units if p is not None]
        if liable and self.policy['pao_scope'] == 'all_yakuman':
            check(len(liable) == 1, 'UNSUPPORTED_PAYMENT_FACTS', 'all-yakuman liability has multiple responsible seats')
            units = {liable[0]: sum(units.values())}
        score = calculate_score(13, 0, win.actor == dealer, win.target is None, 0)
        for payer, count in units.items():
            if payer is None:
                self._base(transfers, win.actor, win.target, dealer, score, count)
            elif win.target is None or win.target == payer:
                transfers.add(payer, win.actor, score.total * count, 'PAO')
            else:
                half = score.total * count // 2
                transfers.add(payer, win.actor, half, 'PAO')
                transfers.add(win.target, win.actor, half, 'PAO')

    def win(self, scores, *, dealer, honba, kyotaku, claims):
        """Resolve supplied legal claims; does not fabricate unrecorded responses."""
        balances(scores, kyotaku)
        seat(dealer, 'dealer')
        natural(honba, 'honba')
        check(type(claims) in (list, tuple) and 1 <= len(claims) <= 3, 'TYPE_MISMATCH', 'claims')
        for win in claims:
            self._validate_win(win)
        check(len({w.actor for w in claims}) == len(claims), 'INVALID_REFERENCE', 'duplicate winner')
        target = claims[0].target
        check(all(w.target == target for w in claims) and (target is not None or len(claims) == 1),
              'INVALID_REFERENCE', 'shared target/method')
        selected = sorted(claims, key=lambda w: (w.actor-target) % 4) if target is not None else list(claims)
        transfers = Transfers()
        if self.value['multiple_ron'] == 'MULTIRON_TRIPLE_ABORT' and len(selected) == 3:
            return dict(transfers.settlement(scores, kyotaku), winners=[], resolution='ABORT_SANCHA_HOU')
        if self.value['multiple_ron'] == 'MULTIRON_HEAD_BUMP':
            selected = selected[:1]
        for index, win in enumerate(selected):
            self._win_base(transfers, win, dealer)
            paid_honba = honba if target is None or self.policy['multi_ron_honba'] == 'each' or index == 0 else 0
            if paid_honba and any(c['pao_player'] is not None for c in win.yakuman):
                check(win.honba_payer is not None, 'UNSUPPORTED_PAYMENT_FACTS',
                      'pao honba allocation requires an explicit verified payer')
                amount = (3 * self.policy['honba_tsumo_each'] if target is None else self.policy['honba_ron_each'])
                transfers.add(win.honba_payer, win.actor, paid_honba * amount, 'HONBA')
            elif target is None:
                for payer in range(4):
                    if payer != win.actor:
                        transfers.add(payer, win.actor, paid_honba * self.policy['honba_tsumo_each'], 'HONBA')
            else:
                transfers.add(target, win.actor, paid_honba * self.policy['honba_ron_each'], 'HONBA')
        transfers.add(None, selected[0].actor, 1000 * kyotaku, 'KYOTAKU')
        return dict(transfers.settlement(scores, kyotaku), winners=[w.actor for w in selected], resolution='WIN')

    def _noten(self, transfers, tenpai):
        check(type(tenpai) is list and len(tenpai) == 4 and all(type(t) is bool for t in tenpai),
              'TYPE_MISMATCH', 'known tenpai vector')
        count = sum(tenpai)
        if count in (0, 4):
            return
        divisor = count * (4-count)
        check(self.policy['noten_total'] % divisor == 0, 'UNSUPPORTED_PROFILE', 'fractional noten transfer')
        for payer in range(4):
            for payee in range(4):
                if not tenpai[payer] and tenpai[payee]:
                    transfers.add(payer, payee, self.policy['noten_total'] // divisor, 'NOTEN')

    def draw(self, scores, *, dealer, honba, kyotaku, tenpai, nagashi, nagashi_extras=None):
        balances(scores, kyotaku)
        seat(dealer, 'dealer')
        natural(honba, 'honba')
        check(type(nagashi) in (tuple, list) and len(nagashi) <= 4, 'TYPE_MISMATCH', 'nagashi winners')
        for actor in nagashi:
            seat(actor, 'nagashi winner')
        check(len(set(nagashi)) == len(nagashi), 'INVALID_REFERENCE', 'duplicate nagashi winner')
        check(nagashi_extras is None or type(nagashi_extras) is str and nagashi_extras == NAGASHI_NO_HONBA_CARRY_POOL,
              'UNSUPPORTED_PAYMENT_FACTS', 'unverified nagashi honba/pool allocation')
        check(nagashi_extras is None or bool(nagashi), 'INVALID_REFERENCE', 'nagashi extras without nagashi winners')
        transfers = Transfers()
        if nagashi:
            check(self.policy['nagashi'] == 'independent_tsumo', 'UNSUPPORTED_PROFILE', 'nagashi disabled')
            check(honba == kyotaku == 0 or nagashi_extras == NAGASHI_NO_HONBA_CARRY_POOL,
                  'UNSUPPORTED_PAYMENT_FACTS', 'nagashi honba/pool distribution needs explicit source facts')
            # This verified source variant pays only base; Transfers leaves the
            # pool intact. Other variants remain unknown, with no default.
            from riichienv import calculate_score
            for actor in nagashi:
                score = calculate_score(5, 30, actor == dealer, True, 0)
                self._base(transfers, actor, None, dealer, score, kind='NAGASHI')
        if not nagashi or self.policy['nagashi_noten']:
            self._noten(transfers, tenpai)
        return transfers.settlement(scores, kyotaku)

    @staticmethod
    def ranks(scores, starting_dealer):
        balances(scores, 0)
        seat(starting_dealer, 'starting dealer')
        order = sorted(range(4), key=lambda p: (-scores[p], (p-starting_dealer) % 4))
        return [order.index(p)+1 for p in range(4)]

    def final(self, scores, kyotaku, *, starting_dealer, recorded_scores=None):
        """Settle an already recorded end_game, without deciding why the game ended."""
        balances(scores, kyotaku)
        ranks = self.ranks(scores, starting_dealer)
        transfers = Transfers()
        if self.policy['final_kyotaku'] == 'first_rank':
            transfers.add(None, ranks.index(1), 1000 * kyotaku, 'KYOTAKU')
        result = transfers.settlement(scores, kyotaku)
        if recorded_scores is not None:
            balances(recorded_scores, result['next_kyotaku'])
            check(recorded_scores == result['scores_after'], 'SETTLEMENT_MISMATCH', 'recorded terminal balance')
        return result
