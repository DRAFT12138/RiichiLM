"""Canonical offline boundaries around the local RiichiEnv state owner.

Native apply_event is permissive. This adapter checks recorded progression and
defers native kan application until commit; it does not enumerate action sets or
invent missing responses. Production use requires separate profile evidence.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

KOKUSHI_YAKU_IDS = frozenset((42,49))
NATIVE_RULE_FIELDS = frozenset((
    'is_kokushi_musou_13machi_double', 'is_suuankou_tanki_double',
    'is_junsei_chuurenpoutou_double', 'is_daisuushii_double',
    'allows_ron_on_ankan_for_kokushi_musou', 'sanchaho_is_draw',
    'kuikae_forbidden', 'yakuman_pao_is_liability_only',
    'dealer_first_discard_is_tedashi',
))


class ReplayError(ValueError):
    def __init__(self, code, reason):
        self.code, self.reason = code, reason
        super().__init__(f'{code}: {reason}')


def check(condition, code, reason):
    if not condition:
        raise ReplayError(code, reason)


def base(tile):
    return tile.removesuffix('r')


@dataclass(frozen=True)
class UnknownCurrentState:
    """An explicit state-read result, never a substitute for an invalid event.

    A backend may confirm public events without delivering a current private
    state at that boundary. Subsequent reads must independently confirm it.
    The full-tile MJAI adapter below normally returns known states for all seats.
    """
    reason: str

    def __post_init__(self):
        check(type(self.reason) is str and bool(self.reason.strip()),
              'TYPE_MISMATCH','unknown current state requires a reason')


def recorded_settlement(options, events, scores, kyotaku):
    """Validate recorded money facts without asserting a missing win or yaku."""
    check(bool(events),'INVALID_PHASE','empty settlement')
    kind=events[0]['type']
    check(kind in ('hora','ryukyoku') and all(e['type']==kind for e in events),
          'INVALID_PHASE','mixed settlement')
    if kind=='hora':
        check(1<=len(events)<=3 and len({e['actor'] for e in events})==len(events),
              'INVALID_REFERENCE','duplicate/too many winners')
        tsumo=events[0]['actor']==events[0]['target']
        check((not tsumo or len(events)==1) and
              all((e['actor']==e['target'])==tsumo and e['target']==events[0]['target'] for e in events),
              'INVALID_REFERENCE','mixed result targets/methods')
    else:
        check(len(events)==1,'INVALID_PHASE','duplicate draw settlement')
    deltas=[e['deltas'] for e in events]
    if options.settlement_deltas=='repeated_group':
        check(all(d==deltas[0] for d in deltas),'SETTLEMENT_MISMATCH','group deltas disagree')
        delta=list(deltas[0])
    else:
        delta=[sum(d[p] for d in deltas) for p in range(4)]
    next_kyotaku=0 if kind=='hora' else kyotaku
    after=[a+d for a,d in zip(scores,delta)]
    check(sum(scores)+1000*kyotaku==sum(after)+1000*next_kyotaku,
          'SETTLEMENT_MISMATCH','score plus kyotaku conservation')
    running=list(scores)
    for index,e in enumerate(events):
        if options.result_schema=='gimite-mjai/result-1' and kind=='hora':
            check(sum(e['deltas'])==(1000*kyotaku if index==0 else 0),
                  'SETTLEMENT_MISMATCH','producer pool must be collected by the first winner')
        if options.settlement_scores=='per_event':
            running=[s+d for s,d in zip(running,e['deltas'])]
        if 'scores' in e:
            expected=running if options.settlement_scores=='per_event' else after
            check(e['scores']==expected,'SETTLEMENT_MISMATCH','recorded settlement balance')
    return dict(scores_before=list(scores),scores_after=after,score_delta=delta,
                kyotaku=kyotaku,next_kyotaku=next_kyotaku,provenance='recorded')


@dataclass(frozen=True)
class OfflineOptions:
    mode: str
    ankan_public: str
    allow_ankan_chankan: bool
    allow_kakan_chankan: bool
    dora_timing: dict
    settlement_deltas: str
    live_cost_at: str = 'rinshan_draw'
    hand_rules: dict | None = None
    payment_rules: dict | None = None
    reveal_before_ron: bool = False
    result_schema: str | None = None
    settlement_scores: str = 'group_final'
    furiten_rules: dict | None = None
    native_rules: dict | None = None
    aka_counts: list | None = None
    win_rules: dict | None = None
    riichi_policy: dict | None = None
    abort_rules: dict | None = None
    round_rules: dict | None = None
    game_metadata_schema: str | None = None

    def __post_init__(self):
        check(self.game_metadata_schema in (None,'convlog-mjai/1','convlog-mjai/assumed-metadata-1','majsoul-mixed-mjai/1'),
              'UNSUPPORTED_PROFILE','game metadata schema')
        check(self.mode in ('MODE_HANCHAN','MODE_TONPUU'), 'UNSUPPORTED_PROFILE', 'mode')
        check(self.ankan_public in ('all_tiles','base_only'), 'UNSUPPORTED_PROFILE', 'ankan_public')
        check(self.settlement_deltas in ('per_event','repeated_group'), 'UNSUPPORTED_PROFILE', 'source deltas convention')
        check(self.settlement_scores in ('group_final','per_event'),'UNSUPPORTED_PROFILE','source scores convention')
        check(self.settlement_scores!='per_event' or self.settlement_deltas=='per_event',
              'UNSUPPORTED_PROFILE','sequential scores require per-event deltas')
        check(self.result_schema in (None,'gimite-mjai/result-1'),'UNSUPPORTED_PROFILE','source result schema')
        if self.result_schema=='gimite-mjai/result-1':
            check(self.settlement_deltas==self.settlement_scores=='per_event',
                  'UNSUPPORTED_PROFILE','gimite result schema requires sequential per-event money facts')
        check(set(self.dora_timing) == {'ankan','kakan','daiminkan'} and
              all(v in ('commit','after_discard_response','before_discard_or_next_rinshan') for v in self.dora_timing.values()),
              'UNSUPPORTED_PROFILE', 'dora_timing')
        check(type(self.reveal_before_ron) is bool,'TYPE_MISMATCH','reveal_before_ron')
        check(self.live_cost_at == 'rinshan_draw', 'UNSUPPORTED_PROFILE','live cost at commit is not implemented')
        if self.aka_counts is not None:
            from .tile_stock import TileStock
            TileStock(self.aka_counts)
        if self.win_rules is not None:
            from .win_rules import WinRules
            WinRules.from_dict(self.win_rules)
        if self.riichi_policy is not None:
            from .riichi_policy import RiichiPolicy
            policy=RiichiPolicy.from_dict(self.riichi_policy)
            check(policy.allow_furiten or self.hand_rules is not None and self.furiten_rules is not None,
                  'UNSUPPORTED_PROFILE','forbidding furiten riichi requires explicit hand and furiten components')
        if self.abort_rules is not None:
            from .abort_rules import AbortRules
            AbortRules.from_dict(self.abort_rules)
        if self.round_rules is not None:
            from .round_rules import RoundRules
            RoundRules.from_dict(self.round_rules)
        if self.hand_rules is not None:
            from .hand_rules import HandRules
            HandRules.from_dict(self.hand_rules)
        if self.payment_rules is not None:
            from .payment_rules import PaymentRules
            PaymentRules(self.payment_rules)
        if self.furiten_rules is not None:
            from .furiten import FuritenRules
            FuritenRules.from_dict(self.furiten_rules)
        if self.native_rules is not None:
            check(type(self.native_rules) is dict and self.native_rules.keys()==NATIVE_RULE_FIELDS,
                  'UNSUPPORTED_PROFILE','native_rules requires every explicit GameRule flag')
            check(all(type(v) is bool for v in self.native_rules.values()),
                  'TYPE_MISMATCH','native_rules flags must be booleans')
            check(self.native_rules['allows_ron_on_ankan_for_kokushi_musou']==self.allow_ankan_chankan,
                  'UNSUPPORTED_PROFILE','native and offline ankan chankan disagree')
            if self.payment_rules is not None:
                policy=self.payment_rules
                check(policy['multiple_ron'] in ('MULTIRON_ALL','MULTIRON_TRIPLE_ABORT'),
                      'UNSUPPORTED_PROFILE','native rule cannot implement payment head-bump')
                check(self.native_rules['sanchaho_is_draw']==(policy['multiple_ron']=='MULTIRON_TRIPLE_ABORT'),
                      'UNSUPPORTED_PROFILE','native and payment multiple ron disagree')
                check(self.native_rules['yakuman_pao_is_liability_only']==(policy['policy']['pao_scope']=='liable_units'),
                      'UNSUPPORTED_PROFILE','native and payment pao scope disagree')
                units={m['name']:m['count'] for m in policy['yakuman_multipliers']}
                for name,flag in (
                    ('Y_KOKUSHI_13','is_kokushi_musou_13machi_double'),
                    ('Y_SUUANKOU_TANKI','is_suuankou_tanki_double'),
                    ('Y_JUNSEI_CHUUREN','is_junsei_chuurenpoutou_double'),
                    ('Y_DAISUUSHII','is_daisuushii_double'),
                ):
                    check(units[name] in (1,2) and self.native_rules[flag]==(units[name]==2),
                          'UNSUPPORTED_PROFILE','native and payment multiplier disagree: '+name)


class OfflineSession:
    def __init__(self, options, *, rule=None):
        from riichienv import GameRule, RiichiEnv
        check(options.game_metadata_schema != 'majsoul-mixed-mjai/1',
              'UNSUPPORTED_PROFILE', 'mixed MJAI adapter requires Rust backend')
        self.options = options
        if options.native_rules is not None:
            if rule is None:
                rule=GameRule(**options.native_rules)
            else:
                check(all(getattr(rule,name,None) is value for name,value in options.native_rules.items()),
                      'UNSUPPORTED_PROFILE','supplied native rule differs from frozen options')
        self.rule = rule
        from .tile_stock import TileStock
        self.tile_stock=TileStock(options.aka_counts)
        from .win_rules import WinRules
        from .riichi_policy import RiichiPolicy
        from .abort_rules import AbortRules
        self.win_rules=None if options.win_rules is None else WinRules.from_dict(options.win_rules)
        self.riichi_policy=None if options.riichi_policy is None else RiichiPolicy.from_dict(options.riichi_policy)
        self.abort_rules=None if options.abort_rules is None else AbortRules.from_dict(options.abort_rules)
        from .round_rules import RoundRules
        self.round_rules=None if options.round_rules is None else RoundRules.from_dict(options.round_rules)
        from .hand_rules import HandRules
        self.hand_rules=None if options.hand_rules is None else HandRules.from_dict(options.hand_rules)
        from .payment_rules import PaymentRules
        self.payment_rules=None if options.payment_rules is None else PaymentRules(options.payment_rules)
        from .furiten import FuritenKnowledge,FuritenRules
        self.furiten=None if options.furiten_rules is None else FuritenKnowledge(FuritenRules.from_dict(options.furiten_rules))
        self.starting_dealer=None
        self.finalized=False
        self.end_reason=None
        self.riichi_waits=[None]*4
        self.env = RiichiEnv(game_mode='4p-red-half' if options.mode=='MODE_HANCHAN' else '4p-red-east',
                             skip_mjai_logging=True, rule=rule)
        self.pending = None
        self.meld_refs = [[] for _ in range(4)]
        self.claimed_discards = Counter()
        self.next_draw = None
        self.action_actor = None
        self.last_draw = None
        self.after_call = False
        self.last_discard = None
        self.declarations = set()
        self.declaration_pending_discard = None
        self.accepted = set()
        self.rinshan_actor = None
        self.rinshan_meld = None
        self.dora_pending = []
        self.ended = True
        self.header = None
        self.round_result = None

    def _hands(self):
        from riichienv.convert import tid_to_mjai
        return [[tid_to_mjai(t) for t in h] for h in self.env.hands]

    def _melds(self):
        from riichienv.convert import tid_to_mjai
        result = []
        for actor, native in enumerate(self.env.melds):
            check(len(native)==len(self.meld_refs[actor]), 'INVALID_REFERENCE','native meld/reference count')
            result.append([dict(**ref,tiles=[tid_to_mjai(t) for t in m.tiles])
                           for ref,m in zip(self.meld_refs[actor],native)])
        return result

    def snapshot(self):
        from riichienv.convert import tid_to_mjai
        hands = self._hands()
        if self.pending is not None:
            for tile in self.pending['reserved']:
                hands[self.pending['actor']].remove(tile)
        return deepcopy(dict(hands=hands,melds=self._melds(),pending=self.pending,
                             scores=self.env.scores(),kyotaku=self.env.riichi_sticks,
                             dora_indicators=[tid_to_mjai(t) for t in self.env.dora_indicators],
                             dora_pending=self.dora_pending,
                             live_remaining=self.env.live_remaining,
                             rinshan_remaining=4-self.env.rinshan_draw_count))

    def read_current_states(self):
        """One boundary read, four known hand/meld dicts or explicit unknowns.

        Unknown means only the current state is unavailable. Exceptions, bad
        tiles and invalid public progression remain failures, not unknown reads.
        """
        snapshot=self.snapshot()
        return [dict(hands=snapshot['hands'][p],melds=snapshot['melds'][p]) for p in range(4)]

    def result_material(self,actor,winning_tile,is_tsumo,*,chankan=False):
        """Native shape/value and recorded context, excluding unknown ura.

        This does not certify missing private response/furiten knowledge or
        admit a historical rules profile.
        """
        from riichienv.convert import tid_to_mjai
        hands=self._hands()[actor]
        melds=self._melds()[actor]
        check(len(hands)+3*len(melds)==(14 if is_tsumo else 13),
              'INVALID_INVENTORY','winner standing hand size')
        candidates=self.evaluate_win(actor,winning_tile,is_tsumo,chankan=chankan,all_candidates=True)
        check(bool(candidates),
              'INVALID_WIN_SHAPE','recorded winner has no complete hand shape')
        evaluated=candidates[0]
        pre=list(hands)
        if is_tsumo:
            check(winning_tile in pre,'INVALID_INVENTORY','winning draw not held')
            pre.remove(winning_tile)
        full=[*pre,winning_tile,*[t for m in melds for t in m['tiles']]]
        return deepcopy(dict(hands=hands,pre_win_hand=pre,full_hand=full,melds=melds,
            menzen=all(m['name']=='MELD_ANKAN' for m in melds),
            native_has_yaku=evaluated.is_win,native_yaku_ids=list(evaluated.yaku),
            native_han=evaluated.han,native_fu=evaluated.fu,
            native_candidates=[dict(native_has_yaku=c.is_win,native_yaku_ids=list(c.yaku),
                native_han=c.han,native_fu=c.fu,native_yakuman_ids=[i for i in c.yaku if i>=35]) for c in candidates],
            native_yakuman_ids=[i for i in evaluated.yaku if i>=35],
            responsibility=self.responsibility_facts(actor),
            dora_count=self.count_dora(full,[tid_to_mjai(t) for t in self.env.dora_indicators]),
            aka_count=sum(t.endswith('r') for t in full)))

    def evaluate_win(self,actor,winning_tile,is_tsumo,*,chankan=False,all_candidates=False):
        from riichienv import Conditions,HandEvaluator
        from riichienv.convert import tid_to_mjai
        win_id=next(t for t in range(136) if tid_to_mjai(t)==winning_tile)
        reached=actor in self.accepted
        evaluator=HandEvaluator(self.env.hands[actor],self.env.melds[actor])
        evaluate=evaluator.calc_candidates if all_candidates else evaluator.calc
        return evaluate(win_id,
            dora_indicators=self.env.dora_indicators,conditions=Conditions(
            tsumo=is_tsumo,player_wind=(actor-self.header['oya'])%4,
            round_wind='ESWN'.index(self.header['bakaze']),
            riichi=reached,double_riichi=reached and self.env.double_riichi_declared[actor],
            ippatsu=reached and self.env.ippatsu_cycle[actor],
            rinshan=is_tsumo and self.env.is_rinshan_flag,chankan=chankan,
            haitei=is_tsumo and self.env.live_remaining==0 and not self.env.is_rinshan_flag,
            houtei=not is_tsumo and self.env.live_remaining==0 and not chankan,
            tsumo_first_turn=is_tsumo and self.env.is_first_turn and not self.env.discards[actor],
            **({} if self.win_rules is None else self.win_rules.conditions())))

    def check_recorded_win(self,actor,winning_tile,is_tsumo,*,chankan=False):
        """Disprove impossible sparse results without filling their fields."""
        check(len(self.env.hands[actor])+3*len(self.env.melds[actor])==(14 if is_tsumo else 13),
              'INVALID_INVENTORY','winner standing hand size')
        value=self.evaluate_win(actor,winning_tile,is_tsumo,chankan=chankan)
        check(value.has_win_shape,'INVALID_WIN_SHAPE','recorded winner has no complete hand shape')
        # Keep the same pending-riichi uncertainty used by ron_available.
        # Missing public acceptance cannot be silently treated as confirmed.
        check(value.is_win or actor in self.declarations and actor not in self.accepted,
              'INVALID_WIN_YAKU','recorded winner has no native nonbonus yaku')

    def furiten_state(self,actor):
        from .hand_rules import HandRules
        discard=None
        if len(self.env.hands[actor])+3*len(self.env.melds[actor])==13:
            waits=HandRules.waits(self.env.hands[actor],self.env.melds[actor])
            discard=bool(waits.intersection(t//4 for t in self.env.discards[actor]))
        return dict(discard=discard,temp=None if self.furiten is None else self.furiten.temp[actor],
                    riichi=None if self.furiten is None else self.furiten.riichi[actor])

    def check_ron_furiten(self,actor):
        states=self.furiten_state(actor)
        check(not any(v is True for v in states.values()),'INVALID_FURITEN','recorded ron has proved furiten')
        return states

    def check_kan_ron(self,actor,winning_tile):
        """A recorded winner must satisfy the pending kan's exception rule."""
        check(self.pending is not None,'INVALID_PHASE','kan ron without a pending attempt')
        kind=self.pending['kind']
        check((kind=='ankan' and self.options.allow_ankan_chankan) or
              (kind=='kakan' and self.options.allow_kakan_chankan),'UNSUPPORTED_PROFILE','kan robbery')
        if kind=='ankan':
            value=self.evaluate_win(actor,winning_tile,False,chankan=True)
            check(value.has_win_shape and value.is_win and KOKUSHI_YAKU_IDS.intersection(value.yaku),
                  'INVALID_ANKAN_CHANKAN','concealed kan robbery requires kokushi')

    def ron_available(self,actor,winning_tile,*,kan_kind=None):
        if kan_kind=='ankan' and not self.options.allow_ankan_chankan:return False
        if kan_kind=='kakan' and not self.options.allow_kakan_chankan:return False
        states=self.furiten_state(actor)
        if any(v is True for v in states.values()):return False
        value=self.evaluate_win(actor,winning_tile,False,chankan=kan_kind is not None)
        if not value.has_win_shape:return False
        if kan_kind=='ankan':
            # Only the explicit kokushi exception is available in strict 4P.
            if not KOKUSHI_YAKU_IDS.intersection(value.yaku):return False
        if not value.is_win:
            return None if actor in self.declarations and actor not in self.accepted else False
        return None if any(v is None for v in states.values()) else True

    def observe_responses(self,event,responders,*,winners=(),abort=False):
        if self.furiten is None:return []
        pending=self.pending
        winning_tile=pending['event'].get('pai',pending['reserved'][0]) if pending else self.last_discard['pai']
        kan_kind=None if pending is None else pending['kind']
        complete=self.furiten.rules.response_collection=='all_before_resolution'
        multiron=None if self.payment_rules is None else self.payment_rules.value['multiple_ron']
        results=[]
        for actor in responders:
            available=self.ron_available(actor,winning_tile,kan_kind=kan_kind)
            if actor in winners:choice,knowledge='RON','public_triple_ron' if abort else 'recorded_ron'
            elif event['type'] in ('chi','pon','daiminkan') and event['actor']==actor:
                choice,knowledge=event['type'].upper(),'recorded_call'
            elif complete and not abort and (event['type'] in ('tsumo','chi','pon','daiminkan') or
                    event['type']=='hora' and multiron in ('MULTIRON_ALL','MULTIRON_TRIPLE_ABORT')):
                choice,knowledge='NO_RON','complete_collection'
            else:choice,knowledge=None,'unobserved'
            reached=True if actor in self.accepted else None if actor in self.declarations else False
            missed=self.furiten.observe(actor,riichi=reached,ron_available=available,choice=choice)
            results.append(dict(actor=actor,ron_available=available,missed=missed,knowledge=knowledge,
                                **self.furiten_state(actor)))
        return results

    def responsibility_facts(self,actor):
        """Attach existing public references to native responsibility facts."""
        result=[]
        for yaku,payer in self.env.pao[actor].items():
            check(yaku in (37,50),'UNSUPPORTED_RESULT_DETAILS','unknown native liability identity')
            family,required=('PFC',3) if yaku==37 else ('ESWN',4)
            melds=[m for m in self.meld_refs[actor] if m['base_tile'] in family]
            check(len(melds)==required,'INVALID_REFERENCE','native liability meld count')
            trigger=max(melds,key=lambda m:m['created_event'])
            check(trigger['name'] in ('MELD_PON','MELD_KAKAN','MELD_DAIMINKAN') and trigger['from']==payer,
                  'INVALID_REFERENCE','native liability/public call source')
            result.append(dict(native_yaku_id=yaku,pao_player=payer,trigger_event=trigger['created_event']))
        return result

    def draw_material(self):
        """Read end-of-round facts without invoking the online draw transition."""
        from .hand_rules import HandRules
        hands=self.snapshot()['hands']
        sizes=[len(hands[p])+3*len(self.env.melds[p]) for p in range(4)]
        check(all(n in (13,14) for n in sizes),'INVALID_INVENTORY','round-result standing hand sizes')
        tenpai=[bool(HandRules.waits(self.env.hands[p],self.env.melds[p]))
                if sizes[p]==13 and not (self.pending and self.pending['actor']==p) else None for p in range(4)]
        nagashi=[p for p in range(4) if self.env.nagashi_eligible[p] and self.env.discards[p]]
        if self.payment_rules is not None and self.payment_rules.policy['nagashi']=='disabled':nagashi=[]
        return dict(hands=hands,sizes=sizes,tenpai=tenpai,nagashi=nagashi,
                    abort_causes=self.env.abortive_draw_causes())

    def boundary_abort_causes(self,native=None):
        if self.abort_rules is None or self.action_actor is not None or self.pending is not None or self.rinshan_actor is not None:
            return []
        causes=self.abort_rules.enabled(self.env.abortive_draw_causes() if native is None else native)
        return [kind for kind in causes if kind!='four_riichi' or self.accepted==set(range(4))]

    def validate_draw(self,kind,material,*,actor=None,claimants=(),trigger_tile=None):
        if self.abort_rules is not None and kind!='unknown':
            available=self.boundary_abort_causes(material['abort_causes'])
            if kind=='kyuushu':
                check(self.abort_rules.kyuushu,'RESULT_DETAILS_MISMATCH','nine-terminals abort disabled')
            else:
                choice='ron' if kind=='triple_ron' else 'exhaustive' if kind in ('exhaustive','nagashi') else kind
                if choice in ('ron','exhaustive'):available.append(choice)
                elif self.env.live_remaining==0:available.append('exhaustive')
                self.abort_rules.check_choice(choice,available)
        if kind=='kyuushu':
            check(actor==self.action_actor and self.pending is None and material['sizes'][actor]==14 and
                  self.env.kyuushu_available(actor),'RESULT_DETAILS_MISMATCH','invalid nine-terminals request')
        elif kind in ('four_winds','four_riichi','four_kan'):
            native={'four_winds':'sufuurenta','four_riichi':'suucha_riichi','four_kan':'suukansansen'}[kind]
            check(self.action_actor is None and self.pending is None and native in material['abort_causes'] and
                  all(n==13 for n in material['sizes']),
                  'RESULT_DETAILS_MISMATCH','abort predicate/boundary is false')
            if kind=='four_riichi':
                check(self.accepted==set(range(4)) and all(t is True for t in material['tenpai']),
                      'RESULT_DETAILS_MISMATCH','four accepted tenpai declarations required')
        elif kind=='triple_ron':
            check(len(claimants)==3 and len(set(claimants))==3 and trigger_tile is not None,
                  'RESULT_DETAILS_MISMATCH','three recorded ron claims require a response trigger')
            if self.payment_rules is not None:
                check(self.payment_rules.value['multiple_ron']=='MULTIRON_TRIPLE_ABORT',
                      'UNSUPPORTED_PROFILE','triple-ron abort disagrees with explicit policy')
            for player in claimants:
                self.result_material(player,trigger_tile,False)
                from .hand_rules import HandRules
                waits=HandRules.waits(self.env.hands[player],self.env.melds[player])
                check(not waits.intersection(t//4 for t in self.env.discards[player]),
                      'RESULT_DETAILS_MISMATCH','recorded triple-ron claimant has discard furiten')
                self.check_ron_furiten(player)
        elif kind in ('exhaustive','nagashi'):
            check(self.action_actor is None and self.pending is None and self.env.live_remaining==0 and
                  all(n==13 for n in material['sizes']),
                  'RESULT_DETAILS_MISMATCH','exhaustive result before last discard response')
            check(bool(material['nagashi'])==(kind=='nagashi'),
                  'RESULT_DETAILS_MISMATCH','recorded exhaustive/nagashi reason disagrees with native rivers')

    def cancel_kan_for_abort(self):
        check(self.pending is not None and self.pending['kind']=='kakan',
              'INVALID_PHASE','only a pending added kan admits source triple-ron abort')
        # Native still owns the reserved tile. No kan, robbery, indicator or
        # supplement is fabricated for the aborted response.
        self.pending=None
        self.action_actor=None
        self.check_inventory()

    @staticmethod
    def count_dora(material,indicators):
        def next_base(indicator):
            t=base(indicator)
            if len(t)==2:
                return str(int(t[0])%9+1)+t[1]
            cycle='ESWN' if t in 'ESWN' else 'PFC'
            return cycle[(cycle.index(t)+1)%len(cycle)]
        counts=Counter(base(t) for t in material)
        return sum(counts[next_base(t)] for t in indicators)

    def balance_checkpoint(self):
        header=None if self.header is None else {k:self.header[k] for k in ('bakaze','kyoku','oya','honba')}
        return deepcopy(dict(scores=self.env.scores(),kyotaku=self.env.riichi_sticks,
                             header=header,settled=self.ended))

    def check_inventory(self,extra_tiles=()):
        from riichienv.convert import tid_to_mjai
        # Native still owns pending tiles until commit; count them only once.
        entities = [t for h in self._hands() for t in h]
        entities += [tid_to_mjai(t) for ms in self.env.melds for m in ms for t in m.tiles]
        discards = Counter(tid_to_mjai(t) for river in self.env.discards for t in river)
        discards.subtract(self.claimed_discards)
        check(all(c>=0 for c in discards.values()),'INVALID_INVENTORY','claimed discard missing')
        entities += [t for t,c in discards.items() for _ in range(c)]
        entities += [tid_to_mjai(t) for t in self.env.dora_indicators]
        entities += list(extra_tiles)
        self.tile_stock.check(entities)

    def begin_round(self, event):
        check(self.ended and not self.finalized, 'INVALID_PHASE', 'new round before prior end or after game end')
        if self.options.round_rules is not None:
            if self.header is None:
                check(event['bakaze']=='E' and event['kyoku']==1 and event['honba']==event['kyotaku']==0,
                      'INVALID_PHASE','profile game must start at the initial round')
                check(event['scores']==self.options.round_rules['initial_scores'],
                      'SETTLEMENT_MISMATCH','profile initial scores')
            else:
                self.round_decision().check_boundary(event)
        if self.header is not None:
            check(event['scores']==self.env.scores() and event['kyotaku']==self.env.riichi_sticks,
                  'SETTLEMENT_MISMATCH','next round balance/sticks')
        self.tile_stock.check([t for hand in event['tehais'] for t in hand]+[event['dora_marker']])
        self.env.apply_event(event)
        if self.header is None and event['bakaze']=='E' and event['kyoku']==1:
            self.starting_dealer=event['oya']
        self.header=deepcopy(event)
        self.round_result=None
        if self.furiten is not None:self.furiten.reset()
        self.pending=None
        self.meld_refs=[[] for _ in range(4)]
        self.claimed_discards=Counter()
        self.next_draw=event['oya']
        self.action_actor=None
        self.last_draw=None
        self.last_discard=None
        self.after_call=False
        self.declarations=set()
        self.declaration_pending_discard=None
        self.accepted=set()
        self.riichi_waits=[None]*4
        self.rinshan_actor=None
        self.rinshan_meld=None
        self.dora_pending=[]
        self.ended=False
        self.check_inventory()

    def _holding(self, actor, consumed):
        check(not (Counter(consumed)-Counter(self._hands()[actor])), 'INVALID_INVENTORY','consumed not held')

    def _without(self,actor,consumed):
        from riichienv.convert import tid_to_mjai
        remaining=list(self.env.hands[actor])
        taken=[]
        for tile in consumed:
            index=next(i for i,t in enumerate(remaining) if tid_to_mjai(t)==tile)
            taken.append(remaining.pop(index))
        return remaining,taken

    def draw(self, event):
        actor=event['actor']
        check(not self.ended and self.pending is None and self.action_actor is None,
              'INVALID_PHASE','draw while an action/pending kan remains')
        check(actor==self.next_draw,'INVALID_PHASE','draw actor')
        check(not self.boundary_abort_causes(),'INVALID_PHASE','mandatory abort before next draw')
        kind='DRAW_RINSHAN' if self.rinshan_actor is not None else 'DRAW_LIVE'
        check(kind=='DRAW_LIVE' or actor==self.rinshan_actor,'INVALID_PHASE','rinshan actor')
        self.require_revealed_dora()
        check(len(self.env.dora_indicators)>=1,'INVALID_PHASE','no initial dora')
        check(self.env.live_remaining>0,'INVALID_PHASE','live wall exhausted')
        check(kind=='DRAW_LIVE' or self.env.rinshan_draw_count<4,'INVALID_PHASE','rinshan exhausted')
        self.check_inventory([event['pai']])
        self.env.is_rinshan_flag=kind=='DRAW_RINSHAN'
        self.env.apply_event(event)
        if self.furiten is not None:self.furiten.own_draw(actor)
        if kind=='DRAW_RINSHAN':
            self.env.rinshan_draw_count+=1
            for pending in self.dora_pending:
                if pending['meld_id']==self.rinshan_meld:
                    pending['rinshan_drawn']=True
        self.action_actor=actor
        self.last_draw=(actor,event['pai'])
        self.after_call=False
        self.rinshan_actor=None
        self.rinshan_meld=None
        self.check_inventory()
        return kind

    def declare(self,event):
        actor=event['actor']
        check(actor==self.action_actor and not self.after_call and actor not in self.declarations,
              'INVALID_PHASE','riichi declaration')
        check(not self.env.melds[actor] or all(not m.opened for m in self.env.melds[actor]),
              'INVALID_PHASE','open riichi')
        if self.hand_rules is not None:
            self.hand_rules.declaration(self.env.hands[actor],self.env.melds[actor],
                                        self.env.scores()[actor],self.env.live_remaining)
        else:
            check(self.env.scores()[actor]>=1000,'INVALID_PHASE','insufficient riichi score')
        self.env.apply_event(event)
        self.declarations.add(actor)
        self.declaration_pending_discard=actor

    def discard(self,event):
        from riichienv.convert import tid_to_mjai
        actor,tile=event['actor'],event['pai']
        check(actor==self.action_actor and self.pending is None,'INVALID_PHASE','discard actor')
        check(not self.before_discard_dora_pending(),'MISSING_DORA','delayed indicator required before discard')
        self.require_revealed_dora()
        self._holding(actor,[tile])
        check(not event['tsumogiri'] or (self.last_draw==(actor,tile) and not self.after_call),
              'DISCARD_SOURCE_MISMATCH','tsumogiri has no matching draw')
        check(base(tile) not in {base(tid_to_mjai(t)) for t in self.env.forbidden_discards[actor]},
              'INVALID_PHASE','kuikae discard')
        if actor in self.accepted:
            check(event['tsumogiri'],'INVALID_PHASE','riichi tedashi')
        riichi_tile=actor in self.declarations and actor not in self.accepted
        waits=None
        if riichi_tile and self.hand_rules is not None:
            remaining,_=self._without(actor,[tile])
            waits=self.hand_rules.declaration_discard(remaining,self.env.melds[actor])
            if self.riichi_policy is not None and not self.riichi_policy.allow_furiten:
                river={t//4 for t in self.env.discards[actor]}
                discarded=next(t for t in range(136) if tid_to_mjai(t)==tile)
                states=self.furiten_state(actor)
                check(not waits.intersection(river|{discarded//4}) and
                      states['temp'] is False and states['riichi'] is False,
                      'INVALID_RIICHI','furiten riichi disabled or unproved')
        self.env.apply_event(event)
        if waits is not None:
            self.riichi_waits[actor]=waits
        self.env.is_rinshan_flag=False
        if self.furiten is not None:
            # A no-yaku completion cannot be claimed, so missing it is already
            # proved at the discard, without observing a private PASS. It still
            # causes temporary furiten across the whole wait, as in native play.
            for player in range(4):
                if player == actor or player in self.declarations:
                    continue
                if any(v is True for v in self.furiten_state(player).values()):
                    continue
                value=self.evaluate_win(player,tile,False)
                if value.has_win_shape and not value.is_win:
                    self.furiten.observe(player,riichi=False,ron_available=True,choice='NO_RON')
        self.action_actor=None
        self.next_draw=(actor+1)%4
        self.last_draw=None
        self.last_discard=deepcopy(event)
        self.after_call=False
        self.declaration_pending_discard=None
        # A delayed indicator becomes eligible only after this discard response.
        for pending in self.dora_pending:
            if pending['actor']==actor and pending['timing']=='after_discard_response':
                pending['discarded']=True
        self.check_inventory()
        return riichi_tile

    def response_closed(self,kind=None):
        for pending in self.dora_pending:
            if pending.get('discarded') and (kind!='hora' or self.options.reveal_before_ron):
                pending['eligible']=True

    def before_discard_dora_pending(self):
        return any(p['timing']=='before_discard_or_next_rinshan' and p['rinshan_drawn']
                   for p in self.dora_pending)

    def prepare_dora_boundary(self,kind,actor):
        if kind=='dahai':
            check(actor==self.action_actor and self.pending is None,'INVALID_PHASE','discard reveal boundary')
            for pending in self.dora_pending:
                if (pending['actor']==actor and pending['timing']=='before_discard_or_next_rinshan'
                        and pending['rinshan_drawn']):
                    pending['eligible']=True

    def require_revealed_dora(self):
        check(not any(p['eligible'] for p in self.dora_pending),'MISSING_DORA','due indicator not recorded before boundary')

    def validate_accept(self,event,*,response_kind=None):
        self.check_play_phase()
        check(response_kind!='hora','INVALID_PHASE','riichi cannot be accepted on a declaration discard won by ron')
        actor=event['actor']
        check(actor in self.declarations and actor not in self.accepted and
              self.last_discard is not None and self.last_discard['actor']==actor and
              self.action_actor is None and self.pending is None,
              'INVALID_PHASE','riichi acceptance')
        before=self.env.scores()
        expected=[v-(1000 if p==actor else 0) for p,v in enumerate(before)]
        if 'scores' in event:
            check(event['scores']==expected,'SETTLEMENT_MISMATCH','recorded acceptance scores')
        if 'deltas' in event:
            check(event['deltas']==[-1000 if p==actor else 0 for p in range(4)],
                  'SETTLEMENT_MISMATCH','recorded acceptance deltas')
        return expected

    def accept(self,event):
        expected=self.validate_accept(event)
        self.env.apply_event(event)
        self.accepted.add(event['actor'])
        check(self.env.scores()==expected,'SETTLEMENT_MISMATCH','riichi debit')

    def _call_ref(self,event,event_id,meld_id,trigger_id):
        actor,target,pai=event['actor'],event['target'],event['pai']
        check(actor not in self.accepted,'INVALID_PHASE','open call after accepted riichi')
        check(not self.boundary_abort_causes(),'INVALID_PHASE','mandatory abort before call')
        check(self.last_discard is not None and self.last_discard['actor']==target and
              self.last_discard['pai']==pai and actor!=target and self.action_actor is None,
              'INVALID_REFERENCE','call trigger/source')
        self._holding(actor,event['consumed'])
        kind=event['type']
        tiles=[pai,*event['consumed']]
        if kind=='chi':
            check(actor==(target+1)%4,'INVALID_PHASE','chi direction')
            sorted_tiles=sorted(base(t) for t in tiles)
            try:
                values=[int(t[0]) for t in sorted_tiles]
            except ValueError:
                values=[]
            check(values and len({t[1] for t in sorted_tiles})==1 and sorted_tiles[0][1] in 'mps' and
                  values==list(range(values[0],values[0]+3)), 'INVALID_INVENTORY','chi composition')
            base_tile=sorted_tiles[0]
        else:
            check(all(base(t)==base(pai) for t in tiles),'INVALID_INVENTORY','pon/kan composition')
            base_tile=base(pai)
        return dict(name='MELD_'+kind.upper(),actor=actor,meld_id=meld_id,base_tile=base_tile,
                    **{'from':target},called_tile=pai,trigger_id=trigger_id,
                    created_event=event_id,updated_event=None)

    def call(self,event,event_id,meld_id,trigger_id):
        if event['type']=='daiminkan':
            self.check_kan_resources()
        ref=self._call_ref(event,event_id,meld_id,trigger_id)
        actor=event['actor']
        if event['type']=='daiminkan':
            check(self.pending is None,'INVALID_PHASE','nested pending kan')
            self.pending=dict(actor=actor,kind='daiminkan',event=deepcopy(event),
                              reserved=list(event['consumed']),ref=ref)
        else:
            self.apply_native_call(event)
            self.meld_refs[actor].append(ref)
            self.claimed_discards[event['pai']]+=1
            self._fix_native_source(actor,ref)
            self.action_actor=actor
            self.after_call=True
            self.last_draw=None
            self.check_inventory()
        return deepcopy(ref)

    def check_kan_resources(self):
        check(sum(m['name'] in ('MELD_ANKAN','MELD_DAIMINKAN','MELD_KAKAN')
                  for ms in self.meld_refs for m in ms)<4,'INVALID_PHASE','kan limit')
        check(self.env.live_remaining>0 and self.env.rinshan_draw_count<4,'INVALID_PHASE','no supplement resources')

    def apply_native_call(self,event):
        previous=list(self.env.ippatsu_cycle)
        self.env.apply_event(event)
        if self.riichi_policy is not None and not self.riichi_policy.cancel_on(event['type']):
            self.env.ippatsu_cycle=previous

    def kan_attempt(self,event,event_id,meld_id,window_id):
        actor=event['actor']
        check(actor==self.action_actor and self.pending is None and not self.after_call,
              'INVALID_PHASE','kan attempt actor/phase')
        check(self.declaration_pending_discard is None,'INVALID_RIICHI_KAN',
              'kan before declaration discard/acceptance')
        self.check_kan_resources()
        kind=event['type']
        if self.hand_rules is not None and actor in self.declarations:
            check(actor in self.accepted,'INVALID_RIICHI_KAN','kan before declaration discard/acceptance')
            check(kind=='ankan','INVALID_RIICHI_KAN','only ankan is permitted after riichi')
        if kind=='ankan':
            consumed=event['consumed']
            self._holding(actor,consumed)
            check(all(base(t)==base(consumed[0]) for t in consumed),'INVALID_INVENTORY','ankan composition')
            if self.hand_rules is not None and actor in self.accepted:
                _,taken=self._without(actor,consumed)
                drawn=None if self.last_draw is None else self.env.drawn_tile
                self.hand_rules.riichi_ankan(self.env.hands[actor],self.env.melds[actor],taken,
                                            drawn,self.riichi_waits[actor])
            ref=dict(name='MELD_ANKAN',actor=actor,meld_id=meld_id,base_tile=base(consumed[0]),
                     **{'from':None},called_tile=None,trigger_id=None,created_event=None,updated_event=None)
        else:
            consumed=[event['pai']]
            self._holding(actor,consumed)
            melds=self._melds()[actor]
            choices=[m for m in melds if m['name']=='MELD_PON' and m['base_tile']==base(event['pai'])]
            check(len(choices)==1,'INVALID_REFERENCE','kakan original pon')
            ref=deepcopy(choices[0])
            check(Counter(event['consumed'])==Counter(ref['tiles']),'INVALID_INVENTORY','kakan consumed pon')
            ref.pop('tiles')
            meld_id=ref['meld_id']
        self.pending=dict(actor=actor,kind=kind,event=deepcopy(event),reserved=list(consumed),ref=ref,
                          attempt_event=event_id,window_id=window_id)
        return deepcopy(self.pending)

    def _fix_native_source(self,actor,ref):
        melds=self.env.melds
        index=next(i for i,m in enumerate(self.meld_refs[actor]) if m['meld_id']==ref['meld_id'])
        melds[actor][index].from_who=-1 if ref['from'] is None else ref['from']
        self.env.melds=melds

    def commit_kan(self,event_id):
        check(self.pending is not None,'INVALID_PHASE','no pending kan')
        pending=self.pending
        actor,kind=pending['actor'],pending['kind']
        ref=deepcopy(pending['ref'])
        self.apply_native_call(pending['event'])
        if kind=='ankan':
            ref['created_event']=event_id
            self.meld_refs[actor].append(ref)
        elif kind=='kakan':
            ref.update(name='MELD_KAKAN',updated_event=event_id)
            index=next(i for i,m in enumerate(self.meld_refs[actor]) if m['meld_id']==ref['meld_id'])
            self.meld_refs[actor][index]=ref
        else:
            self.meld_refs[actor].append(ref)
            self.claimed_discards[pending['event']['pai']]+=1
        self._fix_native_source(actor,ref)
        # Only older, already drawn delayed items become due for the next
        # supplement. The new kan's delayed item is appended afterwards.
        for old in self.dora_pending:
            if old['actor']==actor and old['timing']=='before_discard_or_next_rinshan' and old['rinshan_drawn']:
                old['eligible']=True
        timing=self.options.dora_timing[kind]
        self.dora_pending.append(dict(actor=actor,kind=kind,eligible=timing=='commit',timing=timing,
                                      meld_id=ref['meld_id'],commit_event=event_id,rinshan_drawn=False))
        self.env.pending_kan_dora_count+=1
        self.pending=None
        self.action_actor=None
        self.next_draw=actor
        self.rinshan_actor=actor
        self.rinshan_meld=ref['meld_id']
        self.last_draw=None
        self.check_inventory()
        return deepcopy(ref)

    def rob_kan(self,tile):
        from riichienv.convert import tid_to_mjai
        check(self.pending is not None,'INVALID_PHASE','no kan to rob')
        pending=self.pending
        kind,actor=pending['kind'],pending['actor']
        check((kind=='ankan' and self.options.allow_ankan_chankan) or
              (kind=='kakan' and self.options.allow_kakan_chankan),'UNSUPPORTED_PROFILE','kan robbery')
        check(tile in pending['reserved'],'INVALID_INVENTORY','robbed tile')
        hands=self.env.hands
        index=next(i for i,t in enumerate(hands[actor]) if tid_to_mjai(t)==tile)
        hands[actor].pop(index)
        self.env.hands=hands
        self.pending=None
        self.action_actor=None
        return deepcopy(pending)

    def reveal_dora(self,event):
        check(self.dora_pending and self.dora_pending[0]['eligible'] and len(self.env.dora_indicators)<5,
              'DORA_TIMING_MISMATCH','oldest committed kan is not eligible')
        self.check_inventory([event['dora_marker']])
        consumed=deepcopy(self.dora_pending[0])
        self.env.apply_event(event)
        self.dora_pending.pop(0)
        self.env.pending_kan_dora_count-=1
        self.check_inventory()
        return consumed

    def check_play_phase(self):
        check(not self.ended,'INVALID_PHASE','event after round result')

    def check_result_phase(self,events=None):
        check(not self.ended,'INVALID_PHASE','duplicate settlement')
        check(self.declaration_pending_discard is None,'INVALID_PHASE',
              'riichi declaration must be followed by its discard before a result')
        if (self.abort_rules is not None and events and events[0]['type']=='hora' and
                events[0]['actor']!=events[0]['target']):
            self.abort_rules.check_choice('ron',self.boundary_abort_causes()+['ron'])

    def settle(self,events,*,claims=None,draw_facts=None):
        self.check_result_phase(events)
        check(events,'INVALID_PHASE','empty settlement')
        self.require_revealed_dora()
        result=recorded_settlement(self.options,events,self.env.scores(),self.env.riichi_sticks)
        if draw_facts is not None:
            if draw_facts['kind']=='abort':
                check(result['score_delta']==[0]*4,'SETTLEMENT_MISMATCH','standard abort cannot transfer scores')
                result['payments']=[]
            elif self.payment_rules is not None:
                expected=self.payment_rules.draw(result['scores_before'],dealer=self.header['oya'],
                    honba=self.header['honba'],kyotaku=result['kyotaku'],tenpai=draw_facts['tenpai'],
                    nagashi=draw_facts['nagashi'],nagashi_extras=draw_facts.get('nagashi_extras'))
                check(expected['score_delta']==result['score_delta'] and expected['next_kyotaku']==result['next_kyotaku'],
                      'SETTLEMENT_MISMATCH','recorded draw payments disagree with explicit facts/policy')
                result['payments']=expected['payments']
        if claims is not None and self.payment_rules is not None:
            expected=self.payment_rules.win(result['scores_before'],dealer=self.header['oya'],honba=self.header['honba'],
                                            kyotaku=result['kyotaku'],claims=claims)
            check(set(expected['winners'])=={e['actor'] for e in events} and expected['score_delta']==result['score_delta'],
                  'SETTLEMENT_MISMATCH','recorded win payments disagree with explicit values/policy')
            if self.options.settlement_deltas=='per_event':
                before=list(result['scores_before'])
                for index,(claim,event) in enumerate(zip(claims,events)):
                    honba=self.header['honba'] if index==0 or self.payment_rules.policy['multi_ron_honba']=='each' else 0
                    part=self.payment_rules.win(before,dealer=self.header['oya'],honba=honba,
                        kyotaku=result['kyotaku'] if index==0 else 0,claims=[claim])
                    check(part['score_delta']==event['deltas'],'SETTLEMENT_MISMATCH','recorded per-winner payment')
                    before=part['scores_after']
            result['payments']=expected['payments']
        self.env.set_scores(result['scores_after'])
        self.env.riichi_sticks=result['next_kyotaku']
        self.ended=True
        self.action_actor=None
        self.round_result=(dict(kind='win',winners=[e['actor'] for e in events])
                           if events[0]['type']=='hora' else
                           deepcopy(draw_facts) if draw_facts is not None else
                           dict(kind='unknown_draw',tenpai=events[0].get('tenpais',[None]*4),
                                may_abort=result['score_delta']==[0]*4))
        return result

    def round_decision(self):
        check(self.ended and self.header is not None and not self.finalized,
              'INVALID_PHASE','round decision requires completed settlement before final payment')
        return self.round_rules.decide(self.header,self.env.scores(),self.env.riichi_sticks,
            self.round_result,starting_dealer=self.starting_dealer) if self.round_rules is not None else None

    def finalize(self,recorded_scores=None):
        check(self.ended and self.header is not None and not self.finalized,'INVALID_PHASE','premature/duplicate final settlement')
        decision=self.round_decision()
        if decision is not None:decision.check_boundary()
        scores,kyotaku=self.env.scores(),self.env.riichi_sticks
        if self.payment_rules is None:
            check(kyotaku==0,'MISSING_FINAL_SETTLEMENT','terminal sticks require an explicit payment policy')
            check(recorded_scores is None or recorded_scores==scores,'SETTLEMENT_MISMATCH','final scores')
            self.finalized=True
            self.end_reason=None if decision is None else decision.reason
            return None,None
        check(self.starting_dealer is not None,'MISSING_FINAL_IDENTITY','starting dealer is not proved')
        result=self.payment_rules.final(scores,kyotaku,starting_dealer=self.starting_dealer,
                                        recorded_scores=recorded_scores)
        # Check recorded balances before changing the state owner.
        self.env.set_scores(result['scores_after'])
        self.env.riichi_sticks=result['next_kyotaku']
        ranks=self.payment_rules.ranks(result['scores_after'],self.starting_dealer)
        self.finalized=True
        self.end_reason=None if decision is None else decision.reason
        return result,ranks
