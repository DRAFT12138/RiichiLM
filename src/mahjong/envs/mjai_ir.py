"""Canonical MJAI IR business objects and explicit unknown facts.

These checks are preprocessing checks. They never change the Tokenizer and do
not implement the rules engine, action enumeration, or scoring inference.
"""
from decimal import Decimal

from mahjong.tokenizer import Atom, Event, MahjongTokenizer
from mahjong.tokenizer.codec import EVENT_FIELDS
from .mjai_errors import require
from .ir_primitives import TILES, natural

UNKNOWN = Atom('UNKNOWN')
HIDDEN = Atom('HIDDEN')
ABS = tuple(f'ABS_P{i}' for i in range(4))
PHASES = ('PHASE_SELF_DECISION', 'PHASE_AFTER_CALL',
          'PHASE_DISCARD_RESPONSE', 'PHASE_KAN_RESPONSE')
RESOLUTIONS = ('RESOLVE_NO_CALL', 'RESOLVE_CHI', 'RESOLVE_PON',
               'RESOLVE_DAIMINKAN', 'RESOLVE_RON', 'RESOLVE_ABORT', 'RESOLVE_KAN')
END_REASONS = ('END_NORMAL', 'END_BUST', 'END_TARGET_REACHED', 'END_MAX_ROUND',
               'END_DEALER_STOP', 'END_CANCELLED', 'END_CUSTOM')
ABORTS = ('ABORT_KYUUSHU', 'ABORT_SUUFON', 'ABORT_SUUCHA_RIICHI',
          'ABORT_SUUKAIKAN', 'ABORT_SANCHA_HOU')
LIMITS = ('LIMIT_NONE', 'LIMIT_MANGAN', 'LIMIT_HANEMAN', 'LIMIT_BAIMAN',
          'LIMIT_SANBAIMAN', 'LIMIT_KAZOE_YAKUMAN', 'LIMIT_YAKUMAN')
YAKU = ('Y_RIICHI','Y_IPPATSU','Y_MENZEN_TSUMO','Y_TANYAO','Y_PINFU','Y_IIPEIKOU',
        'Y_SEAT_WIND','Y_ROUND_WIND','Y_HAKU','Y_HATSU','Y_CHUN','Y_RINSHAN','Y_CHANKAN',
        'Y_HAITEI','Y_HOUTEI','Y_DOUBLE_RIICHI','Y_CHIITOITSU','Y_CHANTA','Y_ITTSUU',
        'Y_SANSHOKU_DOUJUN','Y_SANSHOKU_DOUKOU','Y_SANKANTSU','Y_TOITOI','Y_SANANKOU',
        'Y_SHOUSANGEN','Y_HONROUTOU','Y_RYANPEIKOU','Y_JUNCHAN','Y_HONITSU','Y_CHINITSU')
YAKUMAN = ('Y_TENHOU','Y_CHIIHOU','Y_DAISANGEN','Y_SUUANKOU','Y_TSUUIISOU','Y_RYUUIISOU',
           'Y_CHINROUTOU','Y_KOKUSHI','Y_SHOUSUUSHII','Y_DAISUUSHII','Y_SUUKANTSU',
           'Y_CHUUREN','Y_SUUANKOU_TANKI','Y_KOKUSHI_13','Y_JUNSEI_CHUUREN')


class IRProtocolFailure(RuntimeError):
    """Generated IR violates its contract; never classify this as a bad input."""


def fields(value, names, label):
    require(type(value) is dict, 'TYPE_MISMATCH', label)
    expected = set(names.split())
    require(expected <= value.keys(), 'MISSING_FIELD', label)
    require(value.keys() == expected, 'UNKNOWN_FIELD', label)


def enum(value, names, label):
    require(type(value) is Atom and value.name in names, 'TYPE_MISMATCH', label)


def seat(value, label='seat'):
    enum(value, ABS, label)


def tile(value, label='tile'):
    enum(value, TILES, label)


def boolean(value, label):
    require(type(value) is bool, 'TYPE_MISMATCH', label)


def vector(value, validator, label):
    require(type(value) is list and len(value) == 4, 'TYPE_MISMATCH', label)
    for item in value:
        validator(item, label)


def integer(value, label):
    require(type(value) is int, 'TYPE_MISMATCH', label)


def bag(value, label):
    require(type(value) is list, 'TYPE_MISMATCH', label)
    for item in value:
        tile(item, label)


def check_round_start(value):
    fields(value, 'round_wind round_number honba kyotaku rounds_played dealer '
           'scores hands hand_counts dora_indicators', 'RoundStartObservation')
    enum(value['round_wind'], ('ROUND_E', 'ROUND_S', 'ROUND_W', 'ROUND_N'), 'round_wind')
    natural(value['round_number'], 'round_number')
    require(1 <= value['round_number'] <= 4, 'NONINITIAL_VALUE', 'round_number')
    for name in ('honba', 'kyotaku', 'rounds_played'):
        natural(value[name], name)
    seat(value['dealer'], 'dealer')
    vector(value['scores'], integer, 'scores')
    vector(value['hands'], bag, 'initial hands')
    vector(value['hand_counts'], natural, 'hand_counts')
    require(value['hand_counts'] == [13] * 4 and all(len(h) == 13 for h in value['hands']),
            'NONINITIAL_VALUE', 'initial hand size')
    bag(value['dora_indicators'], 'dora_indicators')
    require(len(value['dora_indicators']) == 1, 'NONINITIAL_VALUE', 'initial dora count')


def check_window(value, events):
    fields(value, 'window_id trigger_id phase actor source tile responders', 'Window')
    natural(value['window_id'], 'window_id')
    trigger = value['trigger_id']
    natural(trigger, 'trigger_id')
    require(trigger < len(events), 'INVALID_REFERENCE', 'window trigger')
    enum(value['phase'], PHASES, 'phase')
    source = value['source']
    seat(source, 'source')
    actor = int(source.name[-1])
    slot = PHASES.index(value['phase'].name)
    expected_kind = (('EV_DRAW',), ('EV_CHI', 'EV_PON'), ('EV_DISCARD',),
                     ('EV_ANKAN_ATTEMPT', 'EV_KAKAN_ATTEMPT'))[slot]
    require(events[trigger].kind in expected_kind and events[trigger].payload[0] == source and
            value['window_id'] == 4 * trigger + slot, 'INVALID_REFERENCE', 'window kind/actor/formula')
    require(type(value['responders']) is list, 'TYPE_MISMATCH', 'responders')
    expected = [source] if slot < 2 else [Atom(ABS[(actor + i) % 4]) for i in range(1, 4)]
    require(value['responders'] == expected, 'VISIBILITY_VIOLATION', 'window responders')
    if slot < 2:
        require(value['actor'] == source and value['tile'] is None,
                'INVALID_REFERENCE', 'self window actor/tile')
    else:
        require(value['actor'] is None, 'TYPE_MISMATCH', 'response actor')
        tile(value['tile'])
        expected_tile = events[trigger].payload[1]
        require(value['tile'] == expected_tile, 'INVALID_REFERENCE', 'window tile')


def check_close(window_id, resolution, result, windows):
    natural(window_id, 'window_id')
    require(window_id in windows, 'INVALID_REFERENCE', 'close without open')
    window = windows[window_id]
    phase = window['phase']
    require(phase in (Atom('PHASE_DISCARD_RESPONSE'), Atom('PHASE_KAN_RESPONSE')),
            'INVALID_REFERENCE', 'close requires a response window')
    enum(resolution, RESOLUTIONS, 'resolution')
    fields(result, 'actor meld_id winners reason', 'CloseResult')
    if result['actor'] is not None:
        seat(result['actor'], 'close actor')
    if result['meld_id'] is not None:
        natural(result['meld_id'], 'close meld_id')
    require(type(result['winners']) is list and len(result['winners']) <= 3,
            'TYPE_MISMATCH', 'close winners')
    for winner in result['winners']:
        seat(winner, 'winner')
    require(len(set(result['winners'])) == len(result['winners']), 'INVALID_REFERENCE', 'duplicate winner')
    name = resolution.name
    if name in ('RESOLVE_CHI', 'RESOLVE_PON', 'RESOLVE_DAIMINKAN', 'RESOLVE_KAN'):
        require(result['actor'] is not None and result['meld_id'] is not None and
                result['winners'] == [] and result['reason'] is None,
                'INVALID_REFERENCE', 'call/kan resolution')
        if name == 'RESOLVE_KAN':
            require(phase == Atom('PHASE_KAN_RESPONSE') and result['actor'] == window['source'],
                    'INVALID_REFERENCE', 'kan resolution actor/phase')
        else:
            require(phase == Atom('PHASE_DISCARD_RESPONSE') and result['actor'] in window['responders'],
                    'INVALID_REFERENCE', 'call response actor/phase')
            if name == 'RESOLVE_CHI':
                require(result['actor'] == window['responders'][0], 'INVALID_REFERENCE', 'chi direction')
    elif name == 'RESOLVE_RON':
        require(result['actor'] is None and result['meld_id'] is None and
                result['winners'] and result['reason'] is None,
                'INVALID_REFERENCE', 'ron resolution')
        responders = windows[window_id]['responders']
        require(all(w in responders for w in result['winners']), 'INVALID_REFERENCE', 'winner outside response')
        require(result['winners'] == sorted(result['winners'], key=responders.index),
                'NONCANONICAL', 'winner order')
    elif name == 'RESOLVE_NO_CALL':
        require(phase == Atom('PHASE_DISCARD_RESPONSE') and
                result == dict(actor=None, meld_id=None, winners=[], reason=None),
                'INVALID_REFERENCE', 'no-call facts')
    else:
        enum(result['reason'], ABORTS, 'abort reason')
        require(result['actor'] is None and result['meld_id'] is None and result['winners'] == [],
                'INVALID_REFERENCE', 'abort close contains only the reason')


def check_payments(value):
    if value == UNKNOWN:
        return
    require(type(value) is list, 'TYPE_MISMATCH', 'payments')
    keys=[]
    codec=MahjongTokenizer()
    for payment in value:
        fields(payment, 'payer payee amount payment_kind', 'Payment')
        enum(payment['payment_kind'], ('PAYMENT_BASE', 'PAYMENT_HONBA', 'PAYMENT_KYOTAKU',
             'PAYMENT_NOTEN', 'PAYMENT_NAGASHI', 'PAYMENT_PAO'), 'payment_kind')
        for name in ('payer', 'payee'):
            if payment[name] is not None:
                seat(payment[name], name)
        require(payment['payer'] != payment['payee'] and
                (payment['payer'] is not None and payment['payee'] is not None or
                 payment['payment_kind'] == Atom('PAYMENT_KYOTAKU')),
                'TYPE_MISMATCH', 'payment participants')
        natural(payment['amount'], 'amount')
        require(payment['amount'] > 0, 'TYPE_MISMATCH', 'positive payment')
        keys.append(tuple(codec.token_id('NULL') if payment[k] is None else codec.token_id(payment[k].name)
                          for k in ('payer','payee','payment_kind')))
    require(keys==sorted(set(keys)), 'NONCANONICAL', 'payments must be merged and sorted')


def check_settlement(value):
    fields(value, 'scores_before scores_after score_delta kyotaku next_kyotaku payments', 'Settlement')
    for name in ('scores_before', 'scores_after', 'score_delta'):
        vector(value[name], integer, name)
    for name in ('kyotaku', 'next_kyotaku'):
        natural(value[name], name)
    require([b+d for b,d in zip(value['scores_before'], value['score_delta'])] == value['scores_after'],
            'SETTLEMENT_MISMATCH', 'scores plus delta')
    require(sum(value['scores_before']) + 1000*value['kyotaku'] ==
            sum(value['scores_after']) + 1000*value['next_kyotaku'],
            'SETTLEMENT_MISMATCH', 'money plus sticks')
    check_payments(value['payments'])
    if value['payments'] != UNKNOWN:
        delta=[0]*4
        pool_delta=0
        for payment in value['payments']:
            amount=payment['amount']
            if payment['payer'] is None:
                pool_delta-=amount
            else:
                delta[int(payment['payer'].name[-1])]-=amount
            if payment['payee'] is None:
                pool_delta+=amount
            else:
                delta[int(payment['payee'].name[-1])]+=amount
        require(delta==value['score_delta'] and pool_delta==1000*(value['next_kyotaku']-value['kyotaku']),
                'SETTLEMENT_MISMATCH', 'payment breakdown')


def check_yaku(value):
    if value == UNKNOWN:
        return
    require(type(value) is list and value, 'TYPE_MISMATCH', 'nonempty yaku')
    names=[]
    for yaku in value:
        require(type(yaku) is dict and 'name' in yaku,'MISSING_FIELD','yaku')
        enum(yaku['name'],(*YAKU,*YAKUMAN),'yaku name')
        name=yaku['name'].name
        names.append(name)
        if name in YAKUMAN:
            fields(yaku,'name yakuman_units pao_player','yakuman yaku')
            natural(yaku['yakuman_units'],'yakuman_units')
            require(yaku['yakuman_units']>0,'TYPE_MISMATCH','positive yakuman units')
            if yaku['pao_player'] is not None:
                seat(yaku['pao_player'],'pao_player')
        else:
            fields(yaku,'name han','ordinary yaku')
            natural(yaku['han'],'yaku han')
            require(yaku['han']>0,'TYPE_MISMATCH','positive yaku han')
    codec=MahjongTokenizer()
    require(names==sorted(set(names),key=codec.token_id),'NONCANONICAL','duplicate/unsorted yaku')
    require(not (any(n in YAKUMAN for n in names) and any(n in YAKU for n in names)),
            'TYPE_MISMATCH','mixed ordinary/yakuman yaku')


def check_pao(value, yaku):
    if value == UNKNOWN:
        return
    require(type(value) is list,'TYPE_MISMATCH','pao_components')
    names=[]
    for component in value:
        fields(component,'name yakuman_units pao_player trigger_event','PaoComponent')
        enum(component['name'],YAKUMAN,'pao yaku')
        natural(component['yakuman_units'],'pao units')
        require(component['yakuman_units']>0,'TYPE_MISMATCH','positive pao units')
        seat(component['pao_player'],'pao_player')
        natural(component['trigger_event'],'pao trigger_event')
        require(type(yaku) is list and any(y['name']==component['name'] and
                y.get('yakuman_units')==component['yakuman_units'] and
                y.get('pao_player')==component['pao_player'] for y in yaku),
                'INVALID_REFERENCE','pao/yaku correspondence')
        names.append(component['name'].name)
    require(names==sorted(set(names),key=MahjongTokenizer().token_id),'NONCANONICAL','pao order')
    if type(yaku) is list:
        responsible={y['name'].name for y in yaku if y.get('pao_player') is not None}
        require(set(names)==responsible, 'INVALID_REFERENCE', 'complete pao/yaku correspondence')


def check_winner(value):
    fields(value, 'actor target method tile yaku yaku_han han_total fu dora_count aka_count '
           'ura_count limit yakuman_units pao_components revealed_hands ura_indicators', 'Winner')
    seat(value['actor'], 'winner actor')
    enum(value['method'], ('METHOD_RON', 'METHOD_TSUMO'), 'method')
    if value['method'] == Atom('METHOD_RON'):
        seat(value['target'], 'target')
        require(value['target'] != value['actor'], 'INVALID_REFERENCE', 'self ron')
    else:
        require(value['target'] is None, 'TYPE_MISMATCH', 'tsumo target')
    tile(value['tile'])
    if value['yakuman_units'] != UNKNOWN:
        natural(value['yakuman_units'], 'yakuman_units')
    for name in ('yaku_han', 'han_total', 'fu', 'dora_count', 'aka_count', 'ura_count'):
        if value[name] != UNKNOWN and value[name] is not None:
            natural(value[name], name)
    if value['limit'] != UNKNOWN:
        enum(value['limit'], LIMITS, 'limit')
    check_yaku(value['yaku'])
    check_pao(value['pao_components'],value['yaku'])
    if type(value['yaku']) is list:
        require(all(y.get('pao_player') != value['actor'] for y in value['yaku']),
                'INVALID_REFERENCE', 'winner cannot be responsible for their own win')
        yakuman=any(y['name'].name in YAKUMAN for y in value['yaku'])
        if yakuman:
            require(value['yakuman_units']==sum(y['yakuman_units'] for y in value['yaku']),
                    'TYPE_MISMATCH','yakuman unit sum')
        else:
            require(value['yaku_han']==sum(y['han'] for y in value['yaku']) and value['yakuman_units']==0,
                    'TYPE_MISMATCH','ordinary yaku han/units')
            if type(value['han_total']) is int:
                require(value['han_total']>=value['yaku_han'],'TYPE_MISMATCH','han_total below yaku_han')
    if type(value['yakuman_units']) is int and value['yakuman_units'] > 0:
        require(all(value[k] is None for k in ('yaku_han', 'han_total', 'fu', 'dora_count', 'aka_count', 'ura_count')),
                'TYPE_MISMATCH', 'yakuman non-applicable fields')
    require(type(value['revealed_hands']) is list and len(value['revealed_hands']) == 4,
            'TYPE_MISMATCH', 'revealed_hands')
    for hand in value['revealed_hands']:
        if hand not in (UNKNOWN, HIDDEN):
            bag(hand, 'revealed_hands')
    if value['ura_indicators'] != UNKNOWN and value['ura_indicators'] is not None:
        bag(value['ura_indicators'], 'ura_indicators')
        require(len(value['ura_indicators']) <= 5, 'TYPE_MISMATCH', 'ura size')


def check_win_group(value, events, windows):
    fields(value, 'window_id trigger_id winners', 'WinGroup')
    natural(value['window_id'], 'window_id')
    natural(value['trigger_id'], 'trigger_id')
    require(value['window_id'] in windows and windows[value['window_id']]['trigger_id'] == value['trigger_id'],
            'INVALID_REFERENCE', 'win window')
    require(type(value['winners']) is list and 1 <= len(value['winners']) <= 3,
            'TYPE_MISMATCH', 'win group size')
    for winner in value['winners']:
        check_winner(winner)
        if type(winner['pao_components']) is list:
            for component in winner['pao_components']:
                trigger=component['trigger_event']
                require(trigger<len(events) and events[trigger].kind in ('EV_PON','EV_DAIMINKAN') and
                        events[trigger].payload[0]==winner['actor'] and
                        events[trigger].payload[1]==component['pao_player'],
                        'INVALID_REFERENCE','pao public meld trigger')
    actors = [w['actor'] for w in value['winners']]
    require(len(set(actors)) == len(actors), 'INVALID_REFERENCE', 'duplicate winner')
    methods = {w['method'] for w in value['winners']}
    require(len(methods) == 1, 'INVALID_REFERENCE', 'mixed win methods')
    window = windows[value['window_id']]
    if methods == {Atom('METHOD_TSUMO')}:
        require(len(actors) == 1 and actors == window['responders'] and
                window['phase'] == Atom('PHASE_SELF_DECISION'), 'INVALID_REFERENCE', 'tsumo window')
        require(value['winners'][0]['tile'] == events[value['trigger_id']].payload[1],
                'INVALID_REFERENCE', 'tsumo tile')
    else:
        require(window['phase'] in (Atom('PHASE_DISCARD_RESPONSE'), Atom('PHASE_KAN_RESPONSE')) and
                all(w['target'] == window['source'] for w in value['winners']) and
                all(w['tile'] == value['winners'][0]['tile'] for w in value['winners']),
                'INVALID_REFERENCE', 'shared ron trigger')
        require(all(a in window['responders'] for a in actors) and
                actors == sorted(actors, key=window['responders'].index), 'NONCANONICAL', 'win order/participants')
        if window['phase'] == Atom('PHASE_DISCARD_RESPONSE'):
            require(value['winners'][0]['tile'] == window['tile'], 'INVALID_REFERENCE', 'ron tile')
        else:
            attempt = events[value['trigger_id']]
            winning_tile = value['winners'][0]['tile']
            if attempt.kind == 'EV_ANKAN_ATTEMPT':
                require(winning_tile in attempt.payload[2], 'INVALID_REFERENCE', 'robbed ankan copy')
            else:
                require(winning_tile == window['tile'], 'INVALID_REFERENCE', 'robbed kakan copy')


def check_draw_result(value):
    fields(value, 'reason tenpai_states score_delta', 'DrawResult')
    if value['reason'] != UNKNOWN:
        enum(value['reason'], ('DRAW_EXHAUSTIVE',), 'draw reason')
    vector(value['tenpai_states'], lambda x,label: enum(x, ('TENPAI','NOTEN','HIDDEN','UNKNOWN'), label), 'tenpai_states')
    if value['score_delta'] != UNKNOWN:
        vector(value['score_delta'], integer, 'score_delta')


def check_next_state(value):
    fields(value, 'game_over next_dealer next_round next_honba next_kyotaku reason', 'NextState')
    over = value['game_over']
    if over == UNKNOWN:
        require(all(value[k] == UNKNOWN for k in value if k != 'game_over'),
                'TYPE_MISMATCH', 'unknown next state')
    elif over is False:
        seat(value['next_dealer'], 'next_dealer')
        fields(value['next_round'], 'round_wind round_number rounds_played', 'RoundRef')
        enum(value['next_round']['round_wind'], ('ROUND_E','ROUND_S','ROUND_W','ROUND_N'), 'next round wind')
        natural(value['next_round']['round_number'], 'next round number')
        require(1 <= value['next_round']['round_number'] <= 4, 'TYPE_MISMATCH', 'next round number')
        natural(value['next_round']['rounds_played'], 'rounds_played')
        natural(value['next_honba'], 'next_honba')
        natural(value['next_kyotaku'], 'next_kyotaku')
        require(value['reason'] is None, 'TYPE_MISMATCH', 'live next state reason')
    else:
        require(over is True, 'TYPE_MISMATCH', 'game_over bool')
        require(all(value[k] is None for k in ('next_dealer','next_round','next_honba')),
                'TYPE_MISMATCH', 'terminal next fields')
        natural(value['next_kyotaku'], 'next_kyotaku')
        if value['reason'] != UNKNOWN:
            enum(value['reason'], END_REASONS, 'end reason')


def check_final_result(value):
    fields(value, 'final_scores final_ranks rank_points unit reason', 'FinalResult')
    vector(value['final_scores'], integer, 'final_scores')
    if value['final_ranks'] != UNKNOWN:
        vector(value['final_ranks'], integer, 'final_ranks')
        require(set(value['final_ranks']) == {1,2,3,4}, 'TYPE_MISMATCH', 'rank permutation')
        require(all(value['final_scores'][a] <= value['final_scores'][b] or
                    value['final_ranks'][a] < value['final_ranks'][b]
                    for a in range(4) for b in range(4)), 'INVALID_REFERENCE', 'ranks contradict scores')
    if value['rank_points'] != UNKNOWN:
        vector(value['rank_points'], lambda x,label: require(type(x) is Decimal and x.is_finite(), 'TYPE_MISMATCH', label), 'rank_points')
    enum(value['unit'], ('UNIT_RANK_POINTS',), 'unit')
    if value['reason'] != UNKNOWN:
        enum(value['reason'], END_REASONS, 'end reason')


def check_abort_result(value):
    fields(value,'reason actor score_delta extensions','AbortResult')
    enum(value['reason'],ABORTS,'abort reason')
    if value['reason']==Atom('ABORT_KYUUSHU'):
        seat(value['actor'],'abort actor')
    else:
        require(value['actor'] is None,'TYPE_MISMATCH','non-personal abort actor')
    if value['score_delta']!=UNKNOWN:
        vector(value['score_delta'],integer,'abort score_delta')
    require(value['extensions']==[],'UNKNOWN_FIELD','standard abort extensions')


def check_nagashi_result(value):
    fields(value,'winners payments score_delta tenpai_states','NagashiResult')
    require(type(value['winners']) is list and value['winners'],'TYPE_MISMATCH','nagashi winners')
    for winner in value['winners']:
        seat(winner,'nagashi winner')
    require(len(set(value['winners']))==len(value['winners']),'INVALID_REFERENCE','duplicate nagashi winner')
    check_payments(value['payments'])
    if value['score_delta']!=UNKNOWN:
        vector(value['score_delta'],integer,'nagashi score_delta')
    vector(value['tenpai_states'],lambda x,label: enum(x,('TENPAI','NOTEN','HIDDEN','UNKNOWN'),label),'nagashi tenpai')


def check_reveal(value):
    fields(value,'revealed_hands ura_indicators tenpai_states','Reveal')
    def hand(item,label):
        if item not in (UNKNOWN,HIDDEN):
            bag(item,label)
    vector(value['revealed_hands'],hand,'revealed_hands')
    if value['ura_indicators'] not in (UNKNOWN,HIDDEN,None):
        bag(value['ura_indicators'],'ura_indicators')
        require(len(value['ura_indicators'])<=5,'TYPE_MISMATCH','ura size')
    vector(value['tenpai_states'],lambda x,label: enum(x,('TENPAI','NOTEN','HIDDEN','UNKNOWN'),label),'reveal tenpai')


def check_ir_lifecycle(event, events, windows):
    """Check causal IR transitions against the retained prefix, without state.

    Rollback therefore needs no second mutable lifecycle ledger. Only public
    dora/riichi facts may separate an adjudication from its successful action.
    Hand legality and response eligibility remain environment responsibilities.
    """
    kind = event.kind
    if events and events[-1].kind == 'EV_RESPONSE_OPEN':
        require(kind == 'EV_RESPONSE_CLOSE','INVALID_REFERENCE','open response must close before another IR event')
    latest = next((e for e in reversed(events) if e.kind in ('EV_RESPONSE_OPEN','EV_RESPONSE_CLOSE')),None)
    public = ('EV_DORA_REVEAL','EV_RIICHI_ACCEPT')
    if kind not in ('EV_RESPONSE_OPEN','EV_RESPONSE_CLOSE','EV_WIN','EV_CHI','EV_PON',
                    'EV_DAIMINKAN','EV_KAN_COMMIT','EV_KAN_ROBBED'):
        if latest is not None and latest.kind == 'EV_RESPONSE_CLOSE' and kind not in public:
            expected = {'RESOLVE_CHI':'EV_CHI','RESOLVE_PON':'EV_PON',
                'RESOLVE_DAIMINKAN':'EV_KAN_COMMIT','RESOLVE_KAN':'EV_KAN_COMMIT',
                'RESOLVE_RON':'EV_WIN','RESOLVE_ABORT':'EV_ABORT_END'}.get(latest.payload[1].name)
            done = expected is None or any(e.kind == expected for e in events[latest.event_id+1:])
            require(done or kind == expected == 'EV_ABORT_END',
                    'INVALID_REFERENCE','public adjudication outcome cannot be skipped')
            if not done:
                require(event.payload[0] == latest.payload[2]['reason'],
                        'INVALID_REFERENCE','abort must match its public reason')
        return
    if kind == 'EV_RESPONSE_OPEN':
        window = event.payload[0]
        require(window['phase'] in (Atom('PHASE_DISCARD_RESPONSE'),Atom('PHASE_KAN_RESPONSE')) and
                window['trigger_id'] == event.event_id-1 and window['window_id'] not in windows and
                (latest is None or latest.kind == 'EV_RESPONSE_CLOSE'),
                'INVALID_REFERENCE','response open must follow its trigger, once, without an overlapping window')
        return
    if kind == 'EV_RESPONSE_CLOSE':
        require(latest is not None and latest.kind == 'EV_RESPONSE_OPEN' and
                latest.payload[0]['window_id'] == event.payload[0] and latest.event_id == event.event_id-1,
                'INVALID_REFERENCE','response close requires the current unclosed open')
        return
    if kind == 'EV_WIN' and event.payload[0]['winners'][0]['method'] == Atom('METHOD_TSUMO'):
        trigger = event.payload[0]['trigger_id']
        require((latest is None or latest.kind == 'EV_RESPONSE_CLOSE') and
                all(e.kind in public for e in events[trigger+1:]),
                'INVALID_REFERENCE','tsumo must use its current unconsumed draw')
        return
    require(latest is not None and latest.kind == 'EV_RESPONSE_CLOSE',
            'INVALID_REFERENCE','successful response requires a public adjudication')
    wid,resolution,result = latest.payload
    window = windows[wid]
    tail = events[latest.event_id+1:]
    actions = [e for e in tail if e.kind not in public]
    if kind == 'EV_WIN':
        group = event.payload[0]
        require(group['window_id'] == wid and resolution == Atom('RESOLVE_RON') and
                [w['actor'] for w in group['winners']] == result['winners'],
                'INVALID_REFERENCE','win group must match its public ron adjudication')
        if window['phase'] == Atom('PHASE_KAN_RESPONSE'):
            require(len(actions) == 1 and actions[0].kind == 'EV_KAN_ROBBED' and actions[0].payload[2] == wid,
                    'INVALID_REFERENCE','kan ron requires exactly one robbed transition')
        else:
            require(not actions,'INVALID_REFERENCE','ron window already consumed or passed')
    elif kind in ('EV_CHI','EV_PON','EV_DAIMINKAN'):
        value = dict(zip(EVENT_FIELDS[kind],event.payload))
        require(not actions and resolution == Atom('RESOLVE_'+kind[3:]) and
                value['actor'] == result['actor'] and value['meld_id'] == result['meld_id'] and
                value['target'] == window['source'] and value['tile'] == window['tile'] and
                value['trigger_id'] == window['trigger_id'],
                'INVALID_REFERENCE','call must consume its own public authorization')
    elif kind == 'EV_KAN_COMMIT':
        actor,meld = event.payload
        require(resolution in (Atom('RESOLVE_KAN'),Atom('RESOLVE_DAIMINKAN')) and
                actor == result['actor'] and meld == result['meld_id'],
                'INVALID_REFERENCE','commit must match its accepted kan')
        if resolution == Atom('RESOLVE_DAIMINKAN'):
            require(len(actions) == 1 and actions[0].kind == 'EV_DAIMINKAN' and
                    actions[0].payload[0] == actor and actions[0].payload[4] == meld and
                    actions[0].payload[1] == window['source'] and actions[0].payload[5] == window['trigger_id'],
                    'INVALID_REFERENCE','daiminkan requires its authorized call before commit')
        else:
            require(not actions,'INVALID_REFERENCE','kan already consumed or passed')
    else:
        actor,meld,robbed_window = event.payload
        require(not actions and resolution == Atom('RESOLVE_RON') and robbed_window == wid and
                window['phase'] == Atom('PHASE_KAN_RESPONSE') and actor == window['source'],
                'INVALID_REFERENCE','robbery must match the current kan ron')
        attempt = events[window['trigger_id']]
        expected_meld = attempt.payload[3 if attempt.kind == 'EV_ANKAN_ATTEMPT' else 2]
        require(meld == expected_meld,'INVALID_REFERENCE','robbed meld must match its attempt')


def check_next_state_context(value, events):
    check_next_state(value)
    if value['game_over'] == UNKNOWN:
        return
    settlement = next((e for e in reversed(events) if e.kind == 'EV_SETTLEMENT'),None)
    require(settlement is not None and value['next_kyotaku'] == settlement.payload[0]['next_kyotaku'],
            'INVALID_REFERENCE','next state pool must match the latest settlement')
    if value['game_over'] is False:
        end = next((e for e in reversed(events) if e.kind == 'EV_ROUND_END'),None)
        require(end is None or settlement.event_id < end.event_id,
                'INVALID_REFERENCE','live next state cannot follow a terminal pool adjustment')


def check_result_lifecycle(event, events):
    """Correlate results, money and boundaries; never calculate hand payments."""
    kind = event.kind
    results = ('EV_WIN','EV_DRAW_END','EV_ABORT_END','EV_NAGASHI')
    boundary = (*results,'EV_REVEAL','EV_SETTLEMENT','EV_ROUND_END','EV_GAME_END')
    if events:
        last = events[-1].kind
        require(last != 'EV_GAME_END','INVALID_REFERENCE','IR after game end')
        if last == 'EV_ROUND_END':
            require(kind in ('EV_SETTLEMENT','EV_GAME_END'),'INVALID_REFERENCE','IR after round end')
        if last == 'EV_SETTLEMENT' and len(events)>1 and events[-2].kind == 'EV_ROUND_END':
            require(kind == 'EV_GAME_END','INVALID_REFERENCE','terminal pool adjustment must lead to game end')
        if last in (*results,'EV_REVEAL','EV_SETTLEMENT'):
            require(kind in boundary,'INVALID_REFERENCE','play cannot resume after a round result')
    if kind not in boundary:
        return
    require(events and events[0].kind == 'EV_ROUND_START','INVALID_REFERENCE','result requires round context')
    prior_results = [e for e in events if e.kind in results]
    settlements = [e for e in events if e.kind == 'EV_SETTLEMENT']
    ends = [e for e in events if e.kind == 'EV_ROUND_END']
    if kind in results:
        require(not prior_results and not settlements and not ends,'INVALID_REFERENCE','one result group per round')
    elif kind == 'EV_REVEAL':
        require(len(prior_results)==1 and not settlements and not any(e.kind=='EV_REVEAL' for e in events),
                'INVALID_REFERENCE','reveal requires an unsettled result, once')
        prior = prior_results[0]
        if prior.kind in ('EV_DRAW_END','EV_NAGASHI'):
            known = (Atom('TENPAI'),Atom('NOTEN'))
            require(all(a not in known or b not in known or a==b for a,b in zip(
                prior.payload[0]['tenpai_states'],event.payload[0]['tenpai_states'])),
                'INVALID_REFERENCE','reveal tenpai contradicts result')
    elif kind == 'EV_SETTLEMENT':
        require(len(prior_results)==1 and len(settlements)==int(bool(ends)) and len(ends)<=1,
                'INVALID_REFERENCE','settlement must consume one result or one terminal pool adjustment')
        value = event.payload[0]
        if settlements:
            previous = settlements[-1]
            scores = list(previous.payload[0]['scores_after'])
            pool = previous.payload[0]['next_kyotaku']
            offset = previous.event_id+1
        else:
            scores = list(events[0].payload[0]['scores'])
            pool = events[0].payload[0]['kyotaku']
            offset = 1
        for previous in events[offset:]:
            if previous.kind == 'EV_RIICHI_ACCEPT':
                scores[int(previous.payload[0].name[-1])] -= 1000
                pool += 1
        require(value['scores_before']==scores and value['kyotaku']==pool,
                'SETTLEMENT_MISMATCH','settlement must continue the known IR balance')
        if ends:
            require(value['payments']==UNKNOWN or all(p['payment_kind']==Atom('PAYMENT_KYOTAKU') and
                    p['payer'] is None for p in value['payments']),
                    'INVALID_REFERENCE','terminal adjustment is only a pool payment')
        else:
            prior = prior_results[0]
            details = prior.payload[2] if prior.kind=='EV_ABORT_END' else prior.payload[0]
            if prior.kind != 'EV_WIN':
                require(details['score_delta']==UNKNOWN or details['score_delta']==value['score_delta'],
                        'SETTLEMENT_MISMATCH','result delta must match its settlement')
            if prior.kind=='EV_NAGASHI' and details['payments']!=UNKNOWN:
                require(details['payments']==value['payments'],'SETTLEMENT_MISMATCH','nagashi payments must match settlement')
    elif kind == 'EV_ROUND_END':
        require(len(prior_results)==1 and len(settlements)==1 and not ends,
                'INVALID_REFERENCE','round end requires one settled result')
        check_next_state_context(event.payload[0],events)
    else:
        require(len(ends)==1 and settlements and ends[0].payload[0]['game_over'] is True,
                'INVALID_REFERENCE','game end requires a terminal round boundary')
        check_next_state_context(ends[0].payload[0],events)
        value = event.payload[0]
        require(value['final_scores']==settlements[-1].payload[0]['scores_after'],
                'SETTLEMENT_MISMATCH','final scores must match the latest settlement')
        reason = ends[0].payload[0]['reason']
        require(reason==UNKNOWN or value['reason']==UNKNOWN or reason==value['reason'],
                'INVALID_REFERENCE','final reason contradicts next state')


def check_ir_event(event, events, windows):
    require(type(event) is Event and event.kind in EVENT_FIELDS, 'TYPE_MISMATCH', 'IR Event')
    natural(event.event_id, 'event_id')
    require(event.event_id == len(events), 'INVALID_REFERENCE', 'IR append ID')
    require(type(event.payload) is tuple and len(event.payload) == len(EVENT_FIELDS[event.kind]),
            'TYPE_MISMATCH', 'IR event arity')
    validators = {'EV_ROUND_START': check_round_start, 'EV_SETTLEMENT': check_settlement,
                  'EV_DRAW_END': check_draw_result, 'EV_ROUND_END': check_next_state,
                  'EV_GAME_END': check_final_result,'EV_NAGASHI':check_nagashi_result,
                  'EV_REVEAL':check_reveal}
    if event.kind in validators:
        validators[event.kind](event.payload[0])
        if event.kind == 'EV_NAGASHI':
            require(events and events[0].kind == 'EV_ROUND_START',
                    'INVALID_REFERENCE', 'nagashi requires round start context')
            dealer = int(events[0].payload[0]['dealer'].name[-1])
            winners = event.payload[0]['winners']
            require(winners == sorted(winners, key=lambda p:(int(p.name[-1])-dealer-1)%4),
                    'NONCANONICAL', 'nagashi dealer-relative order')
    elif event.kind == 'EV_RESPONSE_OPEN':
        check_window(event.payload[0], events)
    elif event.kind == 'EV_RESPONSE_CLOSE':
        check_close(*event.payload, windows)
    elif event.kind == 'EV_WIN':
        check_win_group(event.payload[0], events, windows)
    elif event.kind=='EV_ABORT_END':
        reason,actor,result=event.payload
        check_abort_result(result)
        require(reason==result['reason'] and actor==result['actor'],'TYPE_MISMATCH','abort event/result')
    else:
        payload = dict(zip(EVENT_FIELDS[event.kind], event.payload))
        for name in ('actor', 'target'):
            if name in payload:
                seat(payload[name], name)
        for name in ('tile', 'base_tile'):
            if name in payload:
                tile(payload[name], name)
        for name in ('meld_id', 'trigger_id', 'window_id'):
            if name in payload:
                natural(payload[name], name)
        if 'consumed' in payload:
            bag(payload['consumed'], 'consumed')
            expected = 4 if event.kind == 'EV_ANKAN_ATTEMPT' else 3 if event.kind == 'EV_DAIMINKAN' else 2
            require(len(payload['consumed']) == expected, 'TYPE_MISMATCH', 'consumed arity')
        if event.kind == 'EV_DRAW':
            enum(payload['draw_kind'], ('DRAW_LIVE', 'DRAW_RINSHAN'), 'draw_kind')
        if event.kind == 'EV_DISCARD':
            enum(payload['discard_kind'], ('TEDASHI', 'TSUMOGIRI'), 'discard_kind')
            boolean(payload['riichi_tile'], 'riichi_tile')
    check_ir_lifecycle(event,events,windows)
    check_result_lifecycle(event,events)


def unknown_next_state():
    return {k: UNKNOWN for k in ('game_over', 'next_dealer', 'next_round',
                                 'next_honba', 'next_kyotaku', 'reason')}


def to_json(value):
    """Use the protocol JSON enum/unknown representation for side facts."""
    if type(value) is Atom:
        return {'$': value.name} if value.name in ('UNKNOWN','HIDDEN') else {'enum': value.name}
    if type(value) is Decimal:
        return {'decimal': str(value)}
    if type(value) in (tuple, list):
        return [to_json(v) for v in value]
    if type(value) is dict:
        return {k: to_json(v) for k,v in value.items()}
    return value


def field_origins(value, default, *, path=''):
    """Small provenance ledger; values stay in their canonical object only."""
    if type(value) is dict and value:
        return {p: origin for k,v in value.items() for p,origin in
                field_origins(v,default,path=f'{path}.{k}' if path else k).items()}
    if type(value) in (list,tuple) and value:
        return {p: origin for i,v in enumerate(value) for p,origin in
                field_origins(v,default,path=f'{path}.{i}' if path else str(i)).items()}
    return {path:'unknown' if value in (UNKNOWN,HIDDEN) else default}
