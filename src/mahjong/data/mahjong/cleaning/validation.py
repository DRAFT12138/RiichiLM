"""Business validation lives here, never in the lexical Tokenizer."""
from mahjong.envs.ir_primitives import TILES, natural
from mahjong.envs.projection import SEATS
from mahjong.tokenizer import Atom
from mahjong.tokenizer.codec import EVENT_FIELDS
from .common import require
from .models import VISIBLE


def enum(value, names, label):
    require(isinstance(value, Atom) and value.name in names, 'TYPE_MISMATCH', label)


def tile(value, label='tile', *, hidden=False):
    enum(value, (*TILES, 'TILE_HIDDEN') if hidden else TILES, label)


def base(value):
    return value.name.removesuffix('r')


def tile_inventory(tiles, *, aka_counts=None):
    for value in tiles:
        tile(value)
    # Writer-only business validation. Keep rule imports out of the reader's
    # import path; share the configured capacity check with the environment.
    from mahjong.envs.offline_mjai import ReplayError
    from mahjong.envs.tile_stock import TileStock
    from .common import CleaningError
    try:
        TileStock(aka_counts).check([t.name.removeprefix('TILE_') for t in tiles])
    except ReplayError as error:
        raise CleaningError(error.code,error.reason) from error


def check_ir(prepared):
    """Validate the complete reference context before filtering management IDs."""
    natural(prepared.round_id, 'round_id')
    ledger=prepared.ledger
    previous_dora_commit=-1
    for record in ledger:
        natural(record.get('event_id'), 'IR event_id')
    require(len(ledger)>=2 and [r.get('event_id') for r in ledger]==list(range(len(ledger))) and
            ledger[0].get('kind')=='EV_ROUND_START' and ledger[1].get('kind')=='EV_DORA_REVEAL',
            'INVALID_REFERENCE','IR IDs/initial events')
    for record in ledger:
        require(record.get('kind') in EVENT_FIELDS,'TYPE_MISMATCH','IR kind')
        refs=record.get('source_refs')
        require(type(refs) is list and refs and all(type(r) is dict and
                type(r.get('raw_line')) is int and r['raw_line']>=1 for r in refs),
                'INVALID_REFERENCE','source line provenance')
        require(record.get('provenance') in ('recorded','derived','unknown'),
                'TYPE_MISMATCH','IR provenance')
        if record['kind'] in VISIBLE-{'EV_DORA_REVEAL'}:
            require(type(record.get('actor')) is int and 0<=record['actor']<4,
                    'TYPE_MISMATCH','IR actor')
        if record['kind']=='EV_DORA_REVEAL' and ('dora_meld_id' in record or 'dora_commit_event' in record):
            commit, meld_id=record.get('dora_commit_event'),record.get('dora_meld_id')
            natural(commit,'dora commit reference')
            natural(meld_id,'dora meld reference')
            require(previous_dora_commit<commit<record['event_id'] and ledger[commit]['kind']=='EV_KAN_COMMIT' and
                    ledger[commit].get('meld_id')==meld_id,'INVALID_REFERENCE','dora committed meld/order')
            previous_dora_commit=commit
    expected=[r['event_id'] for r in ledger if r['kind'] in VISIBLE]
    require(prepared.event_ids==expected,'INVALID_REFERENCE','shared visible IDs')
    windows=set()
    for window in prepared.windows:
        trigger,cutoff=window['trigger_id'],window['cutoff_event_id']
        natural(window['window_id'],'window_id')
        natural(trigger,'trigger_id')
        natural(cutoff,'cutoff')
        require(trigger<=cutoff<len(ledger),'INVALID_REFERENCE','window boundary')
        kind=ledger[trigger]['kind']
        slot={'EV_DRAW':0,'EV_CHI':1,'EV_PON':1,'EV_DISCARD':2,
              'EV_ANKAN_ATTEMPT':3,'EV_KAKAN_ATTEMPT':3}.get(kind)
        require(slot is not None and window['window_id']==4*trigger+slot and window['window_id'] not in windows,
                'INVALID_REFERENCE','window formula/trigger kind')
        windows.add(window['window_id'])
        actor=ledger[trigger]['actor']
        viewers=[actor] if slot in (0,1) else [(actor+i)%4 for i in range(1,4)]
        require(window['viewers']==viewers,'VISIBILITY_VIOLATION','window participants')
        require((cutoff==trigger if slot in (0,1) else
                 cutoff==trigger+1 and ledger[cutoff]['kind']=='EV_RESPONSE_OPEN'),
                'INVALID_REFERENCE','window cutoff phase')
    gaps=prepared.facts.get('state_gaps',[])
    require(type(gaps) is list,'TYPE_MISMATCH','state gaps')
    by_window={w['window_id']:w for w in prepared.windows}
    seen=set()
    for gap in gaps:
        require(type(gap) is dict and gap.keys()=={'code','stage','reason','round_id','raw_line','event_id','window_id','viewer'},
                'TYPE_MISMATCH','state gap fields')
        for name in ('round_id','raw_line','event_id','window_id','viewer'):
            natural(gap[name],'state gap '+name)
        require(gap['code']=='CURRENT_STATE_UNKNOWN' and gap['stage']=='projection' and
                type(gap['reason']) is str and bool(gap['reason'].strip()),'TYPE_MISMATCH','state gap diagnostic')
        window=by_window.get(gap['window_id'])
        key=(gap['window_id'],gap['viewer'])
        require(gap['round_id']==prepared.round_id and window is not None and
                gap['viewer'] in window['viewers'] and gap['event_id']==window['cutoff_event_id'] and
                gap['raw_line'] in [r['raw_line'] for r in window['source_refs']] and key not in seen,
                'INVALID_REFERENCE','state gap window/seat/source')
        seen.add(key)
    require(not seen or not any((a.window_id,v.viewer) in seen for v in prepared.views for a in v.anchors),
            'INVALID_REFERENCE','unconfirmed state has an anchor')
    check_furiten_facts(prepared,by_window)


def check_furiten_facts(prepared,windows):
    """Check private knowledge references; it never becomes model history."""
    for name in ('furiten_windows','furiten_transitions'):
        records=prepared.facts.get(name,[])
        require(type(records) is list,'TYPE_MISMATCH',name)
        seen=set()
        last=-1
        for record in records:
            transition=name=='furiten_transitions'
            keys={'event_id','window_id','trigger_id','states'}|({'source_refs'} if transition else set())
            require(type(record) is dict and record.keys()==keys,'TYPE_MISMATCH','furiten record fields')
            for field in ('event_id','window_id','trigger_id'):natural(record[field],'furiten '+field)
            window=windows.get(record['window_id'])
            event_id=record['event_id']
            require(window is not None and record['trigger_id']==window['trigger_id'] and
                    record['window_id'] not in seen and last<event_id<len(prepared.ledger),
                    'INVALID_REFERENCE','furiten window/event')
            seen.add(record['window_id'])
            last=event_id
            if transition:
                ledger=prepared.ledger[event_id]
                require(ledger['kind']=='EV_RESPONSE_CLOSE' and ledger.get('window_id')==record['window_id'] and
                        event_id>window['cutoff_event_id'] and record['source_refs']==[r['raw_line'] for r in ledger['source_refs']],
                        'INVALID_REFERENCE','furiten visible-after close/source')
            else:
                require(event_id==window['cutoff_event_id'],'INVALID_REFERENCE','furiten open snapshot cutoff')
            states=record['states']
            require(type(states) is list and all(type(s) is dict for s in states) and
                    [s.get('actor') for s in states]==window['viewers'],
                    'INVALID_REFERENCE','furiten actors')
            response=record['window_id']%4>=2
            for state in states:
                keys={'actor','discard','temp','riichi'}|({'ron_available'} if response else set())
                if transition:keys|={'knowledge','missed'}
                require(state.keys()==keys and type(state['actor']) is int,'TYPE_MISMATCH','furiten state fields')
                for field in keys-{'actor','knowledge'}:
                    require(type(state[field]) is bool or state[field]=={'$':'UNKNOWN'},'TYPE_MISMATCH','furiten tri-state')
                if transition:
                    require(state['knowledge'] in ('recorded_ron','public_triple_ron','recorded_call','complete_collection','unobserved'),
                            'TYPE_MISMATCH','furiten response evidence')


def check_header(header):
    expected = {'mode', 'round_wind', 'round_number', 'honba', 'kyotaku',
                'dealer', 'scores', 'seat_winds'}
    require(isinstance(header, dict), 'TYPE_MISMATCH', 'header')
    require(expected <= header.keys(), 'MISSING_FIELD', 'header')
    require(header.keys() == expected, 'UNKNOWN_FIELD', 'header')
    enum(header['mode'], ('MODE_HANCHAN', 'MODE_TONPUU', 'MODE_CUSTOM'), 'mode')
    enum(header['round_wind'], ('ROUND_E', 'ROUND_S', 'ROUND_W', 'ROUND_N'), 'round_wind')
    natural(header['round_number'], 'round_number')
    require(1 <= header['round_number'] <= 4, 'NONINITIAL_VALUE', 'round_number')
    for name in ('honba', 'kyotaku'):
        natural(header[name], name)
    enum(header['dealer'], SEATS, 'dealer')
    require(type(header['scores']) is list and len(header['scores']) == 4 and
            all(type(s) is int for s in header['scores']), 'TYPE_MISMATCH', 'scores')
    require(type(header['seat_winds']) is list and len(header['seat_winds']) == 4,
            'TYPE_MISMATCH', 'seat_winds')
    require(all(isinstance(t, Atom) for t in header['seat_winds']) and
            {t.name for t in header['seat_winds']} == {'TILE_E', 'TILE_S', 'TILE_W', 'TILE_N'},
            'NONINITIAL_VALUE', 'seat_winds permutation')
    require(header['seat_winds'][SEATS.index(header['dealer'].name)] == Atom('TILE_E'),
            'NONINITIAL_VALUE', 'dealer seat wind')


def check_event(event, *, viewer):
    require(event.kind in VISIBLE, 'VISIBILITY_VIOLATION', event.kind)
    natural(event.event_id, 'event_id')
    fields = EVENT_FIELDS[event.kind]
    require(type(event.payload) is tuple and len(event.payload) == len(fields),
            'TYPE_MISMATCH', f'{event.kind} payload arity')
    payload = dict(zip(fields, event.payload))
    for name, value in payload.items():
        if name in ('actor', 'target'):
            enum(value, SEATS, name)
        elif name in ('event_id', 'meld_id', 'trigger_id', 'window_id'):
            natural(value, name)
        elif name in ('tile', 'base_tile'):
            tile(value, name, hidden=event.kind == 'EV_DRAW')
        elif name == 'consumed':
            expected = 4 if event.kind == 'EV_ANKAN_ATTEMPT' else (3 if event.kind == 'EV_DAIMINKAN' else 2)
            require(type(value) is list and len(value) == expected, 'TYPE_MISMATCH', name)
            for t in value:
                tile(t, hidden=event.kind == 'EV_ANKAN_ATTEMPT')
        elif name == 'riichi_tile':
            require(type(value) is bool, 'TYPE_MISMATCH', name)
        elif name == 'draw_kind':
            enum(value, ('DRAW_LIVE', 'DRAW_RINSHAN'), name)
        elif name == 'discard_kind':
            enum(value, ('TEDASHI', 'TSUMOGIRI'), name)
    if event.kind == 'EV_DRAW':
        own = payload['actor'] == Atom('SELF')
        require((payload['tile'] != Atom('TILE_HIDDEN')) == own,
                'VISIBILITY_VIOLATION', 'draw visibility')
    if event.kind == 'EV_ANKAN_ATTEMPT':
        consumed = payload['consumed']
        hidden = [t == Atom('TILE_HIDDEN') for t in consumed]
        require(all(hidden) or not any(hidden), 'VISIBILITY_VIOLATION', 'partial ankan reveal')
        require(payload['actor'] != Atom('SELF') or not any(hidden),
                'VISIBILITY_VIOLATION', 'own ankan hidden')
        require(payload['base_tile'].name == base(payload['base_tile']),
                'TYPE_MISMATCH', 'red base_tile')
        if not any(hidden):
            require(all(base(t) == base(payload['base_tile']) for t in consumed),
                    'INVALID_INVENTORY', 'ankan composition')
    return payload


def check_event_references(event, payload, *, viewer, ledger):
    record=ledger[event.event_id]
    if 'actor' in payload:
        actor=(viewer+SEATS.index(payload['actor'].name))%4
        require(record.get('actor')==actor,'INVALID_REFERENCE','projected actor/IR')
    if 'meld_id' in payload:
        require(record.get('meld_id')==payload['meld_id'],'INVALID_REFERENCE','event meld_id')
    if 'trigger_id' in payload:
        trigger=payload['trigger_id']
        require(trigger<event.event_id and trigger in ledger and ledger[trigger]['kind']=='EV_DISCARD' and
                ledger[trigger].get('actor')==(viewer+SEATS.index(payload['target'].name))%4 and
                record.get('trigger_id')==trigger and record.get('called_tile')==payload['tile'].name and
                ledger[trigger].get('tile')==payload['tile'].name,
                'INVALID_REFERENCE','call trigger/payload')
        require(payload['actor']!=payload['target'],'INVALID_REFERENCE','self call')
    if event.kind=='EV_CHI':
        require(SEATS.index(payload['actor'].name)==(SEATS.index(payload['target'].name)+1)%4,
                'INVALID_REFERENCE','chi direction')
    if event.kind in ('EV_ANKAN_ATTEMPT','EV_KAKAN_ATTEMPT'):
        require(payload['window_id']==4*event.event_id+3,'INVALID_REFERENCE','kan window')
    if event.kind=='EV_KAN_COMMIT':
        require(any(i<event.event_id and r['kind'] in ('EV_DAIMINKAN','EV_ANKAN_ATTEMPT','EV_KAKAN_ATTEMPT') and
                    r.get('actor')==record['actor'] and r.get('meld_id')==payload['meld_id']
                    for i,r in ledger.items()),'INVALID_REFERENCE','kan commit without attempt')


def check_hand(state, *, viewer, cutoff, ledger, aka_counts=None):
    require(type(state.hands) is list and type(state.melds) is list,
            'TYPE_MISMATCH', 'hand state')
    entities = list(state.hands)
    fields = {'name', 'actor', 'meld_id', 'tiles', 'base_tile', 'from',
              'called_tile', 'trigger_id', 'created_event', 'updated_event'}
    ids = set()
    for meld in state.melds:
        require(type(meld) is dict and meld.keys() == fields, 'MISSING_FIELD', 'meld fields')
        enum(meld['name'], ('MELD_CHI', 'MELD_PON', 'MELD_DAIMINKAN', 'MELD_ANKAN', 'MELD_KAKAN'), 'meld.name')
        require(meld['actor'] == Atom('SELF'), 'VISIBILITY_VIOLATION', 'foreign current meld')
        natural(meld['meld_id'], 'meld_id')
        require(meld['meld_id'] not in ids, 'INVALID_REFERENCE', 'duplicate meld_id')
        ids.add(meld['meld_id'])
        name = meld['name'].name
        expected = 3 if name in ('MELD_CHI', 'MELD_PON') else 4
        require(type(meld['tiles']) is list and len(meld['tiles']) == expected,
                'INVALID_INVENTORY', 'meld size')
        entities.extend(meld['tiles'])
        tile(meld['base_tile'])
        require(meld['base_tile'].name == base(meld['base_tile']), 'INVALID_INVENTORY', 'red base')
        composition = [base(t) for t in meld['tiles']]
        if name == 'MELD_CHI':
            try:
                values = sorted((int(t[5]), t[6]) for t in composition)
            except (ValueError, IndexError):
                values = []
            require(len(values) == 3 and values[0][1] in 'mps' and
                    len({v[1] for v in values}) == 1 and
                    [v[0] for v in values] == list(range(values[0][0], values[0][0] + 3)) and
                    meld['base_tile'].name == f'TILE_{values[0][0]}{values[0][1]}',
                    'INVALID_INVENTORY', 'chi composition')
        else:
            require(all(t == meld['base_tile'].name for t in composition), 'INVALID_INVENTORY', 'meld composition')
        created = meld['created_event']
        natural(created, 'created_event')
        require(created <= cutoff and created in ledger, 'INVALID_REFERENCE', 'created event')
        kind = {'MELD_CHI': 'EV_CHI', 'MELD_PON': 'EV_PON', 'MELD_DAIMINKAN': 'EV_DAIMINKAN',
                'MELD_ANKAN': 'EV_KAN_COMMIT', 'MELD_KAKAN': 'EV_PON'}[name]
        record = ledger[created]
        require(record['kind'] == kind and record.get('actor') == viewer and
                record.get('meld_id') == meld['meld_id'], 'INVALID_REFERENCE', 'meld creator mismatch')
        if name == 'MELD_ANKAN':
            require(all(meld[k] is None for k in ('from', 'called_tile', 'trigger_id')),
                    'INVALID_REFERENCE', 'ankan source')
        else:
            enum(meld['from'], SEATS, 'meld.from')
            require(meld['from'] != Atom('SELF'), 'INVALID_REFERENCE', 'self call')
            tile(meld['called_tile'])
            require(meld['called_tile'] in meld['tiles'], 'INVALID_INVENTORY', 'called_tile')
            trigger = meld['trigger_id']
            natural(trigger, 'trigger_id')
            require(trigger < created and trigger in ledger and ledger[trigger]['kind'] == 'EV_DISCARD',
                    'INVALID_REFERENCE', 'call trigger')
            require(ledger[trigger].get('actor') == (viewer + SEATS.index(meld['from'].name)) % 4 and
                    record.get('trigger_id') == trigger, 'INVALID_REFERENCE', 'call source')
            require(record.get('called_tile') == meld['called_tile'].name,
                    'INVALID_REFERENCE', 'called tile provenance')
            require(ledger[trigger].get('tile')==meld['called_tile'].name,
                    'INVALID_REFERENCE','meld called tile/discard mismatch')
        updated = meld['updated_event']
        if name == 'MELD_KAKAN':
            natural(updated, 'updated_event')
            require(created < updated <= cutoff and updated in ledger and
                    ledger[updated]['kind'] == 'EV_KAN_COMMIT' and
                    ledger[updated].get('actor') == viewer and
                    ledger[updated].get('meld_id') == meld['meld_id'],
                    'INVALID_REFERENCE', 'kakan commit')
        else:
            require(updated is None, 'INVALID_REFERENCE', 'unexpected updated_event')
        if name == 'MELD_DAIMINKAN':
            require(any(r['kind'] == 'EV_KAN_COMMIT' and r.get('actor') == viewer and
                        r.get('meld_id') == meld['meld_id'] and created < i <= cutoff
                        for i, r in ledger.items()), 'INVALID_REFERENCE', 'uncommitted daiminkan')
    tile_inventory(entities,aka_counts=aka_counts)
