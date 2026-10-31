"""MJAI clean text/3: tabular ASCII records and exact sequence edits.

This is a storage codec, not a Mahjong state machine. All four view arrays and
source facts round trip exactly; no business decisions are made during decode.
"""
from bisect import bisect_right
from collections import Counter
from urllib.parse import quote, unquote

from .common import CleaningError, require

MAX_VECTOR = 1024 * 1024


def numbers(values):
    return ' '.join(map(str, values))


def integers(text):
    return [int(v) for v in text.split()]


def deltas(values):
    return numbers(v - p for v, p in zip(values, [0, *values]))


def undeltas(text):
    values, previous = [], 0
    for delta in integers(text):
        previous += delta
        values.append(previous)
    return values


def edit(base, value):
    """Choose the shortest textual exact edit (no tile or event semantics)."""
    if value == base:
        return '='
    candidates = ['f ' + numbers(value)]
    if len(base) == len(value):
        changes = [x for i, (a, b) in enumerate(zip(base, value)) if a != b for x in (i, b)]
        candidates.append('s ' + numbers(changes))
    prefix = 0
    while prefix < min(len(base), len(value)) and base[prefix] == value[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(base), len(value)) - prefix and base[-suffix-1] == value[-suffix-1]:
        suffix += 1
    end = len(value) - suffix
    candidates.append('p ' + numbers([prefix, len(base)-prefix-suffix, *value[prefix:end]]))
    return min(candidates, key=len)


def apply(base, text):
    if text == '=':
        return list(base)
    mode, _, payload = text.partition(' ')
    values = integers(payload)
    if mode == 'f':
        result = values
    elif mode == 's':
        require(len(values)%2 == 0, 'STORAGE_CORRUPT', 'sparse edit arity')
        result = list(base)
        positions = values[::2]
        require(positions == sorted(set(positions)) and all(0 <= i < len(base) for i in positions),
                'STORAGE_CORRUPT', 'sparse edit positions')
        for i, value in zip(positions, values[1::2]):
            result[i] = value
    else:
        require(mode == 'p' and len(values) >= 2, 'STORAGE_CORRUPT', 'sequence edit')
        start, count, *insert = values
        require(0 <= start <= len(base) and 0 <= count <= len(base)-start,
                'STORAGE_CORRUPT', 'splice edit extent')
        result = [*base[:start], *insert, *base[start+count:]]
    require(len(result) <= MAX_VECTOR, 'STORAGE_CORRUPT', 'expanded vector limit')
    return result


def value_text(value):
    """Rare structured notes: typed, counted words; strings are percent escaped."""
    if value is None: return 'n'
    if type(value) is bool: return 't' if value else 'b'
    if type(value) is int: return 'i' + str(value)
    if type(value) is float: return 'r' + value.hex()
    if type(value) is str: return 's' + quote(value, safe='')
    if type(value) is list:
        return ' '.join(['l'+str(len(value)), *(value_text(v) for v in value)])
    require(type(value) is dict and all(type(k) is str for k in value),
            'TYPE_MISMATCH', 'text note type')
    return ' '.join(['m'+str(len(value)), *(word for k in sorted(value)
                     for word in (value_text(k), value_text(value[k])))])


def read_value(text):
    words = iter(text.split())
    def read(depth=0):
        require(depth <= 32, 'STORAGE_CORRUPT', 'text note nesting')
        word = next(words)
        tag, tail = word[0], word[1:]
        if word == 'n': return None
        if word in ('t', 'b'): return word == 't'
        if tag == 'i': return int(tail)
        if tag == 'r': return float.fromhex(tail)
        if tag == 's': return unquote(tail, errors='strict')
        require(tag in ('l','m') and 0 <= int(tail) <= MAX_VECTOR,
                'STORAGE_CORRUPT', 'text note tag/count')
        count = int(tail)
        if tag == 'l': return [read(depth+1) for _ in range(count)]
        result = {}
        for _ in range(count):
            key, item = read(depth+1), read(depth+1)
            require(type(key) is str and key not in result, 'STORAGE_CORRUPT', 'text note key')
            result[key] = item
        return result
    try:
        result = read()
        require(next(words, None) is None, 'STORAGE_CORRUPT', 'trailing text note')
        return result
    except (StopIteration, ValueError, IndexError, TypeError) as error:
        raise CleaningError('STORAGE_CORRUPT', 'invalid text note') from error


def policy_decisions_text(decisions):
    """Compact v3 wire form: fixed decision fields and length-prefixed token lists."""
    require(type(decisions) is list, 'TYPE_MISMATCH', 'policy decision list')
    words = [len(decisions)]
    fields = {'window_id','trigger_id','viewer','response','candidates','target_index','target_event_id'}
    for decision in decisions:
        require(type(decision) is dict and set(decision) == fields,
                'TYPE_MISMATCH', 'policy decision fields')
        window, trigger, viewer = (decision[key] for key in ('window_id','trigger_id','viewer'))
        response, target, event_id = (decision[key] for key in ('response','target_index','target_event_id'))
        candidates = decision['candidates']
        require(all(type(value) is int and value >= 0 for value in (window, trigger, viewer, target, event_id)) and
                window // 4 == trigger and viewer < 4 and type(response) is bool and
                type(candidates) is list and 2 <= len(candidates) <= 512 and
                type(target) is int and target < len(candidates),
                'TYPE_MISMATCH', 'policy decision values')
        words.extend((window, trigger, viewer, int(response), target, event_id, len(candidates)))
        for action in candidates:
            require(type(action) is list and 1 <= len(action) <= 64 and
                    all(type(token) is int and 0 <= token <= 65535 for token in action),
                    'TYPE_MISMATCH', 'policy action token list')
            words.append(len(action))
            words.extend(action)
    return numbers(words)


def read_policy_decisions(text):
    values = integers(text)
    require(values and all(value >= 0 for value in values),
            'STORAGE_CORRUPT', 'policy decision integers')
    count = values[0]
    require(count <= MAX_VECTOR, 'STORAGE_CORRUPT', 'policy decision count')
    cursor = 1

    def take():
        nonlocal cursor
        require(cursor < len(values), 'STORAGE_CORRUPT', 'truncated policy decisions')
        value = values[cursor]
        cursor += 1
        return value

    decisions = []
    for _ in range(count):
        window, trigger, viewer, response, target, event_id, candidate_count = (take() for _ in range(7))
        require(window // 4 == trigger and viewer < 4 and response in (0, 1) and
                2 <= candidate_count <= 512 and target < candidate_count,
                'STORAGE_CORRUPT', 'policy decision header')
        candidates = []
        for _ in range(candidate_count):
            action_length = take()
            require(1 <= action_length <= 64, 'STORAGE_CORRUPT', 'policy action length')
            candidates.append([take() for _ in range(action_length)])
        decisions.append({'window_id':window,'trigger_id':trigger,'viewer':viewer,
            'response':bool(response),'candidates':candidates,'target_index':target,
            'target_event_id':event_id})
    require(cursor == len(values), 'STORAGE_CORRUPT', 'trailing policy decision integers')
    return decisions


def encode_round(record):
    lines = []
    def row(tag, *fields): lines.append('\t'.join([tag, *map(str, fields)]))
    views, event_ids = record['views'], record['event_ids']
    row('R', 3, record['source_key'], record['round_id'], len(views))
    row('I', deltas(event_ids))
    # Share event payloads across views, retaining explicit private differences.
    for i in range(len(event_ids)):
        blocks = []
        for view in views:
            a = view['arrays']; offsets = a['history_offsets']
            blocks.append(a['history'][offsets[i]:offsets[i+1]] if i+1 < len(offsets) else None)
        counts = Counter(tuple(b) for b in blocks if b is not None)
        base = list(counts.most_common(1)[0][0]) if counts else []
        row('E', numbers(base), *('-' if b is None else edit(base,b) for b in blocks))
    windows = {}
    for view in views:
        a = view['arrays']['anchors']
        for i in range(0,len(a),5):
            window, trigger, cutoff = a[i:i+3]
            require(window not in windows or windows[window] == (trigger,cutoff),
                    'INVALID_REFERENCE', 'shared window identity')
            windows[window] = (trigger,cutoff)
    refs = dict(record['facts']['source_refs'])
    require(len(refs)==len(record['facts']['source_refs']), 'INVALID_REFERENCE', 'duplicate source window')
    # Keep even windows with no surviving view, preserving source references.
    order = list(dict.fromkeys([*refs, *windows]))
    indices = {w:i for i,w in enumerate(order)}
    for window in order:
        trigger, cutoff = windows.get(window,(-1,-1))
        source_refs = refs.get(window)
        simple = (type(source_refs) is list and all(type(r) is dict and set(r)=={'raw_line'} and
                  type(r['raw_line']) is int for r in source_refs))
        references = ('L '+deltas([r['raw_line'] for r in source_refs])) if simple else value_text(source_refs)
        row('W',window,trigger,cutoff,references)
    row('O',value_text(record['facts']['outcomes']))
    row('P',policy_decisions_text(record['facts'].get('policy_decisions', [])))
    for view in views:
        a = view['arrays']
        row('V',view['viewer'],view['valid_through_event_id'],value_text(view['invalid_reason']))
        row('T',numbers(a['header']))
        previous = []
        for lo,hi in zip(a['hand_offsets'],a['hand_offsets'][1:]):
            block = a['hand_blocks'][lo:hi]
            row('D',edit(previous,block)); previous = block
        anchors = [a['anchors'][i:i+5] for i in range(0,len(a['anchors']),5)]
        row('A',deltas([indices[x[0]] for x in anchors]),deltas([x[4] for x in anchors]))
    row('Z')
    return ('\n'.join(lines)+'\n').encode('ascii')


def decode_round(raw):
    try:
        return _decode_round(raw)
    except (ValueError, IndexError, KeyError, TypeError, UnicodeError) as error:
        if isinstance(error, CleaningError): raise
        raise CleaningError('STORAGE_CORRUPT', 'invalid text round') from error


def _decode_round(raw):
    rows = iter(line.split('\t') for line in raw.decode('ascii').splitlines())
    def take(tag, fields):
        row = next(rows, [])
        require(len(row)==fields+1 and row[0]==tag, 'STORAGE_CORRUPT', 'expected '+tag+' record')
        return row[1:]
    version, source_key, rid, count = take('R',4)
    require(version in ('2','3') and 0 <= int(count) <= 4, 'STORAGE_CORRUPT', 'text version/views')
    count = int(count)
    event_ids = undeltas(take('I',1)[0])
    histories = [[] for _ in range(count)]; offsets = [[0] for _ in range(count)]
    for event_index in range(len(event_ids)):
        base, *patches = take('E', count+1)
        base = integers(base)
        for i,patch in enumerate(patches):
            if patch != '-':
                require(len(offsets[i]) == event_index+1, 'STORAGE_CORRUPT', 'noncontiguous view history')
                histories[i].extend(apply(base,patch)); offsets[i].append(len(histories[i]))
    windows, refs = [], []
    row = next(rows, [])
    while row and row[0] == 'W':
        require(len(row)==5,'STORAGE_CORRUPT','window arity')
        _,window,trigger,cutoff,ref = row
        windows.append((int(window),int(trigger),int(cutoff)))
        source_refs = ([dict(raw_line=n) for n in undeltas(ref[2:])] if ref.startswith('L ') else read_value(ref))
        if source_refs is not None: refs.append([int(window),source_refs])
        row = next(rows, [])
    require(len({w[0] for w in windows})==len(windows), 'STORAGE_CORRUPT', 'duplicate window')
    require(len(row)==2 and row[0]=='O', 'STORAGE_CORRUPT','outcome record')
    outcomes = read_value(row[1])
    policy_decisions = read_policy_decisions(take('P',1)[0]) if version == '3' else []
    views = []
    for i in range(count):
        viewer, valid, reason = take('V',3)
        header = integers(take('T',1)[0])
        hands, hand_offsets, previous = [], [0], []
        row = next(rows, [])
        while row and row[0] == 'D':
            require(len(row)==2,'STORAGE_CORRUPT','hand arity')
            previous = apply(previous,row[1]); hands.extend(previous); hand_offsets.append(len(hands))
            row = next(rows, [])
        require(len(row)==3 and row[0]=='A', 'STORAGE_CORRUPT','anchor record')
        window_refs, blocks = undeltas(row[1]), undeltas(row[2])
        require(len(window_refs)==len(blocks), 'STORAGE_CORRUPT','anchor columns')
        anchors, first_use = [], {}
        for w, block in zip(window_refs,blocks):
            require(0 <= w < len(windows) and 0 <= block < len(hand_offsets)-1,
                    'STORAGE_CORRUPT','anchor reference')
            window,trigger,cutoff = windows[w]
            anchors.extend((window,trigger,cutoff,bisect_right(event_ids,cutoff),block))
            first_use.setdefault(block,cutoff)
        require(set(first_use)==set(range(len(hand_offsets)-1)), 'STORAGE_CORRUPT','unused hand')
        arrays = dict(header=header,history=histories[i],history_offsets=offsets[i],hand_blocks=hands,
                      hand_offsets=hand_offsets,state_event_ids=[first_use[b] for b in range(len(first_use))],anchors=anchors)
        views.append(dict(viewer=int(viewer),valid_through_event_id=int(valid),invalid_reason=read_value(reason),arrays=arrays))
    take('Z',0)
    require(next(rows,None) is None,'STORAGE_CORRUPT','trailing round text')
    facts=dict(source_refs=refs,outcomes=outcomes)
    if policy_decisions:
        facts['policy_decisions']=policy_decisions
    return dict(source_key=source_key,round_id=int(rid),event_ids=event_ids,views=views,facts=facts)
