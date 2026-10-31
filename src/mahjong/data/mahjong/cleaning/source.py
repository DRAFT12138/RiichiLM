"""Strict JSON Lines source validation with independent raw EOF accounting."""
from mahjong.envs.mjai_events import SourceEvent
from dataclasses import dataclass
import hashlib
import json
import math

from .common import CleaningError, require

PAIS = frozenset([f'{n}{s}' for s in 'mps' for n in range(1, 10)] +
                ['E', 'S', 'W', 'N', 'P', 'F', 'C', '5mr', '5pr', '5sr'])
REQUIRED = {
    'start_game': (), 'start_kyoku': ('bakaze', 'dora_marker', 'kyoku', 'honba', 'kyotaku', 'oya', 'scores', 'tehais'),
    'tsumo': ('actor', 'pai'), 'dahai': ('actor', 'pai', 'tsumogiri'),
    'chi': ('actor', 'target', 'pai', 'consumed'), 'pon': ('actor', 'target', 'pai', 'consumed'),
    'daiminkan': ('actor', 'target', 'pai', 'consumed'), 'ankan': ('actor', 'consumed'),
    'kakan': ('actor', 'pai', 'consumed'), 'reach': ('actor',), 'reach_accepted': ('actor',),
    'dora': ('dora_marker',), 'hora': ('actor', 'target', 'deltas'),
    'ryukyoku': ('deltas',), 'end_kyoku': (), 'end_game': (),
}
# Enumerated source facts, retained intact; unknown core keys are never discarded.
OPTIONAL = {
    'start_game': ('names', 'seed', 'id', 'game_id', 'uri', 'log_id', 'kyoku_first', 'aka_flag'),
    'hora': ('pai', 'tehai', 'hora_tehais', 'uradora_markers', 'ura_markers', 'yakus', 'fu', 'han', 'fan', 'pao', 'hora_points', 'scores'),
    'ryukyoku': ('actor', 'reason', 'tehais', 'tenpais', 'scores'),
    'reach_accepted': ('deltas', 'scores'), 'end_game': ('scores',),
}
METADATA = frozenset(('meta',))


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise CleaningError('DUPLICATE_KEY', key, stage='source')
            value[key] = item
        return value

    def constant(value):
        raise CleaningError('TYPE_MISMATCH', f'nonfinite {value}', stage='source')
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            constant(value)
        return parsed
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant,
                           parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CleaningError('INVALID_JSON', str(error), stage='source') from error
    require(type(value) is dict, 'TYPE_MISMATCH', 'event must be an Object')
    return value


def check_source_event(event):
    kind = event.get('type')
    require(type(kind) is str and kind in REQUIRED, 'UNKNOWN_EVENT', repr(kind))
    required = set(REQUIRED[kind]) | {'type'}
    require(required <= event.keys(), 'MISSING_FIELD', f'{kind}: {required - event.keys()}')
    require(event.keys() <= required | set(OPTIONAL.get(kind, ())) | METADATA,
            'UNKNOWN_FIELD', f'{kind}: {event.keys() - required - set(OPTIONAL.get(kind, ())) - METADATA}')
    for key in ('actor', 'target', 'oya'):
        if key in event:
            if key=='actor' and kind=='ryukyoku' and event[key] is None:
                continue
            require(type(event[key]) is int and 0 <= event[key] <= 3, 'TYPE_MISMATCH', key)
    if 'pao' in event:
        require(event['pao'] is None or type(event['pao']) is int and 0<=event['pao']<=3,'TYPE_MISMATCH','pao')
    for key in ('honba', 'kyotaku', 'kyoku'):
        if key in event:
            require(type(event[key]) is int and event[key] >= (1 if key == 'kyoku' else 0), 'TYPE_MISMATCH', key)
    if 'kyoku' in event:
        require(event['kyoku'] <= 4, 'TYPE_MISMATCH', 'kyoku')
    if 'bakaze' in event:
        require(event['bakaze'] in ('E', 'S', 'W', 'N'), 'TYPE_MISMATCH', 'bakaze')
    if 'tsumogiri' in event:
        require(type(event['tsumogiri']) is bool, 'TYPE_MISMATCH', 'tsumogiri')
    for key in ('pai', 'dora_marker'):
        if key in event:
            require(type(event[key]) is str and event[key] in PAIS, 'TYPE_MISMATCH', key)
    for key in ('scores', 'deltas'):
        if key in event:
            require(type(event[key]) is list and len(event[key]) == 4 and
                    all(type(v) is int for v in event[key]), 'TYPE_MISMATCH', key)
    if 'consumed' in event:
        value = event['consumed']
        count = {'chi': 2, 'pon': 2, 'daiminkan': 3, 'ankan': 4, 'kakan': 3}[kind]
        require(type(value) is list and len(value) == count and
                all(type(v) is str and v in PAIS for v in value), 'TYPE_MISMATCH', 'consumed')
    if 'tehais' in event:
        value = event['tehais']
        require(type(value) is list and len(value) == 4 and all(type(h) is list for h in value),
                'TYPE_MISMATCH', 'tehais')
        allowed=PAIS|{'?'} if kind=='ryukyoku' else PAIS
        require(all(type(t) is str and t in allowed for h in value for t in h), 'TYPE_MISMATCH', 'tehais tiles')
        require(all(len(h)<=14 for h in value),'TYPE_MISMATCH','result hand sizes')
        if kind == 'start_kyoku':
            require(all(len(h) == 13 for h in value), 'TYPE_MISMATCH', 'initial hand sizes')
    if 'names' in event:
        require(type(event['names']) is list and len(event['names']) == 4 and
                all(type(n) is str for n in event['names']), 'TYPE_MISMATCH', 'names')
    if 'kyoku_first' in event:
        require(type(event['kyoku_first']) is int and event['kyoku_first']>=0,
                'TYPE_MISMATCH','kyoku_first metadata')
    if 'aka_flag' in event:
        require(type(event['aka_flag']) is bool,'TYPE_MISMATCH','aka_flag metadata')
    for key in ('id','game_id','log_id','uri','reason'):
        if key in event:
            require(type(event[key]) is str,'TYPE_MISMATCH',key)
    if 'seed' in event:
        value=event['seed']
        require(type(value) is int or (type(value) is list and all(type(v) is int for v in value)),
                'TYPE_MISMATCH','seed metadata')
    for key in ('fu','han','fan','hora_points'):
        if key in event:
            require(type(event[key]) is int and event[key]>=0,'TYPE_MISMATCH',key)
    for key in ('ura_markers','uradora_markers','tehai','hora_tehais'):
        if key in event:
            value=event[key]
            require(type(value) is list and all(type(v) is str and v in PAIS for v in value),
                    'TYPE_MISMATCH',key)
            require(len(value)<=(14 if key in ('tehai','hora_tehais') else 5),'TYPE_MISMATCH',key+' length')
    for key in ('tenpais',):
        if key in event:
            value=event[key]
            require(type(value) is list and len(value)==4 and all(type(v) is bool for v in value),
                    'TYPE_MISMATCH',key)
    if 'yakus' in event:
        require(type(event['yakus']) is list,'TYPE_MISMATCH','yakus')
    if 'meta' in event:
        require(type(event['meta']) is dict, 'TYPE_MISMATCH', 'meta')
    return event


class RawReader:
    """Keep the raw stream alive after semantic failure so EOF still gates COMMIT."""
    def __init__(self, path, *, opener=open, max_line_bytes=1024 * 1024):
        self.stream = opener(path, 'rb')
        self.bytes = 0
        self.hasher = hashlib.sha256()
        self.line = 0
        self.eof = False
        self.max_line_bytes = max_line_bytes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def raw_line(self):
        value = self.stream.readline(self.max_line_bytes + 1)
        if not value:
            self.eof = True
            return None
        self.bytes += len(value)
        self.hasher.update(value)
        self.line += 1
        if len(value)>self.max_line_bytes:
            # Consume the whole oversized physical line before allowing a caller
            # to continue. Never parse its remaining chunks as independent events.
            while not value.endswith(b'\n'):
                value=self.stream.readline(self.max_line_bytes+1)
                if not value:
                    self.eof=True
                    break
                self.bytes+=len(value)
                self.hasher.update(value)
            raise CleaningError('RESOURCE_LIMIT','source line too large',stage='source',raw_line=self.line)
        return value

    def read_event(self):
        value=None
        try:
            raw=self.raw_line()
            if raw is None:
                return None
            value=strict_json(raw)
            return SourceEvent(self.line,check_source_event(value))
        except CleaningError as error:
            error.stage,error.raw_line='source',self.line
            error.source_kind=value.get('type') if type(value) is dict and type(value.get('type')) is str else None
            raise

    def events(self):
        while (event := self.read_event()) is not None:
            yield event

    def verify_eof(self, source):
        if not self.eof:
            # Drain bytes without parsing, even after a bad line or an absent profile.
            for chunk in iter(lambda: self.stream.read(1024 * 1024), b''):
                self.bytes += len(chunk)
                self.hasher.update(chunk)
            self.eof = True
        require(self.bytes == source.raw_bytes and self.hasher.hexdigest() == source.raw_sha256,
                'INPUT_CHANGED', f'{source.source_key}: frozen raw bytes/digest changed')


def __getattr__(name):
    """Resolve optional source-processing entry points only when requested."""
    if name in ('check_game_metadata',):
        from . import source_metadata
        value = getattr(source_metadata, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
