"""Structural validation of non-model text/3 cleaning audit facts."""
from .common import require


def check_cleaning_audits(outcomes):
    def integer(value):
        return type(value) is int and value >= 0

    def lines(values):
        return type(values) is list and bool(values) and all(integer(v) and v > 0 for v in values)

    def scores(values):
        return type(values) is list and len(values) == 4 and all(type(v) is int for v in values)

    require(type(outcomes) is list and all(type(v) is dict for v in outcomes),
            'STORAGE_CORRUPT', 'cleaning outcomes')
    for item in outcomes:
        if item.get('kind') != 'CLEANING_AUDIT':
            continue
        require(set(item) == {'kind', 'revision', 'method', 'proof'}
                and item['revision'] == 'jade-dialect-repair/1'
                and type(item['proof']) is dict,
                'STORAGE_CORRUPT', 'cleaning audit schema')
        proof = item['proof']
        if item['method'] == 'kan':
            valid = (set(proof) == {'raw_line', 'original_type', 'normalized_type', 'actor', 'event_id'}
                     and lines([proof['raw_line']]) and integer(proof['event_id'])
                     and integer(proof['actor']) and proof['actor'] < 4
                     and (proof['original_type'], proof['normalized_type']) in
                     (('ankan', 'kakan'), ('kakan', 'ankan')))
        elif item['method'] == 'cancel_confirmation':
            valid = (set(proof) == {'raw_lines', 'result_lines', 'event_id', 'scores', 'kyotaku'}
                     and lines(proof['raw_lines']) and lines(proof['result_lines'])
                     and integer(proof['event_id']) and scores(proof['scores'])
                     and integer(proof['kyotaku']))
        elif item['method'] == 'recovery':
            valid = (set(proof) == {'failure', 'checkpoint_scores', 'checkpoint_kyotaku',
                                   'result_raw_line', 'next_raw_line', 'score_delta', 'scores_after',
                                   'next_kyotaku', 'evidence', 'boundary_verified'}
                     and type(proof['failure']) is dict and proof['failure'].get('code') == 'MISSING_DORA'
                     and lines([proof['result_raw_line'], proof['next_raw_line']])
                     and proof['next_raw_line'] > proof['result_raw_line']
                     and all(scores(proof[k]) for k in ('checkpoint_scores', 'scores_after', 'score_delta'))
                     and all(integer(proof[k]) for k in ('checkpoint_kyotaku', 'next_kyotaku'))
                     and proof['evidence'] in ('recorded_scoreboard', 'recorded_deltas')
                     and proof['boundary_verified'] is True)
        else:
            valid = False
        require(valid, 'STORAGE_CORRUPT', 'cleaning audit proof')
