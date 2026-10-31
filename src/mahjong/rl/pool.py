"""Bounded on-disk pure-weight pool, with deterministic per-table selection."""
import random
from mahjong.contracts import derive_seed, clone_state, MahjongError


def initial_pool(bc):
    return dict(bc=bc, latest=bc, history=[])


def members(pool):
    return {r['artifact_id']: r for r in [pool['bc'], pool['latest'], *pool['history']]}


def promote(pool, ref, iteration, settings):
    result = clone_state(pool)
    if iteration % settings['history_interval'] == 0:
        old = result['latest']
        if old['artifact_id'] not in {result['bc']['artifact_id'], ref['artifact_id']} and old not in result['history']:
            result['history'].append(old)
    result['latest'] = ref
    result['history'] = [r for r in result['history'] if r['artifact_id'] not in {ref['artifact_id'], result['bc']['artifact_id']}]
    while len(members(result)) > settings['active_capacity']:
        result['history'].pop(0)
    # At most one new file per accepted generation.
    if len(set(members(result))-set(members(pool))) > 1:
        raise MahjongError('CONTRACT_MISMATCH', 'pool update added multiple weights')
    return result


def select(pool, root_seed, episode, weights):
    rng = random.Random(derive_seed(root_seed, f'mahjong-rl/pool/v1/{episode}'))
    x = rng.random()
    requested = 'bc' if x < weights[0] else 'history' if x < weights[0]+weights[1] else 'latest'
    actual = 'bc' if requested=='history' and not pool['history'] else requested
    opponent = rng.choice(pool['history']) if actual=='history' else pool[actual]
    seat = random.Random(derive_seed(root_seed, f'mahjong-rl/seat/v1/{episode}')).randrange(4)
    return dict(learner=seat, opponent=opponent, requested_category=requested, actual_category=actual)
