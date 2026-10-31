"""Hand-emitted net score ledger, complete MC returns and fixed GAE targets."""
import math
import torch
from mahjong.contracts import MahjongError
from mahjong.policy.scoring import score_action
from mahjong.inference.policy import digest
from mahjong.policy.execution import policy_precision, policy_execution, sampling_definition

TARGET_SCHEMA = 'mahjong-rl-targets/v1'


def critic_digest(model):
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if not key.startswith('actor.'):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def returns_and_gae(rewards, values, lam=1., gamma=1.):
    if gamma != 1. or not 0 <= lam <= 1 or len(rewards) != len(values):
        raise MahjongError('CONTRACT_MISMATCH', 'invalid finite-match GAE parameters')
    mc, adv = [0.] * len(values), [0.] * len(values)
    future_return = future_adv = next_value = 0.
    for i in reversed(range(len(values))):
        future_return += rewards[i]
        future_adv = rewards[i] + next_value - values[i] + lam * future_adv
        mc[i], adv[i] = future_return, future_adv
        next_value = values[i]
    return mc, adv


def episode_ledger(reader):
    learner = reader.header['learner']
    decisions, rewards = [], []
    previous = reader.header['initial_scores'][learner]
    pre_decision = 0.
    for record in reader.records():
        if record['type'] == 'decision' and record['viewer'] == learner:
            decisions.append(record['key']); rewards.append(0.)
        if record['type'] == 'boundary' and record['status'] in ('round_ended', 'game_ended'):
            score = record['scores'][learner]
            reward = (score - previous) / 10000
            if rewards:
                rewards[-1] += reward
            else:
                pre_decision += reward
            previous = score
    net = (reader.trailer['final_scores'][learner] - reader.header['initial_scores'][learner]) / 10000
    if not math.isclose(math.fsum(rewards)+pre_decision, net, abs_tol=1e-10):
        raise MahjongError('CONTRACT_MISMATCH', 'reward ledger differs from final net score')
    return decisions, rewards, pre_decision


def build_targets(readers, model, bc, *, stage, lam, max_bytes, tolerance=2e-5):
    targets = []
    old_actor_ref = None
    device = next(model.parameters()).device
    for reader in readers:
        keys, rewards, pre = episode_ledger(reader)
        values, old, anchors = [], [], []
        expected_actor = reader.header['actors'][reader.header['learner']]
        if old_actor_ref is not None and old_actor_ref != expected_actor:
            raise MahjongError('CONTRACT_MISMATCH', 'mixed behavior Actors in one target bundle')
        old_actor_ref = expected_actor
        precision = policy_precision(model.actor)
        if policy_precision(bc.model)!=precision or reader.header['sampling'] != sampling_definition(precision):
            raise MahjongError('CONTRACT_MISMATCH', 'PPO requires full-match categorical behavior')
        for key in keys:
            record = reader.read(key)
            if (record['actor_ref'] != expected_actor or record['sampling'] != 'categorical'
                    or record['oracle_key'] != record['decision_key']):
                raise MahjongError('CONTRACT_MISMATCH', 'trajectory source/Oracle mismatch')
            with torch.no_grad(), policy_execution(model.actor):
                score = score_action(model.actor, record['prefix'], record['suffixes'], record['chosen'])
                logged = torch.tensor(record['token_logprobs'], device=device)
                if (not torch.allclose(score.token_logprobs, logged, atol=tolerance, rtol=tolerance)
                        or not math.isclose(float(score.action_logprob), record['old_logprob'], abs_tol=tolerance, rel_tol=tolerance)):
                    raise MahjongError('CONTRACT_MISMATCH', 'behavior and old Actor scores differ')
                values.append(float(model.value(score.prefix_hidden, torch.tensor(record['oracle'], device=device))))
                anchors.append(float(score_action(bc.model, record['prefix'], record['suffixes'], record['chosen']).action_logprob))
            old.append(record['old_logprob'])
        mc, adv = returns_and_gae(rewards, values, lam)
        for i, key in enumerate(keys):
            record = reader.read(key)
            target = dict(key=key, trajectory=reader.ref['path'], trajectory_sha=reader.ref['sha256'],
                decision_key=record['decision_key'], old_logprob=old[i], bc_logprob=anchors[i], old_value=values[i],
                A_raw=adv[i], policy_adv=adv[i], value_target=mc[i] if stage=='WARMUP' else adv[i]+values[i],
                mc_return=mc[i], reward=rewards[i], policy_mask=stage=='JOINT' and not record['forced'], value_mask=True)
            targets.append(target)
        if len(targets)*1024 > max_bytes:
            raise MahjongError('DATA_READ_FAILED', 'target array budget exceeded')
    selected = [t['policy_adv'] for t in targets if t['policy_mask']]
    mean = math.fsum(selected)/len(selected) if selected else 0.
    std = math.sqrt(math.fsum((x-mean)**2 for x in selected)/len(selected)) if selected else 0.
    for t in targets:
        t['policy_adv'] = (t['policy_adv']-mean)/std if t['policy_mask'] and std > 1e-8 else 0.
    from mahjong.contracts import canonical_json
    if len(canonical_json(targets).encode()) > max_bytes:
        raise MahjongError('DATA_READ_FAILED', 'serialized target budget exceeded')
    return dict(schema=TARGET_SCHEMA, stage=stage, lam=lam, bc=bc.source['actor_ref'], targets=targets,
                old_actor_ref=old_actor_ref, critic_digest=critic_digest(model),
                binding=digest(targets), advantage_std=std)
