"""Paired four-seat evaluation; full-match categorical labels for value gates."""
import random
import torch
from mahjong.contracts import MahjongError, derive_seed
from mahjong.policy.scoring import score_action
from mahjong.inference.policy import digest
from mahjong.rl.collector import collect
from mahjong.rl.storage import EpisodeReader
from mahjong.rl.targets import episode_ledger, returns_and_gae
from mahjong.rl.diagnostics import value_statistics
from mahjong.policy.execution import policy_precision, policy_execution, sampling_definition


def evaluate_values(readers, model, *, actor_ref, expected_sampling, scope):
    if expected_sampling.get('strategy')!='categorical':
        raise MahjongError('EVALUATION_FAILED', 'greedy returns cannot validate categorical Critic')
    if expected_sampling != sampling_definition(policy_precision(model.actor)):
        raise MahjongError('EVALUATION_FAILED', 'value evaluation precision/sampling mismatch')
    values, returns, groups = [], [], []
    device = next(model.parameters()).device
    for reader in readers:
        header = reader.header
        if header['actors'][header['learner']] != actor_ref or header['sampling'] != expected_sampling:
            raise MahjongError('EVALUATION_FAILED', 'value evaluation policy provenance mismatch')
        groups.append(dict(episode=header['episode'], actors=header['actors'], sampling=header['sampling']))
        keys, rewards, _ = episode_ledger(reader)
        mc, _ = returns_and_gae(rewards, [0.]*len(keys))
        for key, ret in zip(keys, mc):
            r = reader.read(key)
            with torch.no_grad(), policy_execution(model.actor):
                score = score_action(model.actor, r['prefix'], r['suffixes'], r['chosen'])
                value = model.value(score.prefix_hidden, torch.tensor(r['oracle'], device=device))
            values.append(float(value)); returns.append(ret)
    return dict(scope=scope, actor_ref=actor_ref, sampling=expected_sampling,
                matches=len(readers), groups=groups, **value_statistics(values, returns))


def paired_statistics(groups, *, seed, samples, confidence):
    # One observation per seed's four-seat group, never per decision.
    if not groups:
        return dict(status='not_evaluable', reason='no_complete_pairs', groups=0, mean=None, interval=None)
    mean = sum(groups)/len(groups)
    if len(groups) < 2:
        return dict(status='not_evaluable', reason='insufficient_independent_seed_groups', groups=len(groups), mean=mean, interval=None)
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choices(groups, k=len(groups)))/len(groups) for _ in range(samples))
    tail = (1-confidence)/2
    return dict(status='ok', reason=None, groups=len(groups), mean=mean,
                interval=[bootstrap[int(tail*(samples-1))], bootstrap[int((1-tail)*(samples-1))]])


def paired_evaluate(run_dir, candidate, baseline, opponent, *, model, config, iteration, cancel,
                    seed_plan='monitor', strategy='categorical', event=lambda report: None):
    settings, limits = config['evaluation'], config['collection']
    root_seed = settings[seed_plan+'_seed']
    outcomes, candidate_refs, differences = [], [], []
    total_bytes = 0
    for group in range(settings['seed_groups']):
        paired = []
        for seat in range(4):
            pair = []
            for label, learner in [('baseline', baseline), ('candidate', candidate)]:
                episode = f'eval-{seed_plan}-{iteration}-{group}-{seat}-{label}'
                actors = [opponent]*4; actors[seat] = learner
                outcome = collect(run_dir, actors, root_seed=root_seed, episode=episode, learner=seat,
                    environment_seed=derive_seed(root_seed, f'mahjong-rl/eval-env/v1/{group}'),
                    sampling_episode=f'eval-{seed_plan}-{group}-{seat}',
                    limits=limits, header=dict(initial_scores=learner.source['compatibility']['profile']['initial_scores'],
                        iteration_id=iteration, purpose='evaluation', config_hash=digest(config),
                        reward='match_score_delta_hand_emission/v1'), cancel=cancel, strategy=strategy)
                outcome.update(label=label, seed_group=group, seat=seat)
                outcomes.append(outcome); event(outcome)
                if outcome['status']=='cancelled':
                    raise MahjongError('CANCELLED', 'paired evaluation cancelled')
                if outcome['status']=='complete':
                    total_bytes += outcome['ref']['bytes']
                    if total_bytes > limits['max_round_bytes']:
                        raise MahjongError('EVALUATION_FAILED', 'paired evaluation trajectory byte budget exceeded')
                    initial = learner.source['compatibility']['profile']['initial_scores'][seat]
                    pair.append((outcome['final_scores'][seat]-initial)/10000)
                    if label=='candidate': candidate_refs.append(outcome['ref'])
                else: pair.append(None)
            paired.append(pair[1]-pair[0] if all(x is not None for x in pair) else None)
        if all(x is not None for x in paired): differences.append(sum(paired)/4)
    failures = sum(o['status']!='complete' for o in outcomes)
    report = dict(candidate=candidate.source['actor_ref'], baseline=baseline.source['actor_ref'],
                  opponent=opponent.source['actor_ref'], strategy=strategy, seed_plan=seed_plan,
                  attempts=len(outcomes), failures=failures, outcomes=outcomes,
                  strength_claim=False, trajectory_bytes=total_bytes, statistics=paired_statistics(differences, seed=root_seed,
                    samples=settings['bootstrap_samples'], confidence=settings['confidence']))
    report['accepted'] = failures/len(outcomes) <= settings['max_failed_fraction']
    interval = report['statistics']['interval']
    if interval is not None and interval[1] < -settings['regression_tolerance']:
        report['accepted'] = False
    if model is not None and candidate_refs and strategy=='categorical':
        report['value'] = evaluate_values([EpisodeReader(run_dir, ref, limits) for ref in candidate_refs], model,
            actor_ref=candidate.source['actor_ref'], expected_sampling=sampling_definition(policy_precision(candidate.model)),
            scope='candidate_eval')
    return report
