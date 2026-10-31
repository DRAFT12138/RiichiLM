"""Synchronous full-match collection, all intentions recorded before adjudication."""
from dataclasses import asdict
import json
import time
import torch
from mahjong.contracts import MahjongError, derive_seed
from mahjong.inference.policy import DecodeOptions
from mahjong.inference.actor_loading import new_actor_bridge
from mahjong.rl.storage import EpisodeWriter
from mahjong.policy.execution import policy_precision, sampling_definition


def seed(root, domain, episode, seat=None):
    return derive_seed(root, f'mahjong-rl/{domain}/v1/{episode}/{seat}')


def collect(run_dir, actors, *, root_seed, episode, learner, limits, header, cancel=lambda: False,
            strategy='categorical', environment_seed=None, sampling_episode=None):
    started = time.monotonic()
    deadline = started + limits['max_seconds']
    env_seed = seed(root_seed, 'environment', episode) if environment_seed is None else environment_seed
    sampling_episode = episode if sampling_episode is None else sampling_episode
    precision = policy_precision(actors[learner].model)
    if any(policy_precision(a.model)!=precision for a in actors):
        raise MahjongError('CONTRACT_MISMATCH', 'mixed policy precision at one RL table')
    generators = [torch.Generator().manual_seed(seed(root_seed, 'sampling', sampling_episode, p)) for p in range(4)]
    writer = EpisodeWriter(run_dir / 'trajectories', episode, limits, dict(**header,
        episode=episode, learner=learner, actors=[a.source['actor_ref'] for a in actors],
        sampling=sampling_definition(precision, strategy),
        environment_seed=env_seed, sampling_seeds=[g.initial_seed() for g in generators]))
    counts = dict(decisions=0, action_tokens=0, transitions=0, windows=0, hands=0, wins=0, deal_ins=0)
    bridge = new_actor_bridge(actors)

    def check():
        if cancel():
            raise MahjongError('CANCELLED', 'RL collection cancelled')
        if time.monotonic() >= deadline:
            raise MahjongError('DEADLINE_EXCEEDED', 'RL match time budget exceeded')
        if counts['decisions'] > limits['max_decisions'] or counts['transitions'] > limits['max_transitions']:
            raise MahjongError('DATA_READ_FAILED', 'RL match progression budget exceeded')

    def evidence(item, window_key):
        for transition in item['transitions']:
            wins = [json.loads(f) for f in transition.facts if json.loads(f).get('type') == 'hora']
            counts['wins'] += sum(w['actor'] == learner for w in wins)
            counts['deal_ins'] += int(any(w['target'] == learner and w['actor'] != learner for w in wins))
            writer.write(dict(type='transition', key=f't{counts["transitions"]}', window_key=window_key,
                              automatic_passes=list(item.get('automatic_passes', ())), **asdict(transition)))
            counts['transitions'] += 1
        before, after = item['before'], item['after']
        if after['status'] == 'round_ended': counts['hands'] += 1
        if before is None or after['status'] in ('round_ended', 'game_ended') or before['round_id'] != after['round_id']:
            writer.write(dict(type='boundary', key=f'b{counts["transitions"]}', status=after['status'],
                round_id=after['round_id'], scores=list(after['scores']), kyotaku=after['kyotaku'], ranks=item['ranks']))
        check()

    try:
        item = bridge.reset_with_evidence(seed=env_seed)
        evidence(item, None)
        while item['result'].status != 'game_ended':
            check()
            if item['result'].status == 'round_ended':
                item = bridge.next_round_with_evidence(); evidence(item, None)
                continue
            window = bridge.training_window()
            views = window['views']
            if not views:
                raise MahjongError('EVENT_STREAM_INCOMPLETE', 'empty active window')
            decisions = {}
            # Both visible observations and Oracle features were frozen before any generation.
            for view in views:
                check()
                p = view.key.viewer
                actor = actors[p]
                request = actor.prepare(view)
                generated = actor.generate(request, options=DecodeOptions(strategy), generator=generators[p],
                                           cancel=cancel, deadline_monotonic=deadline)
                decision = bridge.bind_result(view, generated)
                decisions[p] = decision
                writer.write(dict(type='decision', key=f'd{counts["decisions"]}',
                    decision_key=asdict(view.key), round_id=window['round_id'], viewer=p,
                    prefix=list(request.prefix_ids), suffixes=[list(a) for a in request.legal_suffixes],
                    chosen=list(generated.suffix_ids), token_logprobs=list(generated.token_logprobs),
                    old_logprob=generated.behavior_action_logprob, sampling=generated.sampling,
                    actor_ref=generated.actor_ref, forced=len(request.legal_suffixes)==1,
                    origin='model', oracle=list(window['oracle'][p]), oracle_key=asdict(view.key)))
                counts['decisions'] += 1; counts['action_tokens'] += len(generated.suffix_ids)
            from riichienv import Phase
            if bridge._native.phase == Phase.WaitResponse:
                item = bridge.apply_responses_with_evidence(views[0].key.window, decisions)
            else:
                if len(decisions) != 1:
                    raise MahjongError('CONTRACT_MISMATCH', 'multiple self decisions')
                item = bridge.apply_with_evidence(next(iter(decisions.values())))
            counts['windows'] += 1
            evidence(item, asdict(views[0].key.window))
        ref = writer.publish(item['after']['scores'], item['ranks'])
        return dict(status='complete', ref=ref, **counts, seconds=time.monotonic()-started,
                    final_scores=list(item['after']['scores']), ranks=item['ranks'], learner=learner)
    except Exception as exc:
        try:
            writer.write(dict(type='failure', key='failure', reason=getattr(exc, 'code', type(exc).__name__),
                              message=str(exc), details=getattr(exc, 'details', {})))
        except Exception:
            pass  # Preserve the already-written bounded prefix even when the byte budget is exhausted.
        writer.abort()
        return dict(status='cancelled' if getattr(exc, 'code', None)=='CANCELLED' else 'failed',
                    reason=getattr(exc, 'code', type(exc).__name__), message=str(exc),
                    **counts, seconds=time.monotonic()-started, learner=learner)
