"""Exclusive native owner, canonical replay projection and atomic action plans.

The native log supplies ordered facts, while accepted kan actions supply the
attempt hook missing from that log. OfflineSession owns project hand/meld and
profile semantics; its omniscient state never crosses the model boundary.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import uuid

from mahjong.envs.projection import project_event, project_hand, project_header, seat
from mahjong.data.mahjong.cleaning.replay import RoundBuilder
from mahjong.envs.mjai_events import SourceEvent
from mahjong.envs.offline_mjai import OfflineSession
from mahjong.envs.rule_profile import offline_options_from_profile
from mahjong.inference.policy import (
    DecisionKey, WindowKey, DecisionView, LegalMove, EnvDecision, digest, legal_digest, fail,
)
from mahjong.tokenizer import Action, Atom, MahjongTokenizer, PolicySequenceInput

PROFILE_DIGEST = '5f711557087a2ab88128f34af8b6ef017532916abae1e7c2fbf7dd2fac8c36a8'
HISTORY_KINDS = frozenset(('EV_DRAW', 'EV_DISCARD', 'EV_CHI', 'EV_PON', 'EV_DAIMINKAN',
    'EV_ANKAN_ATTEMPT', 'EV_KAKAN_ATTEMPT', 'EV_KAN_COMMIT', 'EV_DORA_REVEAL',
    'EV_RIICHI_DECLARE', 'EV_RIICHI_ACCEPT'))


@dataclass(frozen=True)
class ObservedEvent:
    env_instance_id: str
    transition_seq: int
    viewer: int
    event_order: int
    fact_ref: tuple[int, int]
    payload_json: str


@dataclass(frozen=True)
class NativeTransition:
    env_instance_id: str
    transition_seq: int
    before_phase: str
    after_phase: str
    accepted_actions: tuple[tuple[int, str], ...]
    facts: tuple[str, ...]
    observations: tuple[ObservedEvent, ...]


@dataclass(frozen=True)
class ApplyResult:
    env_instance_id: str
    state_revision: int
    status: str


@dataclass(frozen=True)
class SelectedDecision:
    """A rule-validated choice by an explicitly bound non-model controller."""
    key: DecisionKey
    action: Action
    execution_ref: str


class RiichiDecisionBridge:
    _supported_profiles = frozenset((PROFILE_DIGEST,))
    def __init__(self, profile, *, model_binding_digest):
        from riichienv import GameRule, RiichiEnv
        self._profile = deepcopy(profile)
        if digest(self._profile) not in self._supported_profiles:
            fail('UNSUPPORTED_PROFILE', 'only the reviewed complete source profile is supported')
        self._options = offline_options_from_profile(self._profile,
            dict(revision='mjai/recorded-deltas-1', settlement_deltas='per_event'))
        self._rule = GameRule(**self._options.native_rules)
        self._game_mode = '4p-red-half' if profile['mode'] == 'MODE_HANCHAN' else '4p-red-east'
        self._native = RiichiEnv(game_mode=self._game_mode, rule=self._rule)
        self._native.policy_set_external_rounds()
        self._native.policy_set_temp_furiten_own_draw()
        self._model_binding = model_binding_digest
        self._seat_bindings = None
        self._controller_types = None
        self._controller_schemas = None
        self._evidence_sink = None
        self._codec = MahjongTokenizer()
        self._valid = False
        self._views = {}
        self._plans = {}

    def reset(self, *, seed=None):
        return self._reset(seed=seed)

    def _reset(self, **kwargs):
        """Internal explicit wall/header injection is reserved for rule fixtures."""
        from riichienv import RiichiEnv
        self._native = RiichiEnv(game_mode=self._game_mode, rule=self._rule, seed=kwargs.get('seed'))
        self._native.policy_set_external_rounds()
        self._native.policy_set_temp_furiten_own_draw()
        self._instance = uuid.uuid4().hex
        self._game = uuid.uuid4().hex
        self._revision = 0
        self._seq = 0
        self._log_offset = 0
        self._observed = {}
        self._transitions = {}
        self._transition_hashes = {}
        self._round = -1
        self._builder = None
        self._session = OfflineSession(self._options, rule=self._rule)
        self._views, self._plans = {}, {}
        self._valid = True
        self._status = 'advanced'
        self._round_outcome = None
        self._public_result = None
        returned = self._native.reset(scores=self._profile['initial_scores'], **kwargs)
        try:
            self._capture(returned, {}, 'reset')
            self._publish()
        except Exception:
            self._valid = False
            raise
        return self._result()

    def _result(self):
        return ApplyResult(self._instance, self._revision, self._status)

    def _check(self):
        if not self._valid:
            fail('EVENT_STREAM_INCOMPLETE', 'bridge is invalid; reset is required')

    @staticmethod
    def _mask_fact(fact, viewer):
        value = deepcopy(fact)
        if value['type'] == 'start_kyoku':
            value['tehais'] = [hand if p == viewer else ['?']*len(hand)
                               for p, hand in enumerate(value['tehais'])]
        if value['type'] == 'tsumo' and value['actor'] != viewer:
            value['pai'] = '?'
        return value

    def _capture(self, returned, actions, before):
        """Read original returned batches first, then exactly once for other seats."""
        logs = self._native.mjai_log
        new = logs[self._log_offset:]
        observations = []
        batch = dict(returned)
        for viewer in range(4):
            if viewer not in batch:
                batch[viewer] = self._native.get_observation(viewer)
            events = [json.loads(e) for e in batch[viewer].new_events()]
            if events != [self._mask_fact(e, viewer) for e in new]:
                fail('EVENT_STREAM_INCOMPLETE', 'native incremental stream differs from ordered facts', viewer=viewer)
            for order, value in enumerate(events):
                observations.append(ObservedEvent(self._instance, self._seq, viewer, order,
                    (self._seq, order), json.dumps(value, sort_keys=True)))
        transition = NativeTransition(self._instance, self._seq, before, str(self._native.phase),
            tuple((p, json.dumps(a.to_dict(), sort_keys=True)) for p, a in sorted(actions.items())),
            tuple(json.dumps(e, sort_keys=True) for e in new), tuple(observations))
        self._ingest_transition(transition, actions=actions)
        if self._evidence_sink is not None:
            self._evidence_sink.append(transition)
        self._log_offset = len(logs)
        self._seq += 1
        self._batch = batch

    def _ingest_transition(self, transition, *, actions=None):
        self._check()
        key = (transition.env_instance_id, transition.transition_seq)
        signature = digest(asdict(transition))
        previous = self._transition_hashes.get(key)
        if previous is not None:
            if previous != signature:
                self._valid = False
                fail('EVENT_STREAM_INCOMPLETE', 'conflicting transition envelope')
            return
        if transition.env_instance_id != self._instance or transition.transition_seq != self._seq:
            self._valid = False
            fail('EVENT_STREAM_INCOMPLETE', 'missing or foreign transition envelope')
        for observed in transition.observations:
            identity = (observed.transition_seq, observed.viewer, observed.event_order)
            if identity in self._observed and self._observed[identity] != observed:
                self._valid = False
                fail('EVENT_STREAM_INCOMPLETE', 'conflicting observation envelope')
            self._observed[identity] = observed
        facts = [json.loads(e) for e in transition.facts]
        # Native ankan has no log until commit; kakan logs its premature upgrade.
        # The accepted action is the authoritative public attempt in both cases.
        attempts = []
        from riichienv import ActionType
        from riichienv.convert import tid_to_mjai
        for player, action in (actions or {}).items():
            if action.action_type == ActionType.ANKAN:
                attempts.append(dict(type='ankan', actor=player,
                    consumed=[tid_to_mjai(t) for t in action.consume_tiles]))
            elif action.action_type == ActionType.KAKAN:
                pon = next(m for m in self._session.snapshot()['melds'][player]
                           if m['name'] == 'MELD_PON' and m['base_tile'] == tid_to_mjai(action.tile).removesuffix('r'))
                attempts.append(dict(type='kakan', actor=player, pai=tid_to_mjai(action.tile), consumed=pon['tiles']))
        for attempt in attempts:
            match = next((i for i, e in enumerate(facts) if e['type'] == attempt['type']), None)
            if match is None:
                facts.insert(0, attempt)
            else:
                facts[match] = attempt
        # An ankan deferred log after a prior attempt is a commit fact, never a
        # second attempt. The mirror commits it when closing that public window.
        if not attempts and self._session.pending is not None:
            facts = [e for e in facts if e['type'] not in ('ankan', 'kakan')]
        try:
            index = 0
            while index < len(facts):
                event = facts[index]
                if event['type'] in ('hora', 'ryukyoku'):
                    group = [event]
                    while index+1 < len(facts) and facts[index+1]['type'] == 'hora':
                        index += 1
                        group.append(facts[index])
                    self._consume_result(group)
                else:
                    self._consume(event)
                index += 1
            self._transitions[key] = transition
            self._transition_hashes[key] = signature
            # Retain only this transition's immutable evidence, not the full game.
            self._transitions = {key: transition}
            self._observed = {k: v for k, v in self._observed.items() if k[0] == self._seq}
        except Exception as exc:
            self._valid = False
            fail('EVENT_STREAM_INCOMPLETE', f'canonical transition failed: {exc}', transition_seq=self._seq)

    def _consume(self, event):
        kind = event['type']
        if kind == 'start_game':
            return
        if kind == 'start_kyoku':
            self._public_result = None
            if self._builder is not None:
                self._session.round_decision().check_boundary(event)
                self._builder.next_state(dict(game_over=False, next_dealer=Atom('ABS_P'+str(event['oya'])),
                    next_round=dict(round_wind=Atom('ROUND_'+event['bakaze']), round_number=event['kyoku'],
                                    rounds_played=self._round+1),
                    next_honba=event['honba'], next_kyotaku=event['kyotaku'], reason=None), self._seq)
            self._round += 1
            self._builder = RoundBuilder(self._round, event, self._seq, self._session)
        elif kind == 'end_game':
            from mahjong.envs.round_ir import canonical_settlement
            final, ranks = self._final_settlement
            if final is not None and final['payments']:
                self._builder.emit('EV_SETTLEMENT', (canonical_settlement(final),), [self._seq])
            reason = Atom(self._session.end_reason)
            self._builder.next_state(dict(game_over=True, next_dealer=None, next_round=None,
                next_honba=None, next_kyotaku=self._session.env.riichi_sticks, reason=reason), self._seq)
            self._builder.emit('EV_GAME_END', (dict(final_scores=self._session.env.scores(),
                final_ranks=ranks, rank_points=Atom('UNKNOWN'), unit=Atom('UNIT_RANK_POINTS'), reason=reason),), [self._seq])
            self._status = 'game_ended'
        else:
            if self._builder is None:
                fail('EVENT_STREAM_INCOMPLETE', 'event before initial round')
            self._builder.apply(SourceEvent(self._seq, event))

    def _consume_result(self, group):
        # Compute values/payments independently through the full profile before
        # accepting native balances. Shape and liability come from public
        # canonical history plus the winner's verified hand and revealed ura.
        from mahjong.envs.payment_rules import WinValue, NAGASHI_NO_HONBA_CARRY_POOL
        from mahjong.envs.result_selection import select_win_values
        from mahjong.envs.result_constants import NATIVE_YAKUMAN
        session = self._session
        if group[0]['type'] == 'ryukyoku':
            # The result adjudicates the last discard response. Publish deferred
            # reach acceptance/dora before checking abort predicates or reading
            # balances; four-riichi otherwise still sees only three deposits.
            self._builder.close_response(group[0], [self._seq])
        scores, kyotaku = session.env.scores(), session.env.riichi_sticks
        if group[0]['type'] == 'hora':
            claims, public_winners = [], []
            for event in group:
                actor, target = event['actor'], event['target']
                tsumo = actor == target
                tile = session.last_draw[1] if tsumo else (
                    session.pending['event'].get('pai', session.pending['reserved'][0])
                    if session.pending else session.last_discard['pai'])
                material = session.result_material(actor, tile, tsumo, chankan=session.pending is not None)
                ura = event.get('ura_markers', [])
                if actor in session.accepted and len(ura) != len(session.env.dora_indicators):
                    fail('EVENT_STREAM_INCOMPLETE', 'winner ura evidence is incomplete')
                if actor not in session.accepted and ura:
                    fail('EVENT_STREAM_INCOMPLETE', 'ura without accepted riichi')
                bonus = session.count_dora(material['full_hand'], ura)
                liability = {c['native_yaku_id']: c['pao_player'] for c in material['responsibility']}
                candidates, materials = [], []
                for value in material['native_candidates']:
                    if not value['native_has_yaku']:
                        continue
                    if value['native_yakuman_ids']:
                        components = tuple(dict(name=NATIVE_YAKUMAN[i],
                            yakuman_units=session.payment_rules.multipliers[NATIVE_YAKUMAN[i]],
                            pao_player=liability.get(i)) for i in value['native_yakuman_ids'])
                        payers = set(liability.values())
                        if len(payers) > 1:
                            fail('EVENT_STREAM_INCOMPLETE', 'ambiguous native honba liability')
                        claim = WinValue(actor, None if tsumo else target, yakuman=components,
                                         honba_payer=next(iter(payers), None))
                    else:
                        claim = WinValue(actor, None if tsumo else target,
                                         value['native_han']+bonus, value['native_fu'])
                    candidates.append(claim)
                    materials.append(value)
                ranked = select_win_values(candidates, [()] * len(candidates),
                                           dealer=session.header['oya'], payment_rules=session.payment_rules)
                selected = ranked['selected']
                claims.append(candidates[selected])
                from mahjong.envs.result_constants import NATIVE_ORDINARY
                value, claim = materials[selected], candidates[selected]
                public_winners.append(dict(actor=actor, target=target, tsumo=tsumo,
                    hand=material['pre_win_hand'], winning_tile=tile, melds=material['melds'],
                    ura_indicators=list(ura), han=None if claim.yakuman else claim.han,
                    fu=None if claim.yakuman else claim.fu,
                    yakuman_units=sum(c['yakuman_units'] for c in claim.yakuman),
                    yaku=[(NATIVE_YAKUMAN | NATIVE_ORDINARY)[i] for i in value['native_yaku_ids']
                          if i in (NATIVE_YAKUMAN | NATIVE_ORDINARY)],
                    dora=material['dora_count'], aka=material['aka_count'], ura=bonus))
            expected = session.payment_rules.win(scores, dealer=session.header['oya'],
                honba=session.header['honba'], kyotaku=kyotaku, claims=claims)
            round_result = dict(kind='win', winners=[e['actor'] for e in group])
        else:
            material = session.draw_material()
            reason = group[0]['reason']
            if reason in ('exhaustive_draw', 'nagashimangan'):
                nagashi = material['nagashi'] if reason == 'nagashimangan' else []
                session.validate_draw('nagashi' if nagashi else 'exhaustive', material)
                expected = session.payment_rules.draw(scores, dealer=session.header['oya'],
                    honba=session.header['honba'], kyotaku=kyotaku, tenpai=material['tenpai'],
                    nagashi=nagashi, nagashi_extras=NAGASHI_NO_HONBA_CARRY_POOL if nagashi else None)
                round_result = dict(kind='nagashi' if nagashi else 'exhaustive', tenpai=material['tenpai'])
            else:
                kinds = {'kyushu_kyuhai':'kyuushu', 'sufuurenta':'four_winds',
                         'suukansansen':'four_kan', 'suucha_riichi':'four_riichi', 'sanchaho':'triple_ron'}
                kind = kinds.get(reason)
                if kind is None:
                    fail('ENVIRONMENT_STEP_FAILED', f'unknown native draw reason: {reason}')
                actor = self._session.action_actor if kind == 'kyuushu' else None
                session.validate_draw(kind, material, actor=actor)
                expected = dict(score_delta=[0]*4, scores_after=scores, next_kyotaku=kyotaku)
                round_result = dict(kind='abort')
        native_delta = [sum(e['deltas'][p] for e in group) for p in range(4)]
        if native_delta != expected['score_delta']:
            fail('EVENT_STREAM_INCOMPLETE', 'native settlement differs from complete profile',
                 native_delta=native_delta, expected_delta=expected['score_delta'])
        values = [{k: v for k, v in e.items() if k not in ('yakus', 'fan', 'pao', 'hora_points', 'hora_tehais')}
                  for e in group]
        self._builder.result([SourceEvent(self._seq, e) for e in values])
        session.round_result = round_result
        outcome = session.round_decision().outcomes
        if len(outcome) != 1:
            fail('EVENT_STREAM_INCOMPLETE', 'complete round facts did not determine a unique boundary')
        self._round_outcome = outcome[0]
        self._public_result = dict(round_id=self._round, revision=self._revision + 1,
            kind=round_result['kind'], reason=group[0].get('reason'),
            winners=public_winners if group[0]['type'] == 'hora' else [],
            score_delta=list(expected['score_delta']), scores=list(session.env.scores()),
            kyotaku=session.env.riichi_sticks, game_over=outcome[0].game_over)
        self._status = 'round_ended'

    def _normalize(self, native, viewer, *, riichi=False, environment=None):
        from riichienv import ActionType
        from riichienv.convert import tid_to_mjai
        kind = native.action_type
        environment = self._native if environment is None else environment
        tile = None if native.tile is None else Atom('TILE_'+tid_to_mjai(native.tile))
        consumed = tuple(sorted((Atom('TILE_'+tid_to_mjai(t)) for t in native.consume_tiles), key=self._codec.tile_key))
        window = self._builder.pending_window
        source = None if window is None else int(self._builder.ir_windows[window['window_id']]['source'].name[-1])
        if kind == ActionType.DISCARD:
            tsumogiri = (native.tile == environment.drawn_tile and not self._session.after_call
                         and not self._batch[viewer].forced_tedashi)
            return Action('A_RIICHI_DISCARD' if riichi else 'A_DISCARD',
                          (tile, Atom('TSUMOGIRI' if tsumogiri else 'TEDASHI')))
        simple = {ActionType.PASS: 'A_PASS', ActionType.TSUMO: 'A_TSUMO_WIN',
                  ActionType.KYUSHU_KYUHAI: 'A_ABORT_KYUUSHU'}
        if kind in simple:
            return Action(simple[kind])
        if kind == ActionType.RON:
            return Action('A_RON', (seat(source, viewer), tile))
        calls = {ActionType.CHI: 'A_CHI', ActionType.PON: 'A_PON', ActionType.DAIMINKAN: 'A_DAIMINKAN'}
        if kind in calls:
            return Action(calls[kind], (seat(source, viewer), tile, *consumed))
        if kind == ActionType.ANKAN:
            return Action('A_ANKAN', consumed)
        if kind == ActionType.KAKAN:
            melds = self._session.snapshot()['melds'][viewer]
            matches = [m for m in melds if m['name'] == 'MELD_PON' and m['base_tile'] == tid_to_mjai(native.tile).removesuffix('r')]
            if len(matches) != 1:
                fail('ACTION_UNREPRESENTABLE', 'kakan original pon reference is ambiguous')
            return Action('A_KAKAN', (matches[0]['meld_id'], tile))
        fail('ACTION_UNREPRESENTABLE', f'unmapped native action {kind}')

    def _publish(self):
        from riichienv import ActionType, Phase
        self._views, self._plans = {}, {}
        if self._status != 'advanced':
            return
        builder = self._builder
        response = self._native.phase == Phase.WaitResponse
        if response and builder.pending_window is None:
            fail('EVENT_STREAM_INCOMPLETE', 'response phase without canonical window')
        viewers = builder.pending_window['viewers'] if response else [self._native.current_player]
        for viewer in viewers:
            native_actions = self._native._get_legal_actions(viewer)
            normalized = {}
            for native in native_actions:
                plans = [(self._normalize(native, viewer), (native,))] if native.action_type != ActionType.RIICHI else []
                if native.action_type == ActionType.RIICHI:
                    plans = [(self._normalize(discard, viewer, riichi=True), (native, discard))
                             for discard in self._native.policy_riichi_discards(viewer)]
                    if not plans:
                        fail('LEGAL_SET_INCOMPLETE', 'riichi declaration has no legal discard')
                for action, plan in plans:
                    suffix = tuple(self._codec.encode_action_suffix(action))
                    if suffix in normalized and normalized[suffix][0] != action:
                        fail('ACTION_UNREPRESENTABLE', 'semantic action encoding collision')
                    # Native order is physical-ID sorted; explicitly select a
                    # stable representative independent of enumeration order.
                    signature = lambda p: tuple((int(a.action_type), -1 if a.tile is None else a.tile,
                                                  tuple(a.consume_tiles)) for a in p)
                    if suffix not in normalized or signature(plan) < signature(normalized[suffix][1]):
                        normalized[suffix] = (action, plan)
            if not normalized:
                fail('LEGAL_SET_EMPTY', 'native decision has no representable candidates', viewer=viewer)
            self._check_candidates(viewer, set(normalized), response=response)
            anchor = builder.anchors[viewer][-1]
            window_id = builder.pending_window['window_id'] if response else anchor.window_id
            key = DecisionKey(WindowKey(self._instance, self._game, self._round, window_id, self._revision), viewer)
            history = [project_event(e, viewer, ankan_public=self._options.ankan_public)
                       for e in builder.events if e.kind in HISTORY_KINDS]
            sequence = PolicySequenceInput(project_header(builder.initial, viewer, self._options.mode),
                                            history, project_hand(self._session.snapshot(), viewer))
            moves = []
            for index, (suffix, (action, plan)) in enumerate(sorted(normalized.items())):
                ref = str(index)
                moves.append(LegalMove(action, ref))
                self._plans[(key, ref)] = plan
            self._views[viewer] = DecisionView(key, sequence, moves, digest(self._profile), True)

    def bind_controllers(self, controllers):
        """Fix controller bindings before reset, without inventing Actor probabilities."""
        if self._valid or self._seat_bindings is not None or len(controllers) != 4:
            fail('CONTRACT_MISMATCH', 'controllers must be bound once before reset')
        from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, LEGACY_SCHEMA
        kinds, bindings, schemas = [], [], []
        for kind, model in controllers:
            if kind not in ('human', 'random', 'model', 'llm') or ((kind == 'model') != (model is not None)):
                fail('CONTRACT_MISMATCH', 'invalid controller binding')
            if model is not None and model.source and 'actor_ref' in model.source:
                fail('CONTRACT_MISMATCH', 'Web controllers require NTP checkpoints')
            kinds.append(kind)
            bindings.append((None if model is None else model.binding_digest, None))
            schemas.append(CONTEXT_SCHEMA if kind == 'llm' else LEGACY_SCHEMA if model is None else model.input_schema)
        self._controller_types = tuple(kinds)
        self._controller_schemas = tuple(schemas)
        self._seat_bindings = tuple(bindings)

    def select_legal(self, key, execution_ref):
        self._check()
        view = self._views.get(key.viewer)
        if view is None or view.key != key:
            fail('STALE_DECISION', 'non-model choice is stale')
        if self._controller_types is None or self._controller_types[key.viewer] not in ('human', 'random', 'llm'):
            fail('CONTRACT_MISMATCH', 'model seats require a generated decision')
        move = next((m for m in view.legal_moves if m.execution_ref == execution_ref), None)
        if move is None:
            fail('CONTRACT_MISMATCH', 'unknown legal candidate')
        return SelectedDecision(view.key, move.action, move.execution_ref)

    def decision_views(self):
        self._check()
        return tuple(self._views[p] for p in sorted(self._views))

    def is_response_window(self):
        from riichienv import Phase
        self._check()
        return self._native.phase == Phase.WaitResponse

    def round_result(self):
        self._check()
        return deepcopy(self._public_result)

    def public_table(self, viewer):
        """Continuous viewer-only DTO, including phases without DecisionViews."""
        self.snapshot(viewer)  # validates owner and viewer, never reads others' views
        raw = self._session.snapshot()
        header = self._session.header
        rivers = [[] for _ in range(4)]
        riichi_discard = set()
        for event in self._native.mjai_log:
            kind = event['type']
            if kind == 'start_kyoku':
                rivers = [[] for _ in range(4)]
                riichi_discard.clear()
            elif kind == 'reach':
                riichi_discard.add(event['actor'])
            elif kind == 'dahai':
                actor = event['actor']
                rivers[actor].append(dict(tile=event['pai'], tsumogiri=event['tsumogiri'],
                    riichi=actor in riichi_discard, called=False))
                riichi_discard.discard(actor)
            elif kind in ('chi', 'pon', 'daiminkan') and rivers[event['target']]:
                rivers[event['target']][-1]['called'] = True
        players = []
        for actor in range(4):
            melds = deepcopy(raw['melds'][actor])
            if actor != viewer and self._options.ankan_public == 'base_only':
                for meld in melds:
                    if meld['name'] == 'MELD_ANKAN':
                        meld['tiles'] = [meld['base_tile']] * 4
            hand = sorted(raw['hands'][actor], key=lambda t: self._codec.tile_key(Atom('TILE_'+t))) if actor == viewer else None
            drawn = None
            if actor == viewer and len(hand) % 3 == 2 and not self._session.after_call and self._session.last_draw:
                p, tile = self._session.last_draw
                if p == actor and tile in hand:
                    drawn = tile
                    hand.remove(tile)
            players.append(dict(seat=actor, wind='ESWN'[(actor-header['oya']) % 4],
                score=raw['scores'][actor], dealer=actor == header['oya'],
                riichi=actor in self._session.accepted, hand=hand, drawn=drawn,
                hand_count=len(raw['hands'][actor]), melds=melds, river=rivers[actor]))
        view = self._views.get(viewer)
        response_trigger = None
        if self.is_response_window() and self._builder.pending_window is not None:
            trigger = self._builder.ir_windows[self._builder.pending_window['window_id']]
            response_trigger = dict(source=int(trigger['source'].name[-1]),
                tile=trigger['tile'].name.removeprefix('TILE_'),
                kind='kan' if trigger['phase'].name == 'PHASE_KAN_RESPONSE' else 'discard')
        result = dict(round_id=self._round, revision=self._revision, status=self._status,
            round_wind=header['bakaze'], round_number=header['kyoku'], honba=header['honba'],
            kyotaku=raw['kyotaku'], live_remaining=raw['live_remaining'],
            dora_indicators=raw['dora_indicators'], players=players,
            active_seats=sorted(self._views), response_window=self.is_response_window(),
            response_trigger=response_trigger,
            window=None if view is None else asdict(view.key.window),
            candidates=[] if view is None else [dict(id=m.execution_ref, kind=m.action.kind,
                args=[a.name if isinstance(a, Atom) else a for a in m.action.args]) for m in view.legal_moves],
            result=self.round_result())
        if self._status == 'game_ended':
            result['final'] = dict(scores=list(self._session.env.scores()), ranks=list(self._final_settlement[1]))
        return result

    def bind_actors(self, actors):
        """Bind the four actual loaded instances before resetting a match."""
        if self._valid or self._controller_types is not None or len(actors) != 4:
            fail('CONTRACT_MISMATCH', 'seat bindings must be fixed before reset')
        self._seat_bindings = tuple((a.binding_digest, deepcopy(a.source['actor_ref'])) for a in actors)
        compatibility = [a.source['compatibility'] for a in actors]
        if any(c != compatibility[0] for c in compatibility):
            fail('CONTRACT_MISMATCH', 'incompatible seat actors')

    def training_window(self):
        """Detached action-before snapshot; never passed to Actor encoding."""
        self._check()
        from mahjong.rl.oracle import encode_oracle
        raw = self._session.snapshot()
        return dict(views=tuple(self._views[p] for p in sorted(self._views)),
                    oracle={p: encode_oracle(raw, self._session.header, p) for p in self._views},
                    scores=tuple(raw['scores']), kyotaku=raw['kyotaku'],
                    round_id=self._round, revision=self._revision, status=self._status)

    def _with_evidence(self, method, *args, **kwargs):
        if self._evidence_sink is not None:
            fail('CONTRACT_MISMATCH', 'nested evidence capture')
        self._evidence_sink = []
        before = None if not self._valid else self.training_window()
        try:
            result = method(*args, **kwargs)
            after = self.training_window()
            return dict(result=result, transitions=tuple(self._evidence_sink),
                        before=before, after=after,
                        ranks=deepcopy(self._final_settlement[1]) if result.status == 'game_ended' else None)
        except Exception as exc:
            if hasattr(exc, 'details'):
                exc.details['partial_transitions'] = json.loads(json.dumps([asdict(t) for t in self._evidence_sink]))
            raise
        finally:
            self._evidence_sink = None

    def reset_with_evidence(self, **kwargs):
        return self._with_evidence(self.reset, **kwargs)

    def apply_with_evidence(self, decision):
        return self._with_evidence(self.apply, decision)

    def apply_responses_with_evidence(self, window, decisions):
        automatic = tuple(p for p in self._views if p not in decisions)
        evidence = self._with_evidence(self.apply_responses, window, decisions)
        evidence['automatic_passes'] = automatic
        return evidence

    def next_round_with_evidence(self):
        return self._with_evidence(self.next_round)

    def snapshot(self, viewer):
        self._check()
        if type(viewer) is not int or not 0 <= viewer < 4:
            fail('CONTRACT_MISMATCH', 'viewer must be a seat in [0,4)')
        return self._views.get(viewer)

    def bind_result(self, view, result):
        self._check()
        if self._controller_types is not None and self._controller_types[view.key.viewer] != 'model':
            fail('CONTRACT_MISMATCH', 'non-model seat cannot bind model generation')
        original = self._views.get(view.key.viewer)
        if original is None or original != view or result.key != original.key:
            fail('STALE_DECISION', 'view/result differs from current frozen decision')
        suffixes = tuple(sorted(tuple(self._codec.encode_action_suffix(m.action)) for m in original.legal_moves))
        if (result.legal_set_digest != legal_digest(original, suffixes)
                or result.model_binding_digest != (self._model_binding if self._seat_bindings is None else self._seat_bindings[view.key.viewer][0])
                or (self._seat_bindings is not None and result.actor_ref != self._seat_bindings[view.key.viewer][1])
                or type(result.candidate_index) is not int or not 0 <= result.candidate_index < len(suffixes)
                or result.suffix_ids != suffixes[result.candidate_index]
                or result.finish_reason != 'end_decision'
                or len(result.token_logprobs) != len(result.suffix_ids)
                or not all(math.isfinite(p) and p <= 0 for p in result.token_logprobs)
                or not math.isclose(math.fsum(result.token_logprobs), result.model_action_logprob, abs_tol=1e-6)):
            fail('CONTRACT_MISMATCH', 'generation differs from frozen binding/legal set')
        action = self._codec.decode_action_suffix(result.suffix_ids)
        move = next((m for m in original.legal_moves if m.action == action), None)
        if move is None or tuple(self._codec.encode_action_suffix(action)) != result.suffix_ids:
            fail('ACTION_UNREPRESENTABLE', 'generated action differs from frozen execution mapping')
        return EnvDecision(original.key, move.action, move.execution_ref, result)

    def _verify_decision(self, decision):
        original = self._views.get(decision.key.viewer)
        if original is None or original.key != decision.key:
            fail('STALE_DECISION', 'window is stale or already closed')
        expected = (self.select_legal(decision.key, decision.execution_ref)
                    if isinstance(decision, SelectedDecision) else self.bind_result(original, decision.generation))
        if decision != expected:
            fail('CONTRACT_MISMATCH', 'execution plan differs from frozen mapping')
        return self._plans[(decision.key, decision.execution_ref)]

    def _advance(self, plans):
        self._check()
        executed = 0
        try:
            for actions in plans:
                before = str(self._native.phase)
                old_log = len(self._native.mjai_log)
                # Verify every exact physical action against the live legal set.
                for player, action in actions.items():
                    if action.to_dict() not in [a.to_dict() for a in self._native._get_legal_actions(player)]:
                        fail('STALE_DECISION', 'planned native action is no longer legal')
                returned = self._native.step(actions)
                executed += 1
                if self._native.policy_last_error is not None:
                    fail('ENVIRONMENT_STEP_FAILED', self._native.policy_last_error)
                if self._builder.pending_window is not None and self._session.furiten is not None:
                    # Complete private intentions are known here, after the
                    # one atomic native adjudication. They never enter HISTORY.
                    from riichienv import ActionType
                    pending = self._session.pending
                    tile = pending['event'].get('pai', pending['reserved'][0]) if pending else self._session.last_discard['pai']
                    choices = {ActionType.RON:'RON', ActionType.PASS:'PASS', ActionType.CHI:'CHI',
                               ActionType.PON:'PON', ActionType.DAIMINKAN:'DAIMINKAN'}
                    for player, action in actions.items():
                        available = self._session.ron_available(player, tile,
                            kan_kind=None if pending is None else pending['kind'])
                        self._session.furiten.observe(player, riichi=player in self._session.accepted,
                            ron_available=available, choice=choices[action.action_type])
                if len(self._native.mjai_log) == old_log and not returned:
                    fail('ENVIRONMENT_STEP_FAILED', 'native transition did not advance')
                self._capture(returned, actions, before)
            self._revision += 1
            self._publish()
            return self._result()
        except Exception as exc:
            self._valid = False
            fail('ENVIRONMENT_STEP_FAILED', str(exc), executed_steps=executed, bridge_invalid=True)

    def apply(self, decision):
        from riichienv import Phase
        self._check()
        if self._native.phase == Phase.WaitResponse:
            fail('RESPONSE_BATCH_REQUIRED', 'submit complete response intentions atomically')
        plan = self._verify_decision(decision)
        return self._advance([{decision.key.viewer: action} for action in plan])

    def apply_responses(self, window, decisions):
        from riichienv import ActionType, Phase
        self._check()
        current = next(iter(self._views.values()), None)
        if current is None or current.key.window != window:
            fail('WINDOW_ALREADY_CLOSED', 'response window is stale or closed')
        if self._native.phase != Phase.WaitResponse:
            fail('RESPONSE_BATCH_REQUIRED', 'current decision is not a response window')
        if set(decisions) - self._views.keys():
            fail('CONTRACT_MISMATCH', 'response includes a nonparticipant')
        actions = {}
        for player, view in self._views.items():
            if player in decisions:
                if decisions[player].key.viewer != player:
                    fail('CONTRACT_MISMATCH', 'response dictionary seat differs from action viewer')
                plan = self._verify_decision(decisions[player])
                actions[player] = plan[0]
            elif len(view.legal_moves) == 1 and view.legal_moves[0].action.kind == 'A_PASS':
                actions[player] = self._plans[(view.key, view.legal_moves[0].execution_ref)][0]
                if actions[player].action_type != ActionType.PASS:
                    fail('CONTRACT_MISMATCH', 'forced PASS mapping differs')
            else:
                fail('RESPONSE_BATCH_INCOMPLETE', 'non-forced participant is missing', viewer=player)
        return self._advance([actions])

    def next_round(self):
        """Advance a completed round using the compiled project boundary table."""
        self._check()
        if self._status != 'round_ended' or self._round_outcome is None:
            fail('STALE_DECISION', 'next_round requires a completed round')
        outcome = self._round_outcome
        try:
            before = str(self._native.phase)
            if outcome.game_over:
                self._final_settlement = self._session.finalize()
                returned = self._native.policy_finish_game(self._session.env.scores())
            else:
                returned = self._native.policy_begin_round(outcome.next_dealer,
                    (outcome.next_position-1)//4, outcome.next_honba,
                    outcome.next_kyotaku, self._session.env.scores())
                self._status = 'advanced'
            self._capture(returned, {}, before)
            self._revision += 1
            self._round_outcome = None
            self._publish()
            return self._result()
        except Exception:
            self._valid = False
            raise

    def _check_candidates(self, viewer, actual, *, response):
        """Compare the complete online set with the project's offline semantics."""
        from riichienv import Action as NativeAction, ActionType
        from riichienv.convert import tid_to_mjai
        session = self._session
        mirror = session.env
        actions = []
        if response:
            window = self._builder.pending_window
            source = int(self._builder.ir_windows[window['window_id']]['source'].name[-1])
            pending = session.pending
            tile = pending['event'].get('pai', pending['reserved'][0]) if pending else session.last_discard['pai']
            tid = next(t for t in range(136) if tid_to_mjai(t) == tile)
            if pending is None:
                actions = [(a,False) for a in mirror.policy_claim_actions(viewer,source,tid)
                           if a.action_type != ActionType.RON]
            ron = session.ron_available(viewer,tile,kan_kind=None if pending is None else pending['kind'])
            if ron is None:
                fail('LEGAL_SET_INCOMPLETE', 'project ron availability is unknown', viewer=viewer)
            if ron:
                actions.append((NativeAction(ActionType.RON,tid,[],viewer),False))
            actions.append((NativeAction(ActionType.PASS,None,[],viewer),False))
        else:
            for action in mirror._get_legal_actions(viewer):
                if action.action_type == ActionType.RIICHI:
                    actions.extend((a,True) for a in mirror.policy_riichi_discards(viewer))
                else:
                    actions.append((action,False))
        expected = set()
        for a, riichi in actions:
            action = self._normalize(a, viewer, riichi=riichi, environment=mirror)
            expected.add(tuple(self._codec.encode_action_suffix(action)))
            # MJAI has tile names, so ordinary physical copies can share an ID
            # in the replay state. A held copy of the drawn tile also permits
            # TEDASHI; accepted riichi still permits only the actual draw.
            if (action.kind in ('A_DISCARD','A_RIICHI_DISCARD') and action.args[1] == Atom('TSUMOGIRI')
                    and viewer not in session.accepted and mirror.hands[viewer].count(a.tile) > 1):
                alternative = Action(action.kind, (action.args[0], Atom('TEDASHI')))
                expected.add(tuple(self._codec.encode_action_suffix(alternative)))
        if actual != expected:
            fail('LEGAL_SET_INCOMPLETE', 'native complete candidates differ from project replay',
                 viewer=viewer, missing=[list(s) for s in sorted(expected-actual)],
                 extra=[list(s) for s in sorted(actual-expected)])
