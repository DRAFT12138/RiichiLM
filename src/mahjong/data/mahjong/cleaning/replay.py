"""Source orchestration and canonical IR; all rules state belongs to envs."""
from mahjong.envs.round_ir import abs_seat, tile, furiten_values, canonical_settlement
from copy import deepcopy

from mahjong.envs.offline_mjai import OfflineSession, ReplayError, UnknownCurrentState
from mahjong.envs.offline_recovery import RoundRecovery
from mahjong.tokenizer import Atom, Event
from .common import CleaningError, require
from .models import FrozenAnchor, GameResult, PreparedRound, PreparedView, VISIBLE
from .projection import project_event, project_current_hand, project_header
from .ir import (UNKNOWN, HIDDEN, PHASES, IRProtocolFailure, check_ir_event,
                 field_origins, to_json, unknown_next_state)
GIMITE_RESULT = 'gimite-mjai/result-1'


class RoundBuilder:
    def __init__(self, round_id, initial, raw_line, session):
        self.session=session
        self.events=[]
        self.ledger=[]
        self.windows=[]
        self.ir_windows={}
        self.anchors=[[] for _ in range(4)]
        self.pending_window=None
        self.deferred_public=[]
        self.next_meld=0
        self.facts=dict(results=[],settlements=[],canonical_results=[],state_gaps=[])
        self.initial=deepcopy(initial)
        self.round_id=round_id
        session.begin_round(initial)
        start=dict(round_wind=Atom('ROUND_'+initial['bakaze']),round_number=initial['kyoku'],
            honba=initial['honba'],kyotaku=initial['kyotaku'],rounds_played=round_id,
            dealer=abs_seat(initial['oya']),scores=list(initial['scores']),
            hands=[[tile(t) for t in h] for h in initial['tehais']],hand_counts=[13]*4,
            dora_indicators=[tile(initial['dora_marker'])])
        self.emit('EV_ROUND_START',(start,),[raw_line],origins={'state.rounds_played':'derived'})
        self.emit('EV_DORA_REVEAL',(tile(initial['dora_marker']),),[raw_line])
        self.complete=False
        self.invalid_reason=None

    def emit(self,kind,payload,lines,*,origins=None,**references):
        event_id=len(self.events)
        require(event_id<100_000,'RESOURCE_LIMIT','too many events in one round')
        event=Event(kind,event_id,tuple(payload))
        try:
            check_ir_event(event,self.events,self.ir_windows)
        except CleaningError as error:
            raise IRProtocolFailure(str(error)) from error
        self.events.append(event)
        from mahjong.tokenizer.codec import EVENT_FIELDS
        derived=kind in ('EV_RESPONSE_OPEN','EV_RESPONSE_CLOSE','EV_KAN_COMMIT','EV_SETTLEMENT','EV_GAME_END')
        origin='derived' if derived else 'recorded'
        field_provenance=field_origins(dict(zip(EVENT_FIELDS[kind],event.payload)),origin)
        field_provenance.update(origins or {})
        # Persist the scalar context needed for references, not full payloads.
        scalar=dict(zip(EVENT_FIELDS[kind],event.payload))
        context={}
        for name in ('actor','target'):
            if name in scalar:
                context[name]=None if scalar[name] is None else int(scalar[name].name[-1])
        for name in ('tile','base_tile'):
            if name in scalar:
                context[name]=scalar[name].name
        for name in ('meld_id','trigger_id','window_id'):
            if name in scalar:
                context[name]=scalar[name]
        for name,value in references.items():
            if name in context and context[name]!=value:
                raise IRProtocolFailure(f'conflicting {name} reference')
        context.update(references)
        self.ledger.append(dict(event_id=event_id,kind=kind,source_refs=[dict(raw_line=l) for l in lines],
                                provenance=origin,field_provenance=field_provenance,
                                **context))
        if kind in ('EV_RESPONSE_CLOSE','EV_WIN','EV_DRAW_END','EV_ABORT_END','EV_NAGASHI','EV_REVEAL',
                    'EV_SETTLEMENT','EV_ROUND_END','EV_GAME_END'):
            self.facts['canonical_results'].append(dict(event_id=event_id,kind=kind,payload=to_json(event.payload)))
        return event_id

    def window(self,trigger,actor,slot,lines):
        window_id=4*trigger+slot
        viewers=[actor] if slot in (0,1) else [(actor+i)%4 for i in range(1,4)]
        event=self.events[trigger]
        ir_window=dict(window_id=window_id,trigger_id=trigger,phase=Atom(PHASES[slot]),
            actor=abs_seat(actor) if slot<2 else None,source=abs_seat(actor),
            tile=None if slot<2 else event.payload[1],responders=[abs_seat(v) for v in viewers])
        if slot in (2,3):
            cutoff=self.emit('EV_RESPONSE_OPEN',(ir_window,),lines,
                             window_id=window_id,trigger_id=trigger)
        else:
            cutoff=trigger
        self.ir_windows[window_id]=ir_window
        window=dict(window_id=window_id,trigger_id=trigger,cutoff_event_id=cutoff,viewers=viewers,
                    source_refs=[dict(raw_line=l) for l in lines])
        self.windows.append(window)
        if self.session.furiten is not None:
            knowledge=[]
            for viewer in viewers:
                fields=self.session.furiten_state(viewer)
                if slot in (2,3):
                    pending=self.session.pending
                    fields['ron_available']=self.session.ron_available(viewer,
                        pending['event'].get('pai',pending['reserved'][0]) if pending else self.session.last_discard['pai'],
                        kan_kind=None if pending is None else pending['kind'])
                knowledge.append(dict(actor=viewer,**furiten_values(fields)))
            self.facts.setdefault('furiten_windows',[]).append(dict(event_id=cutoff,window_id=window_id,
                trigger_id=trigger,states=knowledge))
        states=self.session.read_current_states()
        if type(states) not in (list,tuple) or len(states)!=4:
            raise IRProtocolFailure('environment current-state read must contain four seats')
        for viewer in viewers:
            current=states[viewer]
            if type(current) is UnknownCurrentState:
                self.facts['state_gaps'].append(dict(code='CURRENT_STATE_UNKNOWN',stage='projection',reason=current.reason,
                    round_id=self.round_id,raw_line=lines[0],event_id=cutoff,window_id=window_id,viewer=viewer))
                continue
            if (type(current) is not dict or current.keys()!={'hands','melds'} or
                    type(current['hands']) is not list or type(current['melds']) is not list):
                raise IRProtocolFailure('environment current state must be known lists or an explicit unknown')
            state=project_current_hand(current['hands'],current['melds'],viewer)
            self.anchors[viewer].append(FrozenAnchor.capture(window_id,trigger,cutoff,state))
        if slot in (2,3):
            self.pending_window=window
        return window

    def close_response(self,next_event,lines,*,winners=(),abort_reason=None):
        if self.pending_window is None:
            if self.deferred_public:
                self.session.prepare_dora_boundary(next_event['type'],next_event.get('actor'))
                self.flush_public()
            return
        window=self.pending_window
        close=dict(actor=None,meld_id=None,winners=[],reason=None)
        kind=next_event['type']
        response_kind='hora' if abort_reason==Atom('ABORT_SANCHA_HOU') else kind
        pending=self.session.pending
        for public in self.deferred_public:
            if public.value['type']=='reach_accepted':
                try:
                    self.session.validate_accept(public.value,response_kind=response_kind)
                except ReplayError as error:
                    error.raw_line=public.raw_line
                    raise
        if abort_reason is not None:
            resolution='RESOLVE_ABORT'
            close['reason']=abort_reason
        elif kind=='hora':
            resolution='RESOLVE_RON'
            order=window['viewers']
            actors=list(winners) if winners else [next_event['actor']]
            require(all(a in order for a in actors),'INVALID_REFERENCE','ron response actors')
            close['winners']=[abs_seat(a) for a in sorted(actors,key=order.index)]
        elif kind in ('chi','pon','daiminkan'):
            require(pending is None,'INVALID_PHASE','call during kan response')
            resolution='RESOLVE_'+kind.upper()
            close.update(actor=abs_seat(next_event['actor']),meld_id=self.next_meld)
        elif pending is not None:
            resolution='RESOLVE_KAN'
            close.update(actor=abs_seat(pending['actor']),meld_id=pending['ref']['meld_id'])
        else:
            resolution='RESOLVE_NO_CALL'
        close_lines=list(dict.fromkeys([e.raw_line for e in self.deferred_public]+lines))
        if self.session.furiten is not None:
            claimants=window['viewers'] if abort_reason==Atom('ABORT_SANCHA_HOU') else winners
            knowledge=self.session.observe_responses(next_event,window['viewers'],winners=claimants,abort=abort_reason is not None)
            self.facts.setdefault('furiten_transitions',[]).append(dict(event_id=len(self.events),
                window_id=window['window_id'],trigger_id=window['trigger_id'],
                source_refs=list(close_lines),states=[furiten_values(r) for r in knowledge]))
        self.emit('EV_RESPONSE_CLOSE',(window['window_id'],Atom(resolution),close),close_lines)
        self.session.response_closed(response_kind)
        self.pending_window=None
        if self.session.pending is not None:
            pending=self.session.pending
            if abort_reason is not None:
                self.session.cancel_kan_for_abort()
            elif next_event['type']=='hora':
                robbed_tile=next_event.get('pai', pending['event'].get('pai'))
                if robbed_tile is None and len(set(pending['reserved']))==1:
                    robbed_tile=pending['reserved'][0]
                # Ankan chankan without a recorded exact tile cannot select a red copy.
                require(robbed_tile is not None,'MISSING_FIELD','robbed kan tile')
                self.session.rob_kan(robbed_tile)
                self.emit('EV_KAN_ROBBED',(abs_seat(pending['actor']),pending['ref']['meld_id'],window['window_id']),lines)
            else:
                event_id=len(self.events)
                ref=self.session.commit_kan(event_id)
                self.emit('EV_KAN_COMMIT',(abs_seat(pending['actor']),ref['meld_id']),lines,
                          actor=pending['actor'],meld_id=ref['meld_id'])
        self.flush_public()

    def flush_public(self):
        deferred,self.deferred_public=self.deferred_public,[]
        for source_event in deferred:
            try:
                self.apply(source_event,allow_defer=False)
            except (CleaningError,ReplayError) as error:
                error.raw_line=source_event.raw_line
                raise

    def apply(self,source_event,*,allow_defer=True):
        e=source_event.value
        kind=e['type']
        lines=[source_event.raw_line]
        if kind!='end_kyoku':self.session.check_play_phase()
        if allow_defer and ((kind in ('reach_accepted','dora') and self.pending_window is not None and self.session.pending is None)
                or (kind=='dora' and self.pending_window is None and self.session.action_actor is not None
                    and self.session.before_discard_dora_pending())):
            # No decision anchor can occur between these source facts and the
            # adjudicating event. Insert CLOSE before the facts only once the
            # actual call/no-call result is known; never claim an early NO_CALL.
            require(len(self.deferred_public)<5,'RESOURCE_LIMIT','too many interleaved public facts')
            if kind=='reach_accepted':
                self.session.validate_accept(e)
                require(not any(p.value['type']=='reach_accepted' for p in self.deferred_public),
                        'INVALID_PHASE','duplicate pending riichi acceptance')
            self.deferred_public.append(source_event)
            return
        if kind in ('chi','pon','daiminkan'):
            require(self.pending_window is not None,'INVALID_REFERENCE','call without an open response window')
        self.close_response(e,lines)
        actor=e.get('actor')
        if kind=='tsumo':
            draw_kind=self.session.draw(e)
            trigger=self.emit('EV_DRAW',(abs_seat(actor),tile(e['pai']),Atom(draw_kind)),lines,actor=actor)
            self.window(trigger,actor,0,lines)
        elif kind=='dahai':
            riichi_tile=self.session.discard(e)
            trigger=self.emit('EV_DISCARD',(abs_seat(actor),tile(e['pai']),
                 Atom('TSUMOGIRI' if e['tsumogiri'] else 'TEDASHI'),riichi_tile),lines,actor=actor)
            self.window(trigger,actor,2,lines)
        elif kind in ('chi','pon','daiminkan'):
            # The last public discard is shared, independent of projected history.
            triggers=[r['event_id'] for r in self.ledger if r['kind']=='EV_DISCARD']
            require(triggers,'INVALID_REFERENCE','call without discard')
            trigger_id=triggers[-1]
            meld_id=self.next_meld
            self.next_meld+=1
            created=len(self.events)
            self.session.call(e,created,meld_id,trigger_id)
            event_id=self.emit('EV_'+kind.upper(),(abs_seat(actor),abs_seat(e['target']),tile(e['pai']),
                [tile(t) for t in e['consumed']],meld_id,trigger_id),lines,
                actor=actor,meld_id=meld_id,trigger_id=trigger_id,called_tile='TILE_'+e['pai'])
            if kind=='daiminkan':
                commit=len(self.events)
                self.session.commit_kan(commit)
                self.emit('EV_KAN_COMMIT',(abs_seat(actor),meld_id),lines,actor=actor,meld_id=meld_id)
            else:
                self.window(event_id,actor,1,lines)
        elif kind in ('ankan','kakan'):
            attempt_id=len(self.events)
            window_id=4*attempt_id+3
            pending=self.session.kan_attempt(e,attempt_id,self.next_meld,window_id)
            meld_id=pending['ref']['meld_id']
            if kind=='ankan':
                self.next_meld+=1
                payload=(abs_seat(actor),tile(pending['ref']['base_tile']),[tile(t) for t in e['consumed']],meld_id,window_id)
            else:
                payload=(abs_seat(actor),tile(e['pai']),meld_id,window_id)
            trigger=self.emit('EV_'+kind.upper()+'_ATTEMPT',payload,lines,actor=actor,meld_id=meld_id)
            self.window(trigger,actor,3,lines)
        elif kind=='reach':
            self.session.declare(e)
            self.emit('EV_RIICHI_DECLARE',(abs_seat(actor),),lines,actor=actor)
        elif kind=='reach_accepted':
            self.session.accept(e)
            self.emit('EV_RIICHI_ACCEPT',(abs_seat(actor),),lines,actor=actor)
        elif kind=='dora':
            pending=self.session.reveal_dora(e)
            self.emit('EV_DORA_REVEAL',(tile(e['dora_marker']),),lines,
                      dora_meld_id=pending['meld_id'],dora_commit_event=pending['commit_event'])
        elif kind=='end_kyoku':
            require(self.session.ended,'MISSING_SETTLEMENT','end_kyoku without recorded result')
            self.emit('EV_ROUND_END',(unknown_next_state(),),lines)
            self.complete=True
        else:
            raise CleaningError('INVALID_PHASE',kind,raw_line=source_event.raw_line,stage='replay')

    def result(self,events):
        self.session.check_result_phase([e.value for e in events])
        lines=[e.raw_line for e in events]
        values=[e.value for e in events]
        response=self.pending_window
        robbed=self.session.pending
        rich=self.session.options.result_schema==GIMITE_RESULT
        claims=None
        draw_result=None
        selection_audits=[]
        if rich:
            from .result_adapter import (check_result_group, normalize_hora,
                                         normalize_ryukyoku, ryukyoku_kind)
            check_result_group(values)
        if values[0]['type']=='hora':
            require(len({e['actor'] for e in values})==len(values),'INVALID_REFERENCE','duplicate winner')
            is_tsumo=values[0]['actor']==values[0]['target']
            require(all((e['actor']==e['target'])==is_tsumo for e in values),
                    'INVALID_REFERENCE','mixed tsumo/ron group')
            if is_tsumo:
                require(len(values)==1 and self.session.action_actor==values[0]['actor'] and
                        self.session.last_draw is not None,'INVALID_PHASE','tsumo without own draw')
                trigger=next(e.event_id for e in reversed(self.events) if e.kind=='EV_DRAW')
                window_id=4*trigger
                winning_tile=self.session.last_draw[1]
            else:
                require(response is not None,'INVALID_REFERENCE','ron without response window')
                trigger,window_id=response['trigger_id'],response['window_id']
                expected_actor=int(self.ir_windows[window_id]['source'].name[-1])
                require(all(e['target']==expected_actor for e in values),'INVALID_REFERENCE','ron target')
                if robbed is not None:
                    winning_tile=values[0].get('pai',robbed['event'].get('pai'))
                    if winning_tile is None and len(set(robbed['reserved']))==1:
                        winning_tile=robbed['reserved'][0]
                    require(winning_tile is not None,'MISSING_FIELD','ambiguous robbed tile')
                else:
                    winning_tile=self.session.last_discard['pai']
            for e in values:
                require('pai' not in e or e['pai']==winning_tile,'INVALID_REFERENCE','recorded win tile')
                require(rich or not any(k in e for k in ('yakus','fan','hora_tehais','pao')),
                        'UNSUPPORTED_RESULT_DETAILS','source result schema not admitted')
                if not is_tsumo and (rich or self.session.furiten is not None):
                    self.session.check_ron_furiten(e['actor'])
                if not is_tsumo and robbed is not None:
                    self.session.check_kan_ron(e['actor'],winning_tile)
                if not rich and self.session.win_rules is not None:
                    self.session.check_recorded_win(e['actor'],winning_tile,is_tsumo,chankan=robbed is not None)
            self.close_response(values[0],lines,winners=[e['actor'] for e in values])
            order=self.ir_windows[window_id]['responders']
            canonical_winners=[]
            revealed=[HIDDEN]*4
            normalized={}
            normalized_claims={}
            normalized_origins={}
            for e in values:
                if rich:
                    audit={}
                    fields,claim,origins=normalize_hora(e,self.session,winning_tile,is_tsumo,chankan=robbed is not None,audit=audit)
                    selection_audits.append(dict(actor=e['actor'],**audit))
                    normalized[e['actor']],normalized_claims[e['actor']],normalized_origins[e['actor']]=fields,claim,origins
                    hand=e.get('hora_tehais')
                else:
                    hand=e.get('tehai')
                revealed[e['actor']]=[tile(t) for t in hand] if hand is not None else UNKNOWN
            for e in sorted(values,key=lambda e:order.index(abs_seat(e['actor']))):
                actor=e['actor']
                known_ura=e.get('ura_markers',e.get('uradora_markers'))
                if actor not in self.session.accepted:
                    require(not known_ura,'VISIBILITY_VIOLATION','ura without riichi')
                    ura=None
                else:
                    ura=UNKNOWN if known_ura is None else [tile(t) for t in known_ura]
                canonical_winners.append(dict(actor=abs_seat(actor),target=None if is_tsumo else abs_seat(e['target']),
                    method=Atom('METHOD_TSUMO' if is_tsumo else 'METHOD_RON'),tile=tile(winning_tile),
                    yaku=UNKNOWN,yaku_han=UNKNOWN,han_total=e.get('han',UNKNOWN),fu=e.get('fu',UNKNOWN),
                    dora_count=UNKNOWN,aka_count=UNKNOWN,ura_count=UNKNOWN,limit=UNKNOWN,yakuman_units=UNKNOWN,
                    pao_components=UNKNOWN,revealed_hands=deepcopy(revealed),ura_indicators=ura))
                if rich:
                    canonical_winners[-1].update(normalized[actor])
            origins={f'win_group.winners.{i}.tile':'recorded' if 'pai' in e else 'derived'
                     for i,e in enumerate(sorted(values,key=lambda e:order.index(abs_seat(e['actor']))))}
            origins.update({'win_group.window_id':'derived','win_group.trigger_id':'derived'})
            origins.update({f'win_group.winners.{i}.method':'derived' for i in range(len(canonical_winners))})
            if rich:
                for index,e in enumerate(sorted(values,key=lambda e:order.index(abs_seat(e['actor'])))):
                    origins.update({f'win_group.winners.{index}.{name}':origin
                                    for name,origin in normalized_origins[e['actor']].items()})
                if all(normalized_claims[e['actor']] is not None for e in values):
                    claims=[normalized_claims[e['actor']] for e in values]
            self.emit('EV_WIN',(dict(window_id=window_id,trigger_id=trigger,winners=canonical_winners),),lines,origins=origins)
        else:
            e=values[0]
            if rich:
                kind,abort=ryukyoku_kind(e)
                require(robbed is None or kind=='triple_ron','UNSUPPORTED_RESULT_DETAILS','unresolved kan at draw result')
                material=self.session.draw_material()
                self.close_response(e,lines,abort_reason=abort)
                draw_result=normalize_ryukyoku(e,self.session,material,response=response,robbed=robbed)
            else:
                self.close_response(e,lines)
                tenpai=[Atom('TENPAI' if t else 'NOTEN') for t in e['tenpais']] if 'tenpais' in e else [UNKNOWN]*4
                self.emit('EV_DRAW_END',(dict(reason=UNKNOWN,tenpai_states=tenpai,score_delta=list(e['deltas'])),),lines)
        settlement=self.session.settle(values,claims=claims,draw_facts=None if draw_result is None else draw_result['payment'])
        settlement={k:v for k,v in settlement.items() if k!='provenance'}
        settlement=canonical_settlement(settlement) if 'payments' in settlement else dict(settlement,payments=UNKNOWN)
        if draw_result is not None:
            result=draw_result
            tenpai_origins={f'result.tenpai_states.{p}':'derived' for p in range(4)} if 'tenpais' not in values[0] else {}
            if result['abort'] is not None:
                abort=dict(reason=result['abort'],actor=result['actor'],score_delta=list(values[0]['deltas']),extensions=[])
                self.emit('EV_ABORT_END',(result['abort'],result['actor'],abort),lines,origins={'result.extensions':'derived'})
            elif result['kind']=='nagashi':
                facts=dict(winners=result['winners'],payments=settlement['payments'],score_delta=list(values[0]['deltas']),
                           tenpai_states=result['tenpai'])
                origins={f'result.{p}':v for p,v in field_origins(dict(winners=result['winners'],payments=settlement['payments']),'derived').items()}
                self.emit('EV_NAGASHI',(facts,),lines,origins=dict(origins,**tenpai_origins))
            else:
                self.emit('EV_DRAW_END',(dict(reason=Atom('DRAW_EXHAUSTIVE') if result['kind']=='exhaustive' else UNKNOWN,
                    tenpai_states=result['tenpai'],score_delta=list(values[0]['deltas'])),),lines,origins=tenpai_origins)
            if result['revealed'] is not None:
                origins={k.replace('result.','reveal.'):v for k,v in tenpai_origins.items()}
                self.emit('EV_REVEAL',(dict(revealed_hands=result['revealed'],ura_indicators=None,
                    tenpai_states=result['tenpai']),),lines,origins=origins)
        self.emit('EV_SETTLEMENT',(settlement,),lines)
        result_fields=('tenpais','tehais','reason') if values[0]['type']=='ryukyoku' else ('yakus','fu','fan' if rich else 'han','pai')
        self.facts['results'].append(dict(source_refs=lines,recorded=deepcopy(values),
            unknown_fields=[k for k in result_fields if all(k not in e for e in values)]))
        if selection_audits:self.facts['results'][-1]['scoring_selections']=selection_audits
        self.facts['settlements'].append(settlement)

    def next_state(self,value,raw_line):
        """Resolve the recorded next boundary; never backfill any input history."""
        last=next(e for e in reversed(self.events) if e.kind=='EV_ROUND_END')
        from .ir import check_next_state_context
        try:
            check_next_state_context(value,self.events)
        except CleaningError as error:
            raise IRProtocolFailure(str(error)) from error
        self.events[last.event_id]=Event(last.kind,last.event_id,(value,))
        record=self.ledger[last.event_id]
        record['source_refs'].append(dict(raw_line=raw_line))
        record['field_provenance']=field_origins({'next_state':value},'recorded')
        if value['game_over'] is False:
            record['field_provenance']['next_state.next_round.rounds_played']='derived'
        else:
            record['field_provenance']['next_state.next_kyotaku']='derived'
            if value['reason']!=UNKNOWN:
                record['field_provenance']['next_state.reason']='derived'
        fact=next(f for f in self.facts['canonical_results'] if f['event_id']==last.event_id)
        fact['payload']=to_json((value,))

    def rollback_boundary(self,event_count):
        """An invalid raw event cannot leave an emitted close/win suffix behind."""
        self.events=self.events[:event_count]
        self.ledger=self.ledger[:event_count]
        self.windows=[w for w in self.windows if w['cutoff_event_id']<event_count]
        surviving={w['window_id'] for w in self.windows}
        self.ir_windows={k:w for k,w in self.ir_windows.items() if k in surviving}
        self.anchors=[[a for a in anchors if a.cutoff_event_id<event_count] for anchors in self.anchors]
        self.facts['canonical_results']=[f for f in self.facts['canonical_results'] if f['event_id']<event_count]
        self.facts['state_gaps']=[g for g in self.facts['state_gaps'] if g['event_id']<event_count]
        for name in ('furiten_windows','furiten_transitions'):
            if name in self.facts:self.facts[name]=[r for r in self.facts[name] if r['event_id']<event_count]

    def prepared(self):
        event_ids=[e.event_id for e in self.events if e.kind in VISIBLE]
        views=[]
        for viewer in range(4):
            history=[project_event(e,viewer,ankan_public=self.session.options.ankan_public)
                     for e in self.events if e.kind in VISIBLE]
            views.append(PreparedView(viewer,project_header(self.initial,viewer,self.session.options.mode),
                                      history,self.anchors[viewer],len(self.events)-1,deepcopy(self.invalid_reason)))
        # Only the small canonical result facts are serialized, not full IR payloads.
        facts=deepcopy(self.facts)
        if self.deferred_public:
            facts['unresolved_public_facts']=[dict(kind=e.value['type'],raw_line=e.raw_line) for e in self.deferred_public]
        aka_counts=self.session.tile_stock.aka_counts
        return PreparedRound(self.round_id,event_ids,self.ledger,self.windows,views,self.complete,to_json(facts),
                             None if aka_counts is None else list(aka_counts))


def __getattr__(name):
    """Resolve optional source-processing entry points only when requested."""
    if name in ('process_source', '_GameReplay'):
        from . import source_replay
        value = getattr(source_replay, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
