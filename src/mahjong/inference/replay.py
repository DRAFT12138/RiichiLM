"""Bounded validation from published four-view IR, without original archives.

Only a round with all four initial private anchors can be reconstructed. The
wall is a compatible completion of frozen draws/indicators, not a claim to have
recovered the source's unknown wall. Future facts are used by this validation
driver; the bridge still projects only facts already reached by native step.
"""
from collections import defaultdict

from mahjong.envs.policy_bridge import RiichiDecisionBridge
from mahjong.tokenizer import Action, Atom, Event, MahjongTokenizer

RELATIVE = ('SELF','SHIMOCHA','TOIMEN','KAMICHA')


def absolute(value, viewer):
    if isinstance(value, Atom) and value.name in RELATIVE:
        return Atom('ABS_P'+str((RELATIVE.index(value.name)+viewer)%4))
    if isinstance(value, list):
        return [absolute(v,viewer) for v in value]
    if isinstance(value, tuple):
        return tuple(absolute(v,viewer) for v in value)
    if isinstance(value, dict):
        return {k:absolute(v,viewer) for k,v in value.items()}
    return value


class FrozenRoundReplay:
    def __init__(self, reader, source, round_id, profile, *, model_binding_digest):
        from riichienv.convert import tid_to_mjai
        self.codec = MahjongTokenizer()
        self.record = reader.read_round(source, round_id)
        self.source, self.round_id = source, round_id
        self.events = {}
        initial = {}
        header = None
        for viewer in range(4):
            view = reader.read_view(source, round_id, viewer)
            a = view.arrays
            if not a['anchors']:
                raise ValueError('initial private anchor is missing')
            window = a['anchors'][0]
            prefix, _ = reader.read_prefix(source,round_id,viewer,window)
            sequence = self.codec.decode_policy_prefix(prefix)
            hand = [t.name.removeprefix('TILE_') for t in sequence.hand_state.hands]
            if sequence.hand_state.melds:
                raise ValueError('initial anchor already has a meld')
            h = absolute(sequence.header,viewer)
            dealer = int(h['dealer'].name[-1])
            if viewer == dealer:
                own_draws = [e for e in sequence.visible_history if e.kind=='EV_DRAW' and e.payload[0]==Atom('SELF')]
                if len(own_draws) != 1:
                    raise ValueError('initial dealer draw is unavailable')
                hand.remove(own_draws[0].payload[1].name.removeprefix('TILE_'))
            elif any(e.kind=='EV_DRAW' and e.payload[0]==Atom('SELF') for e in sequence.visible_history):
                raise ValueError('initial private anchor follows a private draw')
            if len(hand) != 13:
                raise ValueError('initial private hand is not reconstructable')
            initial[viewer] = hand
            if viewer == 0:
                header = h
            offsets = a['history_offsets']
            for lo,hi in zip(offsets, offsets[1:]):
                event = self.codec.decode_event(a['history'][lo:hi])
                event = Event(event.kind,event.event_id,absolute(event.payload,viewer))
                # Every actor's own view is authoritative for private payloads.
                actor = event.payload[0] if event.kind in ('EV_DRAW','EV_ANKAN_ATTEMPT') else None
                if event.event_id not in self.events or actor == Atom('ABS_P'+str(viewer)):
                    self.events[event.event_id] = event
        # Freeze outcomes separately; they never become policy HISTORY.
        self.outcomes = self.record['facts'].get('outcomes', [])
        if header['round_wind'] != Atom('ROUND_E') or header['round_number'] != 1 or header['honba'] != 0:
            raise ValueError('standalone replay requires the initial round')
        pool = defaultdict(list)
        for t in range(136):
            pool[tid_to_mjai(t)].append(t)
        def take(name):
            if not pool[name]:
                raise ValueError('frozen round exceeds tile inventory: '+name)
            return pool[name].pop()
        hands = {p:[take(t) for t in hand] for p,hand in initial.items()}
        order = [None]*136
        cursor = 0
        for chunk in range(3):
            for relative in range(4):
                p=(dealer+relative)%4
                for t in hands[p][4*chunk:4*chunk+4]:
                    order[cursor]=t; cursor+=1
        for relative in range(4):
            order[cursor]=hands[(dealer+relative)%4][12]; cursor+=1
        rinshan = dora = 0
        for event in sorted(self.events.values(),key=lambda e:e.event_id):
            if event.kind=='EV_DRAW':
                _,tile,kind = event.payload
                if tile == Atom('TILE_HIDDEN'):
                    raise ValueError('private draw is unavailable')
                if kind==Atom('DRAW_RINSHAN'):
                    slot=135-rinshan;rinshan+=1
                else:
                    slot=cursor;cursor+=1
                if order[slot] is not None:
                    raise ValueError('frozen wall slot collision')
                order[slot]=take(tile.name.removeprefix('TILE_'))
            elif event.kind=='EV_DORA_REVEAL':
                slot=131-2*dora;dora+=1
                if order[slot] is not None:
                    raise ValueError('frozen indicator slot collision')
                order[slot]=take(event.payload[0].name.removeprefix('TILE_'))
        rest = [t for values in pool.values() for t in values]
        for slot in range(136):
            if order[slot] is None:
                order[slot]=rest.pop()
        self.bridge = RiichiDecisionBridge(profile,model_binding_digest=model_binding_digest)
        self.bridge._reset(wall=order,oya=dealer,seed=0)

    def recorded_action(self, view):
        trigger = view.key.window.window_id//4
        viewer = view.key.viewer
        slot = view.key.window.window_id%4
        future = [e for e in sorted(self.events.values(),key=lambda e:e.event_id) if e.event_id > trigger]
        if slot in (2,3):
            outcome = next((o for o in self.outcomes if o['kind']=='EV_WIN' and o['payload'][0]['trigger_id']==trigger),None)
            if outcome:
                winner=next((w for w in outcome['payload'][0]['winners'] if w['actor']=={'enum':'ABS_P'+str(viewer)}),None)
                if winner:
                    target=int(winner['target']['enum'][-1])
                    return Action('A_RON',(Atom(RELATIVE[(target-viewer)%4]),Atom(winner['tile']['enum'])))
            for event in future:
                if event.kind in ('EV_DRAW','EV_DISCARD','EV_ANKAN_ATTEMPT','EV_KAKAN_ATTEMPT'):
                    break
                if event.kind in ('EV_CHI','EV_PON','EV_DAIMINKAN'):
                    actor,target,tile,consumed,_,reference=event.payload
                    if reference==trigger and actor==Atom('ABS_P'+str(viewer)):
                        return Action('A_'+event.kind.removeprefix('EV_'),
                            (Atom(RELATIVE[(int(target.name[-1])-viewer)%4]),tile,*consumed))
                    break
            return Action('A_PASS')
        declared=False
        for event in future:
            if event.kind=='EV_RIICHI_DECLARE':
                declared=True
            elif event.kind=='EV_DISCARD':
                actor,tile,origin,_=event.payload
                if actor != Atom('ABS_P'+str(viewer)):
                    raise ValueError('recorded turn actor differs')
                return Action('A_RIICHI_DISCARD' if declared else 'A_DISCARD',(tile,origin))
            elif event.kind=='EV_ANKAN_ATTEMPT':
                return Action('A_ANKAN',tuple(event.payload[2]))
            elif event.kind=='EV_KAKAN_ATTEMPT':
                return Action('A_KAKAN',(event.payload[2],event.payload[1]))
        outcome=next((o for o in self.outcomes if o['kind']=='EV_WIN' and o['payload'][0]['trigger_id']==trigger),None)
        if outcome:
            return Action('A_TSUMO_WIN')
        raise ValueError('recorded turn outcome is unavailable')

    def advance_recorded(self):
        """The validation driver supplies source intentions through exact plans.

        This private path deliberately does not forge model probabilities. The
        public model-action path is exercised separately on forks of each point.
        """
        from riichienv import Phase
        b=self.bridge
        views=[b.snapshot(p) for p in range(4) if b.snapshot(p) is not None]
        actions={}
        for view in views:
            action=self.recorded_action(view)
            move=next((m for m in view.legal_moves if m.action==action),None)
            if move is None:
                raise ValueError('recorded target is not in current complete legal set: '+repr(action))
            actions[view.key.viewer]=b._plans[(view.key,move.execution_ref)]
        if b._native.phase==Phase.WaitResponse:
            return b._advance([{p:plan[0] for p,plan in actions.items()}])
        player=views[0].key.viewer
        return b._advance([{player:a} for a in actions[player]])
