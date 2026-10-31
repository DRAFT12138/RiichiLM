"""Shared seat rotation and visibility projection for online and offline play."""
from mahjong.tokenizer import Atom, Event, HandMeldState

SEATS = ('SELF', 'SHIMOCHA', 'TOIMEN', 'KAMICHA')


def seat(actor, viewer):
    return Atom(SEATS[(actor-viewer)%4])


def rotate(value, viewer):
    if isinstance(value, Atom) and value.name in ('ABS_P0','ABS_P1','ABS_P2','ABS_P3'):
        return seat(int(value.name[-1]),viewer)
    if isinstance(value,list):
        return [rotate(v,viewer) for v in value]
    if isinstance(value,tuple):
        return tuple(rotate(v,viewer) for v in value)
    if isinstance(value,dict):
        return {k:rotate(v,viewer) for k,v in value.items()}
    return value


def project_event(event, viewer, *, ankan_public):
    payload=list(rotate(event.payload,viewer))
    if event.kind=='EV_DRAW' and payload[0]!=Atom('SELF'):
        payload[1]=Atom('TILE_HIDDEN')
    if event.kind=='EV_ANKAN_ATTEMPT' and payload[0]!=Atom('SELF') and ankan_public=='base_only':
        payload[2]=[Atom('TILE_HIDDEN')]*4
    return Event(event.kind,event.event_id,tuple(payload))


def project_header(initial, viewer, mode):
    return dict(mode=Atom(mode),round_wind=Atom('ROUND_'+initial['bakaze']),
                round_number=initial['kyoku'],honba=initial['honba'],kyotaku=initial['kyotaku'],
                dealer=seat(initial['oya'],viewer),
                scores=[initial['scores'][(viewer+i)%4] for i in range(4)],
                seat_winds=[Atom('TILE_'+'ESWN'[(viewer+i-initial['oya'])%4]) for i in range(4)])


def project_hand(snapshot, viewer):
    return project_current_hand(snapshot['hands'][viewer],snapshot['melds'][viewer],viewer)


def project_current_hand(hands, current_melds, viewer):
    melds=[]
    for original in current_melds:
        meld=dict(original)
        meld['name']=Atom(meld['name'])
        meld['actor']=Atom('SELF')
        meld['from']=None if meld['from'] is None else seat(meld['from'],viewer)
        for key in ('base_tile','called_tile'):
            meld[key]=None if meld[key] is None else Atom('TILE_'+meld[key])
        meld['tiles']=[Atom('TILE_'+t) for t in meld['tiles']]
        melds.append(meld)
    return HandMeldState([Atom('TILE_'+t) for t in hands],melds)
