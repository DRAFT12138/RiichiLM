"""Explicit legal inventories, packed into the native 4/4/4/1 deal order."""
from collections import defaultdict
import random

from riichienv.convert import tid_to_mjai


def wall_for(hands, *, drawn, oya=0, draws=()):
    pool = defaultdict(list)
    for tid in range(136):
        pool[tid_to_mjai(tid)].append(tid)
    def take(name):
        assert pool[name], f'fifth tile in fixture: {name}'
        return pool[name].pop()
    explicit = {p:[take(t) for t in tiles] for p,tiles in hands.items()}
    first = take(drawn)
    later = [take(t) for t in draws]
    rest = [t for tiles in pool.values() for t in tiles]
    random.Random(999).shuffle(rest)
    for p in range(4):
        explicit.setdefault(p,[])
        assert len(explicit[p]) <= 13
        explicit[p].extend(rest.pop() for _ in range(13-len(explicit[p])))
    order = []
    for chunk in range(3):
        for i in range(4):
            order.extend(explicit[(i+oya)%4][4*chunk:4*chunk+4])
    order.extend(explicit[(i+oya)%4][12] for i in range(4))
    order.extend([first,*later,*rest])
    assert len(order) == len(set(order)) == 136
    return order
