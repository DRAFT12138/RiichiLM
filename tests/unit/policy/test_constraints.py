"""Finite legal-set invariant and normalized leaf mass, independent of models."""
import math
import random

import pytest

from mahjong.contracts import MahjongError
from mahjong.policy.constraints import build_trie


def test_random_finite_tries_preserve_exact_successors_and_total_leaf_mass():
    rng=random.Random(20261020)
    for _ in range(40):
        actions=sorted({tuple(rng.randrange(1,20) for _ in range(rng.randrange(1,12)))+(30,)
                        for _ in range(rng.randrange(1,35))})
        trie=build_trie(actions)
        total=0.
        for index,action in enumerate(actions):
            node=trie;prefix=();probability=1.
            for token in action:
                allowed={a[len(prefix)] for a in actions if a[:len(prefix)]==prefix}
                assert set(node)==allowed
                logits={t:math.sin(t+len(prefix))*3 for t in allowed}
                maximum=max(logits.values())
                weights={t:math.exp(z-maximum) for t,z in logits.items()}
                probability*=weights[token]/sum(weights.values())
                prefix+=(token,);node=node[token]
            assert node=={None:index}
            total+=probability
        assert total==pytest.approx(1.,abs=1e-12)


@pytest.mark.parametrize('actions',[[],[[1],[1]],[[1],[1,2]],[[1,2],[1]], [[]]])
def test_empty_duplicate_and_prefix_collisions_are_rejected(actions):
    with pytest.raises(MahjongError):build_trie(actions)
