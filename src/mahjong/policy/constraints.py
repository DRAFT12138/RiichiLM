"""Prefix-free complete-action trie shared by training and inference."""
from mahjong.contracts import MahjongError


def build_trie(actions):
    root = {}
    if not actions:
        raise MahjongError('LEGAL_SET_EMPTY', 'legal action set is empty')
    for index, action in enumerate(actions):
        if not action:
            raise MahjongError('CONTRACT_MISMATCH', 'empty legal action')
        node = root
        for token in action:
            if None in node:
                raise MahjongError('CONTRACT_MISMATCH', 'legal action is a prefix of another')
            node = node.setdefault(token, {})
        if node:
            raise MahjongError('CONTRACT_MISMATCH', 'duplicate/prefix legal action')
        node[None] = index
    return root


def deterministic_tail(node):
    """Whether exactly one complete suffix remains, including its terminator.

    A single successor alone is insufficient: another branch may follow it.
    """
    while None not in node:
        if len(node) != 1:
            return False
        node = next(iter(node.values()))
    return True
