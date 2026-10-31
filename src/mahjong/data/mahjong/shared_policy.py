"""Deterministic same-round, same-view policy groups with committed recovery."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import uuid

import torch

from mahjong.contracts import Batch, BatchLease, MahjongError
from mahjong.policy.shared_context import PLANNER_VERSION, SHARED_LAYOUT
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
from .policy import MahjongPolicyTrainSource, _fingerprint


def plan_shared_groups(samples, *, max_decisions, max_tokens, max_branch_cells):
    """Keep every decision, splitting before any deterministic budget is exceeded."""
    for value in (max_decisions, max_tokens, max_branch_cells):
        if type(value) is not int or value < 1:
            raise MahjongError('CONFIG_INVALID', 'shared group budgets must be positive integers', 'preparing_data')
    ordered = sorted(samples, key=lambda s: (*s['group_key'], s['window_id']))
    groups, group, branches = [], [], 0
    for sample in ordered:
        cut = sample['cut_length']
        branch = len(sample['input_ids']) - cut + len(sample['candidates'][sample['target_index']])
        if cut < 1 or cut >= len(sample['input_ids']):
            raise MahjongError('CONTRACT_MISMATCH', 'invalid persistent prefix boundary', 'preparing_data')
        if cut + branch > max_tokens or branch * (cut + branch) > max_branch_cells:
            raise MahjongError('CONFIG_INVALID', 'shared budgets cannot hold a single decision',
                               'preparing_data', {'sample_id':sample['sample_id']})
        if group:
            previous = group[-1]
            nested = (cut >= previous['cut_length'] and
                sample['input_ids'][:previous['cut_length']] == previous['input_ids'][:previous['cut_length']])
            total = branches + branch
            if (sample['group_key'] != previous['group_key'] or not nested or
                    len(group) >= max_decisions or cut + total > max_tokens or
                    total * (cut + total) > max_branch_cells):
                groups.append(group)
                group, branches = [], 0
        group.append(sample)
        branches += branch
    if group:
        groups.append(group)
    return groups


def collate_shared_group(samples):
    if not samples:
        raise MahjongError('CONTRACT_MISMATCH', 'empty shared group', 'preparing_data')
    trunk = samples[-1]['input_ids'][:samples[-1]['cut_length']]
    private, offsets, cuts = [], [0], []
    width = max(len(s['candidates']) for s in samples)
    action_width = max(len(a) for s in samples for a in s['candidates'])
    candidates = torch.zeros((len(samples), width, action_width), dtype=torch.int64)
    targets = []
    for row, sample in enumerate(samples):
        cut = sample['cut_length']
        if (sample['group_key'] != samples[0]['group_key'] or sample['input_ids'][:cut] != trunk[:cut]):
            raise MahjongError('CONTRACT_MISMATCH', 'shared prefixes are not nested in one round/view', 'preparing_data')
        private.extend(sample['input_ids'][cut:])
        offsets.append(len(private)); cuts.append(cut); targets.append(sample['target_index'])
        for col, action in enumerate(sample['candidates']):
            candidates[row, col, :len(action)] = torch.tensor(action, dtype=torch.int64)
    ids = tuple(s['sample_id'] for s in samples)
    if len(set(ids)) != len(ids):
        raise MahjongError('CONTRACT_MISMATCH', 'duplicate decision in shared group', 'preparing_data')
    return Batch({'layout':SHARED_LAYOUT, 'trunk_ids':torch.tensor(trunk, dtype=torch.int64),
        'private_ids':torch.tensor(private, dtype=torch.int64),
        'branch_offsets':torch.tensor(offsets, dtype=torch.int64),
        'cut_lengths':torch.tensor(cuts, dtype=torch.int64), 'candidate_ids':candidates,
        'target_indices':torch.tensor(targets, dtype=torch.int64)}, ids, len(samples), CONTEXT_SCHEMA)


class MahjongSharedPolicyTrainSource(MahjongPolicyTrainSource):
    def __init__(self, *, max_tokens=8192, max_branch_cells=4194304, **kwargs):
        if kwargs.get('input_schema', CONTEXT_SCHEMA) != CONTEXT_SCHEMA:
            raise MahjongError('CONFIG_INVALID', 'shared policy requires the cross-round v2 schema', 'preparing_data')
        kwargs['input_schema'] = CONTEXT_SCHEMA
        for value in (max_tokens, max_branch_cells):
            if type(value) is not int or value < 1:
                raise MahjongError('CONFIG_INVALID', 'shared budgets must be positive integers', 'preparing_data')
        super().__init__(**kwargs)
        self.max_tokens, self.max_branch_cells = max_tokens, max_branch_cells
        self.planning_fingerprint = _fingerprint({'planner':PLANNER_VERSION, 'layout':SHARED_LAYOUT,
            'max_decisions':self.batch_size, 'max_tokens':max_tokens, 'max_branch_cells':max_branch_cells})
        self._descriptor = replace(self._descriptor, implementation_id='majang.mahjong_shared_policy_train/v1',
            state_version=1, config_fingerprint=_fingerprint({'base':self._descriptor.config_fingerprint,
                                                           'planning':self.planning_fingerprint}))
        self._prefetch_executor = self._prefetch_future = self._prefetch_game = None
        self._batch_preview = None

    def supports_batch_preview(self):
        return (type(self) is MahjongSharedPolicyTrainSource and all(
            getattr(getattr(self, name), '__func__', None) is original
            for name, original in _PREVIEW_METHODS.items()))

    def discard_batch_preview(self):
        self._batch_preview = None

    def preview_next_batch(self):
        """CPU-only peek after an open lease; never advances committed state."""
        if self._lease is None:
            raise MahjongError('CONTRACT_MISMATCH', 'preview requires an open lease', 'training')
        if self._batch_preview is None:
            position, offset, epoch, permutation, pending_generator = self._pending
            generator = torch.Generator(device='cpu')
            generator.set_state(pending_generator.get_state())
            try:
                self._batch_preview = self._next_batch(position, offset, epoch, permutation, generator)
            except Exception as error:
                self._batch_preview = error.with_traceback(None)
        if isinstance(self._batch_preview, Exception):
            raise self._batch_preview
        return self._batch_preview[0]

    def configure_prefetch(self, enabled):
        """One CPU-only future game; sampler state remains owned by acquire."""
        if enabled and self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1,
                thread_name_prefix='mahjong-game-prefetch')
        elif not enabled and self._prefetch_executor is not None:
            # shutdown waits for reader ownership, but never observes an unused
            # future exception (close must not replace the training outcome).
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = self._prefetch_future = self._prefetch_game = None

    def _prepare_game(self, game_index):
        samples = self.dataset.samples_for_game(game_index, shared_parts=True)
        return plan_shared_groups(samples, max_decisions=self.batch_size,
            max_tokens=self.max_tokens, max_branch_cells=self.max_branch_cells)

    def _get_samples(self, position, permutation):
        game_index = int(self.indices[int(permutation[position])])
        if self._cached_game != game_index:
            groups = None
            if self._prefetch_future is not None:
                if self._prefetch_game == game_index:
                    future = self._prefetch_future
                    self._prefetch_future = self._prefetch_game = None
                    # Only actual consumption observes a future data error.
                    groups = future.result()
                else:
                    # A changed permutation/state must hand back the shared
                    # reader before any synchronous read.
                    self.configure_prefetch(False)
                    self.configure_prefetch(True)
            if groups is None:
                groups = self._prepare_game(game_index)
            self._cached_game, self._cached_samples = game_index, groups
        if (self._prefetch_executor is not None and self._prefetch_future is None
                and position+1 < len(self.indices)):
            self._prefetch_game = int(self.indices[int(permutation[position+1])])
            self._prefetch_future = self._prefetch_executor.submit(
                self._prepare_game, self._prefetch_game)
        return self._cached_samples

    def acquire(self):
        if self._lease is not None:
            raise MahjongError('CONTRACT_MISMATCH', 'policy lease already open', 'training')
        preview, self._batch_preview = self._batch_preview, None
        if isinstance(preview, Exception):
            raise preview
        if preview is None:
            generator = torch.Generator(device='cpu')
            generator.set_state(self.generator.get_state())
            preview = self._next_batch(self.game_cursor, self.sample_cursor, self.epoch,
                                       self.permutation, generator)
        batch, self._pending = preview
        self._lease = BatchLease(batch, uuid.uuid4().hex)
        return self._lease

    def _next_batch(self, position, offset, epoch, permutation, generator):
        """Shared acquire/preview algorithm; generator belongs to this request."""
        empty_games = 0
        while True:
            if position >= len(self.indices):
                position, offset, epoch = 0, 0, epoch + 1
                permutation = self._new_permutation(generator)
                self._cached_game = self._cached_samples = None
                empty_games = 0
            groups = self._get_samples(position, permutation)
            if offset > len(groups):
                raise MahjongError('CHECKPOINT_INVALID', 'shared group cursor outside game', 'restoring')
            if offset == len(groups):
                position, offset = position + 1, 0
                empty_games += 1
                if empty_games >= len(self.indices):
                    raise MahjongError('EMPTY_TRAIN_DATA', 'no labelled decisions in training split', 'preparing_data')
                continue
            batch = collate_shared_group(groups[offset])
            offset += 1
            if offset == len(groups):
                position, offset = position + 1, 0
                self._cached_game = self._cached_samples = None
            return batch, (position, offset, epoch, permutation, generator)

    def state_dict(self):
        state = super().state_dict()
        state.update(version=1, planning_fingerprint=self.planning_fingerprint)
        state['group_cursor'] = state.pop('sample_cursor')
        return state

    def load_state_dict(self, state):
        self.discard_batch_preview()
        prefetch = self._prefetch_executor is not None
        self.configure_prefetch(False)
        expected = {'version','data_fingerprint','epoch','game_cursor','group_cursor',
                    'permutation','sampler_rng','shuffle','planning_fingerprint'}
        if (type(state) is not dict or state.keys() != expected or state['version'] != 1 or
                state['planning_fingerprint'] != self.planning_fingerprint):
            raise MahjongError('CHECKPOINT_INVALID', 'shared sampler layout/planning binding changed', 'restoring')
        translated = {k:v for k,v in state.items() if k not in ('planning_fingerprint','group_cursor')}
        translated.update(version=1, sample_cursor=state['group_cursor'])
        super().load_state_dict(translated)
        if self.game_cursor == len(self.indices):
            if self.sample_cursor != 0:
                raise MahjongError('CHECKPOINT_INVALID', 'nonzero shared cursor at epoch end', 'restoring')
        elif self.sample_cursor >= len(self._get_samples(self.game_cursor, self.permutation)) and self.sample_cursor != 0:
            raise MahjongError('CHECKPOINT_INVALID', 'shared group cursor outside game', 'restoring')
        if prefetch:
            self.configure_prefetch(True)

    def close(self):
        self.discard_batch_preview()
        self.configure_prefetch(False)
        super().close()


_PREVIEW_METHODS = {name: getattr(MahjongSharedPolicyTrainSource, name) for name in
                    ('acquire', 'commit', '_next_batch', '_get_samples', '_prepare_game',
                     '_new_permutation', 'preview_next_batch', 'discard_batch_preview')}
