"""Supervised legal-action samples from a published Rust text/3 cleaning build."""
import hashlib
import uuid

import torch

from mahjong.contracts import Batch, BatchLease, ComponentDescriptor, MahjongError, State, canonical_json
from mahjong.data.mahjong.corpus import open_policy_reader as CleaningReader
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, CONTEXT_BINDING, LEGACY_SCHEMA

INPUT_SCHEMA = 'majang-policy-ntp/v1'
SUPPORTED_EXECUTION = ('cpu/32-true', 'cuda:0/32-true', 'cpu/bf16-mixed', 'cuda:0/bf16-mixed')


def _fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _collate(samples, input_schema=INPUT_SCHEMA):
    size = len(samples)
    token_width = max(len(s['input_ids']) for s in samples)
    candidate_width = max(len(s['candidates']) for s in samples)
    action_width = max(len(a) for s in samples for a in s['candidates'])
    input_ids = torch.zeros((size, token_width), dtype=torch.int64)
    attention_mask = torch.zeros((size, token_width), dtype=torch.bool)
    candidate_ids = torch.zeros((size, candidate_width, action_width), dtype=torch.int64)
    targets = torch.empty(size, dtype=torch.int64)
    for row, sample in enumerate(samples):
        ids = torch.tensor(sample['input_ids'], dtype=torch.int64)
        input_ids[row, :len(ids)] = ids
        attention_mask[row, :len(ids)] = True
        for col, action in enumerate(sample['candidates']):
            candidate_ids[row, col, :len(action)] = torch.tensor(action, dtype=torch.int64)
        targets[row] = sample['target_index']
    return Batch({'input_ids': input_ids, 'attention_mask': attention_mask,
                  'candidate_ids': candidate_ids, 'target_indices': targets},
                 tuple(s['sample_id'] for s in samples), size, input_schema)


def dataset_fingerprint(reader, *, split_seed, validation_fraction, max_sequence_length, input_schema):
    binding = {'build_id': reader.catalog['build_id'], 'binding': reader.catalog['binding'],
        'shards': reader.manifest['shards'], 'split_seed': split_seed,
        'validation_fraction': float(validation_fraction),
        'max_sequence_length': max_sequence_length, 'input_schema': input_schema}
    if input_schema == CONTEXT_SCHEMA:
        binding['context'] = CONTEXT_BINDING
    return _fingerprint(binding)


class _Dataset:
    def __init__(self, build_dir, *, validation_fraction, split_seed, max_sequence_length,
                 input_schema=INPUT_SCHEMA):
        if input_schema not in (LEGACY_SCHEMA, CONTEXT_SCHEMA):
            raise MahjongError('CONFIG_INVALID', 'unsupported policy data input schema', 'preparing_data')
        self.input_schema = input_schema
        self.reader = CleaningReader(build_dir)
        binding = self.reader.catalog['binding']
        capabilities = self.reader.manifest.get('capabilities', {})
        if (binding['storage'] != 'mjai-clean-text/3' or binding.get('backend') != 'rust'
                or not capabilities.get('legal_actions') or not capabilities.get('action_labels')):
            self.reader.close()
            raise MahjongError('CONFIG_INVALID',
                'data.build_dir must be a Rust text/3 build with action labels', 'preparing_data')
        if type(validation_fraction) not in (int, float) or not 0 < validation_fraction < 1:
            self.reader.close()
            raise MahjongError('CONFIG_INVALID', 'data.validation_fraction must be between 0 and 1', 'preparing_data')
        if type(split_seed) is not int or not 0 <= split_seed < 2**63:
            self.reader.close()
            raise MahjongError('CONFIG_INVALID', 'data.split_seed must be in [0,2**63)', 'preparing_data')
        if type(max_sequence_length) is not int or max_sequence_length < 2:
            self.reader.close()
            raise MahjongError('CONFIG_INVALID', 'model context length must be >= 2', 'preparing_data')
        self.validation_fraction = float(validation_fraction)
        self.split_seed = split_seed
        self.max_sequence_length = max_sequence_length
        self.games = []
        for row in self.reader.iter_games():
            identity = row['game_id'] or row['native_id'] or row['source_key']
            self.games.append((row['source_key'], identity))
        if len(self.games) < 2:
            self.reader.close()
            raise MahjongError('INSUFFICIENT_DATA',
                'policy training requires at least two games for a held-out validation split', 'preparing_data')
        scores = [int.from_bytes(hashlib.sha256(
            f'{split_seed}\0{source}\0{game}'.encode()).digest()[:8], 'big') / 2**64
            for source, game in self.games]
        self.validation_mask = [score < self.validation_fraction for score in scores]
        if not any(self.validation_mask):
            self.validation_mask[min(range(len(scores)), key=scores.__getitem__)] = True
        elif all(self.validation_mask):
            self.validation_mask[max(range(len(scores)), key=scores.__getitem__)] = False
        self.fingerprint = dataset_fingerprint(self.reader, split_seed=split_seed,
            validation_fraction=self.validation_fraction, max_sequence_length=max_sequence_length,
            input_schema=input_schema)

    def split_indices(self, validation):
        return torch.tensor([i for i, selected in enumerate(self.validation_mask)
            if selected == validation],
            dtype=torch.int64)

    def samples_for_game(self, index, *, shared_parts=False):
        source_key, game_id = self.games[index]
        samples = []
        refs = list(self.reader.iter_round_refs(source_key))
        context_cache = {}
        round_ids = [r for _, r in refs]
        for source, round_id in refs:
            record = self.reader.read_round(source, round_id)
            for decision in record['facts'].get('policy_decisions', []):
                if shared_parts:
                    from mahjong.policy.context import split_policy_prefix
                    persistent, private, _ = split_policy_prefix(self.reader, source, round_id,
                        decision['viewer'], decision['window_id'], round_ids=round_ids,
                        context_cache=context_cache)
                    prefix = persistent + private
                elif self.input_schema == CONTEXT_SCHEMA:
                    from mahjong.policy.context import read_context_prefix
                    prefix, _ = read_context_prefix(self.reader, source, round_id,
                        decision['viewer'], decision['window_id'], round_ids=round_ids,
                        context_cache=context_cache)
                else:
                    prefix, _ = self.reader.read_prefix(source, round_id, decision['viewer'], decision['window_id'])
                if len(prefix) + max(map(len, decision['candidates'])) <= self.max_sequence_length:
                    samples.append({'input_ids': prefix, 'candidates': decision['candidates'],
                        'target_index': decision['target_index'],
                        'sample_id': f'{source}/{game_id}/{round_id}/{decision["window_id"]}/{decision["viewer"]}'})
                    if shared_parts:
                        samples[-1].update(cut_length=len(persistent),
                            group_key=(source, round_id, decision['viewer']), window_id=decision['window_id'])
                elif self.input_schema == CONTEXT_SCHEMA:
                    raise MahjongError('CONTEXT_TOO_LONG',
                        'policy prefix plus longest legal action exceeds context; no history was truncated',
                        'preparing_data', {'source': source, 'round_id': round_id,
                        'window_id': decision['window_id'], 'viewer': decision['viewer'],
                        'prefix_length': len(prefix), 'longest_suffix': max(map(len, decision['candidates'])),
                        'context': self.max_sequence_length})
        return samples

    def close(self): self.reader.close()


class MahjongPolicyTrainSource:
    def __init__(self, *, build_dir, seed, batch_size=8, validation_fraction=0.02,
                 split_seed=20261018, max_sequence_length=2048, shuffle=True, input_schema=INPUT_SCHEMA):
        if type(seed) is not int or not 0 <= seed < 2**63 or type(batch_size) is not int or batch_size < 1:
            raise MahjongError('CONFIG_INVALID', 'invalid policy sampler seed/batch size', 'preparing_data')
        if type(shuffle) is not bool:
            raise MahjongError('CONFIG_INVALID', 'data.shuffle must be boolean', 'preparing_data')
        self.dataset = _Dataset(build_dir, validation_fraction=validation_fraction,
            split_seed=split_seed, max_sequence_length=max_sequence_length, input_schema=input_schema)
        self.input_schema = input_schema
        self.indices = self.dataset.split_indices(False)
        if self.indices.numel() == 0:
            self.dataset.close()
            raise MahjongError('EMPTY_TRAIN_DATA', 'no games assigned to training', 'preparing_data')
        self.batch_size, self.shuffle = batch_size, shuffle
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.epoch = self.game_cursor = self.sample_cursor = 0
        self.permutation = self._new_permutation()
        self._lease = self._pending = None
        self._cached_game = self._cached_samples = None
        fingerprint = _fingerprint({'data': self.dataset.fingerprint, 'seed': seed,
                                    'batch_size': batch_size, 'shuffle': shuffle})
        self._descriptor = ComponentDescriptor('data','majang.mahjong_policy_train/v1',1,
            input_schema,fingerprint,self.dataset.fingerprint,1,'committed_batch',SUPPORTED_EXECUTION)

    def _new_permutation(self, generator=None):
        return (torch.randperm(len(self.indices), generator=generator if generator is not None else self.generator) if self.shuffle
                else torch.arange(len(self.indices), dtype=torch.int64))

    def descriptor(self): return self._descriptor

    def _get_samples(self, position, permutation):
        game_index = int(self.indices[int(permutation[position])])
        if self._cached_game != game_index:
            self._cached_game = game_index
            self._cached_samples = self.dataset.samples_for_game(game_index)
        return self._cached_samples

    def acquire(self):
        if self._lease is not None:
            raise MahjongError('CONTRACT_MISMATCH','policy lease already open','training')
        position, offset = self.game_cursor, self.sample_cursor
        epoch, permutation = self.epoch, self.permutation
        generator = torch.Generator(device='cpu')
        generator.set_state(self.generator.get_state())
        samples, empty_games = [], 0
        while len(samples) < self.batch_size:
            if position >= len(self.indices):
                position, offset = 0, 0
                epoch += 1
                permutation = self._new_permutation(generator)
                self._cached_game = self._cached_samples = None
                # Empty games can straddle two independently shuffled epochs.
                empty_games = 0
            group = self._get_samples(position, permutation)
            if not group or offset >= len(group):
                position, offset = position + 1, 0
                empty_games += 1
                if empty_games >= len(self.indices):
                    raise MahjongError('EMPTY_TRAIN_DATA','no labelled decisions in training split','preparing_data')
                continue
            take = min(self.batch_size-len(samples),len(group)-offset)
            samples.extend(group[offset:offset+take])
            offset += take
            empty_games = 0
            if offset == len(group):
                position, offset = position + 1, 0
                self._cached_game = self._cached_samples = None
        self._lease = BatchLease(_collate(samples, self.input_schema), uuid.uuid4().hex)
        self._pending = (position, offset, epoch, permutation, generator)
        return self._lease

    def commit(self, token):
        if self._lease is None or self._lease.token != token:
            raise MahjongError('CONTRACT_MISMATCH','wrong/repeated policy lease token','training')
        (self.game_cursor, self.sample_cursor, self.epoch,
         self.permutation, self.generator) = self._pending
        self._lease = self._pending = None

    def has_open_lease(self): return self._lease is not None

    def state_dict(self) -> State:
        return {'version':1,'data_fingerprint':self.dataset.fingerprint,'epoch':self.epoch,
            'game_cursor':self.game_cursor,'sample_cursor':self.sample_cursor,
            'permutation':self.permutation.clone(),'sampler_rng':self.generator.get_state(),
            'shuffle':self.shuffle}

    def load_state_dict(self, state):
        required={'version','data_fingerprint','epoch','game_cursor','sample_cursor',
                  'permutation','sampler_rng','shuffle'}
        if self._lease is not None or type(state) is not dict or set(state) != required:
            raise MahjongError('CHECKPOINT_INVALID','invalid policy sampler state fields','restoring')
        if state['version'] != 1 or state['data_fingerprint'] != self.dataset.fingerprint or state['shuffle'] != self.shuffle:
            raise MahjongError('CHECKPOINT_INVALID','policy data/sampler binding changed','restoring')
        permutation=state['permutation']
        if (not isinstance(permutation,torch.Tensor) or permutation.dtype != torch.int64 or
                permutation.device.type != 'cpu' or permutation.shape != (len(self.indices),) or
                not torch.equal(permutation.sort().values,torch.arange(len(self.indices)))):
            raise MahjongError('CHECKPOINT_INVALID','invalid policy game permutation','restoring')
        for key in ('epoch','game_cursor','sample_cursor'):
            if type(state[key]) is not int or state[key] < 0:
                raise MahjongError('CHECKPOINT_INVALID',f'invalid policy sampler {key}','restoring')
        if state['game_cursor'] > len(self.indices):
            raise MahjongError('CHECKPOINT_INVALID','policy game cursor outside split','restoring')
        generator=torch.Generator(device='cpu')
        try: generator.set_state(state['sampler_rng'])
        except (RuntimeError,TypeError) as exc:
            raise MahjongError('CHECKPOINT_INVALID','invalid policy sampler RNG','restoring') from exc
        self.epoch,self.game_cursor,self.sample_cursor=state['epoch'],state['game_cursor'],state['sample_cursor']
        self.permutation,self.generator=permutation.clone(),generator
        self._cached_game=self._cached_samples=None

    def close(self): self.dataset.close()


class MahjongPolicyEvalSource:
    def __init__(self, *, build_dir, batch_size=8, validation_fraction=0.02,
                 split_seed=20261018, max_sequence_length=2048, max_samples=None, input_schema=INPUT_SCHEMA,
                 layout=None):
        if type(batch_size) is not int or batch_size < 1:
            raise MahjongError('CONFIG_INVALID','evaluation.batch_size must be positive','preparing_data')
        if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
            raise MahjongError('CONFIG_INVALID','data.validation_max_samples must be positive or null','preparing_data')
        self.dataset=_Dataset(build_dir,validation_fraction=validation_fraction,
            split_seed=split_seed,max_sequence_length=max_sequence_length,input_schema=input_schema)
        self.input_schema = input_schema
        self.indices=self.dataset.split_indices(True)
        self.batch_size=batch_size
        self.max_samples=max_samples
        self.layout=layout
        fingerprint=_fingerprint({'data':self.dataset.fingerprint,'batch_size':batch_size,
                                  'split':'validation','max_samples':max_samples,
                                  **({'layout':layout} if layout is not None else {})})
        self._descriptor=ComponentDescriptor('data','majang.mahjong_policy_eval/v1',1,
            input_schema,fingerprint,self.dataset.fingerprint,1,'committed_batch',SUPPORTED_EXECUTION)

    def descriptor(self): return self._descriptor

    def _collate(self, samples):
        batch = _collate(samples, self.input_schema)
        if self.layout is None:
            return batch
        from mahjong.policy.shared_context import INDEPENDENT_LAYOUT
        if self.layout != INDEPENDENT_LAYOUT:
            raise MahjongError('CONTRACT_MISMATCH', 'unsupported evaluation layout', 'evaluating')
        return Batch({**batch.payload, 'layout':self.layout}, batch.sample_ids,
                     batch.sample_count, batch.input_schema)

    def iter_batches(self):
        samples=[]
        produced=False
        selected=0
        for game_index in self.indices:
            group=self.dataset.samples_for_game(int(game_index))
            if self.max_samples is not None:
                group=group[:self.max_samples-selected]
            samples.extend(group)
            selected+=len(group)
            while len(samples)>=self.batch_size:
                current,samples=samples[:self.batch_size],samples[self.batch_size:]
                produced=True
                yield self._collate(current)
            if self.max_samples is not None and selected >= self.max_samples:
                break
        if samples:
            produced=True
            yield self._collate(samples)
        if not produced:
            raise MahjongError('EMPTY_VALIDATION_DATA',
                'no labelled decisions in validation split', 'evaluating')

    def close(self): self.dataset.close()
