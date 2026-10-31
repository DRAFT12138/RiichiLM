"""One finite, checkpoint-owned lease/cursor over immutable episode offsets."""
import random
import uuid
from mahjong import contracts as c
from mahjong.inference.policy import digest
from mahjong.models.qwen3_rl import SCHEMA
from mahjong.rl.storage import EpisodeReader
from mahjong.rl.targets import TARGET_SCHEMA


class RLTrainSource:
    def __init__(self, run_dir, config, origin):
        self.run_dir, self.config, self.origin = run_dir, config, origin
        self.state = dict(stage='WARMUP', phase='READY', iteration_id=0, episode_seq=0,
                          pool=None, refs=[], validation_refs=[], target_bundle=None, cursor=0, gate_windows=0,
                          actor_updates=0, critic_updates=0)
        self._lease = None
        self.readers, self.order = {}, []

    def descriptor(self):
        fingerprint = digest(dict(config=self.config, origin=self.origin))
        return c.ComponentDescriptor('data', 'mahjong.rl_source/v1', 1, SCHEMA, fingerprint,
            fingerprint, None, 'committed_batch', ('cpu/32-true', 'cuda:0/32-true', 'cuda:0/bf16-mixed'))

    def prepare(self, refs, bundle):
        if self.has_open_lease() or self.state['phase'] != 'READY':
            raise c.MahjongError('CONTRACT_MISMATCH', 'prepare requires READY')
        state = c.clone_state(self.state)
        state.update(phase='PREPARED', refs=refs, target_bundle=bundle, cursor=0)
        self.load_state_dict(state)

    def _rebuild(self):
        self.readers = {r['path']: EpisodeReader(self.run_dir, r, self.config['collection']) for r in self.state['refs']}
        self.order = []
        if self.state['phase'] == 'READY':
            if self.state['target_bundle'] is not None or self.state['refs'] or self.state['cursor']:
                raise c.MahjongError('CHECKPOINT_INVALID', 'READY contains consumable data')
            return
        bundle = self.state['target_bundle']
        if (bundle['schema'] != TARGET_SCHEMA or bundle['stage'] != self.state['stage']
                or bundle['binding'] != digest(bundle['targets'])
                or bundle['old_actor_ref'] != self.state['pool']['latest']
                or bundle['bc'] != self.state['pool']['bc']):
            raise c.MahjongError('CHECKPOINT_INVALID', 'target schema/binding mismatch')
        if len(c.canonical_json(bundle).encode()) > self.config['collection']['max_target_bytes']:
            raise c.MahjongError('CHECKPOINT_INVALID', 'target budget exceeded')
        targets = bundle['targets']
        identities = set()
        for t in targets:
            identity = (t['trajectory'], t['key'])
            reader = self.readers[t['trajectory']]
            r = reader.read(t['key'])
            if (identity in identities or reader.ref['sha256'] != t['trajectory_sha'] or r['decision_key'] != t['decision_key']
                    or r['viewer'] != reader.header['learner'] or r['oracle_key'] != r['decision_key']):
                raise c.MahjongError('CHECKPOINT_INVALID', 'target decision provenance mismatch')
            identities.add(identity)
        if not targets:
            raise c.MahjongError('EMPTY_TRAIN_DATA', 'no learner targets')
        for epoch in range(self.config['ppo']['epochs']):
            order = list(range(len(targets)))
            random.Random(c.derive_seed(self.config['seed'], f'mahjong-rl/shuffle/v1/{self.state["iteration_id"]}/{epoch}')).shuffle(order)
            self.order += order
        if not 0 <= self.state['cursor'] <= len(self.order):
            raise c.MahjongError('CHECKPOINT_INVALID', 'invalid RL cursor')

    def acquire(self):
        if self._lease is not None:
            raise c.MahjongError('CONTRACT_MISMATCH', 'RL lease already open')
        cursor = self.state['cursor']
        if cursor >= len(self.order): raise StopIteration
        n = len(self.state['target_bundle']['targets'])
        # Preserve tail batches independently in each epoch.
        end = min(cursor+self.config['ppo']['batch_size'], (cursor//n+1)*n, len(self.order))
        indices = self.order[cursor:end]
        self._lease = c.BatchLease(self.batch(indices), uuid.uuid4().hex)
        return self._lease

    def batch(self, indices):
        payload, identities = [], []
        for index in indices:
            target = self.state['target_bundle']['targets'][index]
            payload.append(dict(record=self.readers[target['trajectory']].read(target['key']), target=c.clone_state(target)))
            identities.append(target['trajectory'] + '/' + target['key'])
        return c.Batch(payload, tuple(identities), len(payload), SCHEMA)

    def commit(self, token):
        self._verify(token)
        self.state['cursor'] += self._lease.batch.sample_count
        self._lease = None

    def release(self, token):
        self._verify(token); self._lease = None

    def _verify(self, token):
        if self._lease is None or self._lease.token != token:
            raise c.MahjongError('CONTRACT_MISMATCH', 'invalid RL lease token')

    def has_open_lease(self): return self._lease is not None
    def state_dict(self):
        if self._lease is not None:
            raise c.MahjongError('CONTRACT_MISMATCH', 'cannot snapshot an open RL lease')
        return c.clone_state(self.state)

    def load_state_dict(self, state):
        if self._lease is not None:
            raise c.MahjongError('CONTRACT_MISMATCH', 'cannot restore with an open RL lease')
        if state['stage'] not in ('WARMUP', 'JOINT') or state['phase'] not in ('READY', 'PREPARED'):
            raise c.MahjongError('CHECKPOINT_INVALID', 'invalid RL state')
        self.state = c.clone_state(state)
        self._rebuild()

    def ready(self, pool, stage, gate_windows):
        state = c.clone_state(self.state)
        state.update(phase='READY', stage=stage, pool=pool, gate_windows=gate_windows,
                     iteration_id=state['iteration_id']+1, refs=[], validation_refs=[], target_bundle=None, cursor=0)
        self.load_state_dict(state)
