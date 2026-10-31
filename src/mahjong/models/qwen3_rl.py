"""Oracle value model and finite-epoch PPO objective. Optimizers live in training."""
from dataclasses import asdict
import torch
from torch import nn
from mahjong import contracts as c
from mahjong.inference.policy import digest
from mahjong.policy.scoring import score_action
from mahjong.rl.oracle import DIM, SCHEMA as ORACLE_SCHEMA

SCHEMA = 'mahjong-policy-rl/v1'


def policy_terms(logprob, old, bc, advantage, clip):
    delta = logprob - old.detach()
    rho = delta.exp()
    d = logprob - bc.detach()
    return dict(policy=-torch.minimum(rho * advantage, rho.clamp(1-clip, 1+clip) * advantage),
                bc_kl=rho * (torch.expm1(-d) + d), old_kl=torch.expm1(delta)-delta,
                entropy=-rho*logprob, ratio=rho, clip_fraction=(torch.abs(rho-1)>clip).float())


class Qwen3RLAdapter(nn.Module):
    def __init__(self, actor, settings):
        super().__init__()
        self.actor, self.settings = actor, dict(settings)
        width = settings['oracle_width']
        self.oracle = nn.Sequential(nn.Linear(DIM, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.value_head = nn.Sequential(nn.Linear(actor.causal_lm.config.hidden_size + width, settings['value_width']),
                                        nn.SiLU(), nn.Linear(settings['value_width'], 1))
        self.set_stage('WARMUP')

    def set_stage(self, stage):
        if stage not in ('WARMUP', 'JOINT'):
            raise c.MahjongError('CONTRACT_MISMATCH', 'invalid RL stage')
        self.actor.requires_grad_(stage == 'JOINT')
        self.actor.eval()

    def parameter_groups(self):
        return [('actor', list(self.actor.parameters())),
                ('critic', list(self.oracle.parameters()) + list(self.value_head.parameters()))]

    def value(self, hidden, oracle):
        features = self.oracle(oracle) if self.settings.get('use_oracle', True) else hidden.new_zeros(self.settings['oracle_width'])
        return self.value_head(torch.cat((hidden.detach(), features), -1)).squeeze(-1)

    def descriptor(self):
        return c.ComponentDescriptor('model', 'mahjong.qwen3_rl/v1', 1, SCHEMA,
            digest(dict(actor=asdict(self.actor.descriptor()), settings=self.settings, oracle=ORACLE_SCHEMA)),
            None, 1, 'state_dict', ('cpu/32-true', 'cuda:0/32-true', 'cuda:0/bf16-mixed'))

    def prepare_batch(self, batch, device):
        self._check(batch)
        return batch  # unpadded records are transferred one path at a time

    def _check(self, batch):
        if batch.input_schema != SCHEMA or not isinstance(batch.payload, list) or len(batch.payload) != batch.sample_count:
            raise c.MahjongError('CONTRACT_MISMATCH', 'invalid RL batch')
        for sample in batch.payload:
            record, target = sample['record'], sample['target']
            if record['decision_key'] != record['oracle_key'] or len(record['oracle']) != DIM:
                raise c.MahjongError('CONTRACT_MISMATCH', 'Oracle alignment/schema mismatch')
            if target['policy_mask'] and (record['forced'] or record['sampling'] != 'categorical' or record['origin'] != 'model'):
                raise c.MahjongError('CONTRACT_MISMATCH', 'policy target is not a categorical learner action')

    def forward(self, batch, *, phase):
        self._check(batch)
        losses, values, terms = [], [], []
        device = next(self.parameters()).device
        for sample in batch.payload:
            r, t = sample['record'], sample['target']
            score = score_action(self.actor, r['prefix'], r['suffixes'], r['chosen'])
            value = self.value(score.prefix_hidden, torch.tensor(r['oracle'], dtype=torch.float32, device=device)).float()
            values.append(value)
            losses.append((value - t['value_target'])**2)
            if t['policy_mask']:
                terms.append(policy_terms(score.action_logprob, score.action_logprob.new_tensor(t['old_logprob']),
                    score.action_logprob.new_tensor(t['bc_logprob']), t['policy_adv'], self.settings['clip']))
        value_loss = torch.stack(losses).mean()
        combined = {k: torch.stack([term[k] for term in terms]).mean() for k in terms[0]} if terms else {}
        loss = value_loss * (self.settings['c_v'] if any(p.requires_grad for p in self.actor.parameters()) else 1.)
        if terms:
            loss = loss + combined['policy'] + self.settings['beta']*combined['bc_kl'] - self.settings['c_e']*combined['entropy']
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(x)) for term in terms for x in term.values()):
            raise c.MahjongError('NONFINITE_LOSS', 'nonfinite RL objective')
        metrics = {'loss': c.MetricStat(float(loss.detach())*len(values), len(values), 'loss'),
                   'value_loss': c.MetricStat(float(value_loss.detach())*len(values), len(values), 'score_squared')}
        for key, v in combined.items():
            metrics[key] = c.MetricStat(float(v.detach())*len(terms), len(terms), 'ratio')
        if terms:
            metrics['ratio_max'] = c.MetricStat(max(float(t['ratio'].detach()) for t in terms), 1, 'ratio')
            metrics['ratio_min'] = c.MetricStat(min(float(t['ratio'].detach()) for t in terms), 1, 'ratio')
        return c.StepResult(loss, metrics)

    def auxiliary_state_dict(self): return {'schema': SCHEMA}
    def load_auxiliary_state_dict(self, state):
        if state != self.auxiliary_state_dict():
            raise c.MahjongError('CHECKPOINT_INVALID', 'RL auxiliary schema mismatch')
    def reset_auxiliary_rng(self, seed):
        self.actor.reset_auxiliary_rng(seed)
