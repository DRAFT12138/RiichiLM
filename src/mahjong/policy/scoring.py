"""Differentiable complete-action scores, with the prefix's last hidden vector."""
from dataclasses import dataclass
import torch
from mahjong.contracts import MahjongError
from mahjong.policy.constraints import build_trie
from mahjong.tokenizer import MahjongTokenizer
from mahjong.policy.execution import policy_execution, policy_precision


@dataclass(frozen=True)
class ActionScore:
    token_logprobs: torch.Tensor
    action_logprob: torch.Tensor
    prefix_hidden: torch.Tensor


def score_action(actor, prefix, suffixes, chosen, *, skip_deterministic_tail=True):
    """One unpadded path, same legal conditional distributions as generation."""
    lm = actor.causal_lm
    prefix, chosen = tuple(prefix), tuple(chosen)
    suffixes = tuple(map(tuple, suffixes))
    trie = build_trie(suffixes)
    end = MahjongTokenizer().token_id('END_DECISION')
    if (not prefix or chosen not in suffixes or any(s[-1] != end for s in suffixes)
            or len(prefix) + max(map(len, suffixes)) > lm.config.max_position_embeddings):
        raise MahjongError('CONTRACT_MISMATCH', 'invalid scoring path or context budget')
    lm.eval()  # disables stochastic layers without disabling autograd
    device = next(lm.parameters()).device
    # Determine the last actual choice. Values still require one prefill even
    # for a forced action; no later deterministic suffix needs a model call.
    branch_end, node = 0, trie
    for offset, token in enumerate(chosen):
        if len(node) > 1:
            branch_end = offset + 1
        node = node[token]
    with policy_execution(actor):
        if policy_precision(actor) == 'bf16-mixed':
            # BF16 parallel teacher forcing can round differently from cached
            # sampling. Reproduce the exact prefill/decode shapes, retaining
            # autograd through the cache for the complete-action PPO gradient.
            rows, cache, prefix_hidden = [], None, None
            steps = max(1, branch_end) if skip_deterministic_tail else len(chosen)
            for offset in range(steps):
                tokens = prefix if offset==0 else (chosen[offset-1],)
                length = len(prefix)+offset
                options = dict(input_ids=torch.tensor([tokens],dtype=torch.long,device=device),
                    attention_mask=torch.ones((1,length),dtype=torch.long,device=device),
                    position_ids=torch.arange(length-len(tokens),length,device=device)[None],
                    use_cache=True,return_dict=True)
                if cache is not None: options['past_key_values']=cache
                output = lm.model(**options)
                cache = output.past_key_values
                last = output.last_hidden_state[0,-1]
                if offset==0: prefix_hidden=last
                rows.append(lm.lm_head(last[None]).float()[0])
            logits = torch.stack(rows)
        else:
            ids = torch.tensor([prefix + chosen[:-1]], dtype=torch.long, device=device)
            hidden = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                              use_cache=False, return_dict=True).last_hidden_state[0]
            logits = lm.lm_head(hidden[len(prefix)-1:]).float()
            prefix_hidden = hidden[len(prefix)-1]
    scores, node = [], trie
    for row, token in enumerate(chosen):
        allowed = sorted(node)
        if row >= logits.shape[0]:
            scores.append(logits.new_zeros(()))
        else:
            legal = logits[row, allowed]
            if not bool(torch.isfinite(legal).all()):
                raise MahjongError('NONFINITE_LOGITS', 'nonfinite legal score')
            scores.append(torch.log_softmax(legal, -1)[allowed.index(token)])
        node = node[token]
    scores = torch.stack(scores)
    return ActionScore(scores, scores.sum(), prefix_hidden)
