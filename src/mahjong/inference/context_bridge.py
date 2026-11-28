"""Cross-round online projection layered on the frozen rules bridge."""
from mahjong.envs.projection import project_event, project_header
from mahjong.envs.policy_bridge import RiichiDecisionBridge, HISTORY_KINDS
from mahjong.policy.context import validate_previous_rounds
from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA, PREVIOUS_ROUNDS, RoundContext


class ContextDecisionBridge(RiichiDecisionBridge):
    def _reset(self, **kwargs):
        self._previous_rounds = []
        self._prev_round_cache = {}
        self._prev_rounds_assembled = {}
        return super()._reset(**kwargs)

    def _consume(self, event):
        old = self._builder if event['type'] == 'start_kyoku' else None
        super()._consume(event)
        if old is not None:
            # The parent has now finalized old.next_state against this boundary.
            self._previous_rounds.append(tuple(RoundContext(
                project_header(old.initial, viewer, self._options.mode),
                [project_event(e, viewer, ankan_public=self._options.ankan_public)
                 for e in old.events if e.kind in HISTORY_KINDS], True)
                for viewer in range(4)))
            self._previous_rounds = self._previous_rounds[-PREVIOUS_ROUNDS:]

    def _encode_previous_rounds_cached(self, viewer):
        prev_tuple = tuple(r[viewer] for r in self._previous_rounds)
        assembled_cache = getattr(self, '_prev_rounds_assembled', None)
        if assembled_cache is None:
            assembled_cache = {}
            self._prev_rounds_assembled = assembled_cache
        per_round_cache = getattr(self, '_prev_round_cache', None)
        if per_round_cache is None:
            per_round_cache = {}
            self._prev_round_cache = per_round_cache

        assembled = assembled_cache.get(viewer)
        if (
            assembled is not None
            and len(assembled[0]) == len(prev_tuple)
            and all(
                a is b and (a is None or len(a.visible_history) == alen)
                for (a, alen), b in zip(assembled[0], prev_tuple)
            )
        ):
            return assembled[1]

        validate_previous_rounds(prev_tuple)
        ids = self._codec._ids
        list_id = ids['LIST']
        end_list_id = ids['END_LIST']
        null_id = ids['NULL']
        true_id = ids['TRUE']
        false_id = ids['FALSE']
        hist_id = ids['HISTORY']
        full_hist_id = ids['FULL_HISTORY']
        end_hist_id = ids['END_HISTORY']

        tokens = [list_id]
        refs = []
        for round_ in prev_tuple:
            if round_ is None:
                tokens.append(null_id)
                refs.append((None, 0))
                continue
            rlen = len(round_.visible_history)
            cache_key = (id(round_), viewer)
            cached = per_round_cache.get(cache_key)
            if cached is not None and cached[0] is round_ and cached[1] == rlen:
                r_tokens = cached[2]
            else:
                buf = [true_id if round_.complete else false_id]
                buf.extend(self._codec.encode_header(round_.header))
                buf.append(hist_id)
                buf.append(full_hist_id)
                for ev in round_.visible_history:
                    buf.extend(self._codec.encode_event(ev))
                buf.append(end_hist_id)
                r_tokens = tuple(buf)
                if len(per_round_cache) >= 16:
                    per_round_cache.clear()
                per_round_cache[cache_key] = (round_, rlen, r_tokens)
            tokens.extend(r_tokens)
            refs.append((round_, rlen))
        tokens.append(end_list_id)
        result = tuple(tokens)
        assembled_cache[viewer] = (tuple(refs), result)
        return result

    def _assemble_view_prefix(self, viewer, current_prefix_ids):
        if self._controller_schemas is not None and self._controller_schemas[viewer] != CONTEXT_SCHEMA:
            return current_prefix_ids
        prev_tokens = self._encode_previous_rounds_cached(viewer)
        return (*current_prefix_ids[:3], *prev_tokens, *current_prefix_ids[3:])
