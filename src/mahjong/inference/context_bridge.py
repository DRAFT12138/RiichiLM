"""Cross-round online projection layered on the frozen rules bridge."""
from mahjong.envs.projection import project_event, project_header
from mahjong.envs.policy_bridge import RiichiDecisionBridge, HISTORY_KINDS
from mahjong.inference.policy import DecisionView
from mahjong.policy.context import validate_context
from mahjong.tokenizer.policy_context import PREVIOUS_ROUNDS, PolicyContextInput, RoundContext


class ContextDecisionBridge(RiichiDecisionBridge):
    def _reset(self, **kwargs):
        self._previous_rounds = []
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

    def _publish(self):
        super()._publish()
        for viewer, view in list(self._views.items()):
            from mahjong.tokenizer.policy_context import CONTEXT_SCHEMA
            if self._controller_schemas is not None and self._controller_schemas[viewer] != CONTEXT_SCHEMA:
                continue
            current = view.sequence
            sequence = PolicyContextInput(current.header, current.visible_history,
                current.hand_state, tuple(r[viewer] for r in self._previous_rounds))
            validate_context(sequence)
            self._views[viewer] = DecisionView(view.key, sequence, view.legal_moves,
                                              view.rules_digest, view.legal_set_complete)
