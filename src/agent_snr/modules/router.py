"""Skip-level router: dynamic DAG execution bypass based on SNR, schema, and goal confidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from agent_snr.core.embedder import cosine_similarity
from agent_snr.core.frames import SignalFrame, TraceLedger


@dataclass(frozen=True)
class SkipRule:
    """Rule specifying conditions under which intermediate execution nodes can be safely skipped."""

    source_node: str
    bypassed_node: str
    target_node: str
    min_goal_similarity: float = 0.72
    min_snr: float = 0.45
    output_schema: type[BaseModel] | None = None
    custom_predicate: Callable[[SignalFrame, TraceLedger], bool] | None = None


class SkipLevelRouter:
    """Bypasses unnecessary intermediate reflection or critique in agent execution graphs."""

    def __init__(self, rules: list[SkipRule] | None = None) -> None:
        """Initialize SkipLevelRouter.

        Args:
            rules: Optional initial list of SkipRule objects.
        """
        self.rules: list[SkipRule] = list(rules) if rules is not None else []

    def register_rule(self, rule: SkipRule) -> None:
        """Register a new skip-level routing rule."""
        self.rules.append(rule)

    def resolve_next_node(
        self,
        current_node: str,
        default_next_node: str,
        latest_frame: SignalFrame,
        ledger: TraceLedger,
    ) -> tuple[str, bool]:
        """Determine whether to proceed to default_next_node or skip directly to a target_node.

        Args:
            current_node: Current node in the agent graph.
            default_next_node: Standard next node if no skip rule applies.
            latest_frame: Most recent SignalFrame emitted by current_node.
            ledger: Active TraceLedger for session state and history.

        Returns:
            Tuple of (next_node, was_skipped: bool).
        """
        matching_rules = [
            r
            for r in self.rules
            if r.source_node == current_node and r.bypassed_node == default_next_node
        ]

        for rule in matching_rules:
            # 1. Minimum SNR check
            if latest_frame.snr_score < rule.min_snr:
                continue

            # 2. Pydantic schema validation check
            if rule.output_schema is not None:
                try:
                    rule.output_schema.model_validate_json(latest_frame.content)
                except (ValidationError, ValueError, TypeError):
                    continue

            # 3. Goal alignment similarity check
            if (
                rule.min_goal_similarity > 0.0
                and ledger.goal_embedding is not None
                and latest_frame.embedding is not None
            ):
                sim = cosine_similarity(latest_frame.embedding, ledger.goal_embedding)
                if sim < rule.min_goal_similarity:
                    continue

            # 4. Optional custom predicate check
            if rule.custom_predicate is not None:
                try:
                    passed = rule.custom_predicate(latest_frame, ledger)
                    if not passed:
                        continue
                except Exception:
                    continue

            # All checks passed: execute skip
            ledger.record_hop_skipped(1)
            return rule.target_node, True

        return default_next_node, False
