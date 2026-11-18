"""Dynamic attention budgeter: Utility-density knapsack allocation for context windows."""

from __future__ import annotations

import math

from agent_snr.core.embedder import cosine_similarity
from agent_snr.core.frames import SignalFrame, TraceLedger


class DynamicAttentionBudgeter:
    """Allocates limited LLM context window tokens to maximize trajectory Signal-to-Noise Ratio."""

    def __init__(
        self,
        max_context_tokens: int = 8192,
        recency_window: int = 2,
        decay_lambda: float = 0.08,
        w_snr: float = 1.0,
        w_goal: float = 1.2,
    ) -> None:
        """Initialize DynamicAttentionBudgeter.

        Args:
            max_context_tokens: Hard ceiling for total effective tokens in allocated context.
            recency_window: Number of trailing non-pinned frames to preserve unconditionally.
            decay_lambda: Exponential age decay rate.
            w_snr: Exponent weight on frame SNR score.
            w_goal: Exponent weight on goal cosine similarity.
        """
        if max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be positive, got {max_context_tokens}")
        self.max_context_tokens = max_context_tokens
        self.recency_window = max(0, recency_window)
        self.decay_lambda = decay_lambda
        self.w_snr = w_snr
        self.w_goal = w_goal

    def compute_utility(
        self,
        frame: SignalFrame,
        current_step: int,
        goal_embedding: list[float] | None,
    ) -> float:
        """Compute the information utility score of a frame.

        Utility = (max(0.05, frame.snr_score) ** w_snr) * (goal_rel ** w_goal) * exp(-decay * age)
        """
        age = max(0, current_step - frame.step_id)

        if frame.embedding is not None and goal_embedding is not None:
            raw_sim = cosine_similarity(frame.embedding, goal_embedding)
            goal_rel = max(0.1, raw_sim)
        else:
            goal_rel = 0.5

        snr_factor = max(0.05, frame.snr_score) ** self.w_snr
        goal_factor = goal_rel**self.w_goal
        recency_factor = math.exp(-self.decay_lambda * age)

        return float(snr_factor * goal_factor * recency_factor)

    def allocate(self, ledger: TraceLedger) -> list[SignalFrame]:
        """Greedily allocate context budget preserving pinned and recent frames.

        Returns:
            List of selected SignalFrames sorted strictly by ascending step_id.
        """
        if not ledger.frames:
            return []

        # If all frames comfortably fit within budget, return them all
        total_tokens = sum(f.effective_tokens for f in ledger.frames)
        if total_tokens <= self.max_context_tokens:
            return list(ledger.frames)

        current_step = max((f.step_id for f in ledger.frames), default=0)

        # 1. Identify unconditionally preserved frames
        pinned_frames = [f for f in ledger.frames if f.is_pinned]
        non_pinned = [f for f in ledger.frames if not f.is_pinned]

        if len(non_pinned) <= self.recency_window:
            recent_frames = list(non_pinned)
            candidate_frames: list[SignalFrame] = []
        else:
            recent_frames = non_pinned[-self.recency_window :]
            candidate_frames = non_pinned[: -self.recency_window]

        # Collect preserved frame IDs
        preserved_ids = {f.step_id for f in pinned_frames} | {f.step_id for f in recent_frames}
        used_tokens = sum(
            f.effective_tokens for f in ledger.frames if f.step_id in preserved_ids
        )
        remaining_budget = max(0, self.max_context_tokens - used_tokens)

        # 2. Score candidate frames by utility density (utility / effective_tokens)
        scored_candidates: list[tuple[float, SignalFrame]] = []
        for f in candidate_frames:
            u = self.compute_utility(f, current_step, ledger.goal_embedding)
            density = u / max(1, f.effective_tokens)
            scored_candidates.append((density, f))

        # Sort descending by density
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 3. Greedily pack candidate frames
        selected_ids = set(preserved_ids)
        for _, frame in scored_candidates:
            cost = frame.effective_tokens
            if cost <= remaining_budget:
                selected_ids.add(frame.step_id)
                remaining_budget -= cost

        # 4. Return selected frames sorted strictly by ascending step_id
        selected_frames = [f for f in ledger.frames if f.step_id in selected_ids]
        selected_frames.sort(key=lambda f: f.step_id)
        return selected_frames
