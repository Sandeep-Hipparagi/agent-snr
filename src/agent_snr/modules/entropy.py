"""Loop entropy and semantic drift detection module for agent trajectories."""

from __future__ import annotations

import math
from collections import Counter

from agent_snr.core.embedder import cosine_similarity
from agent_snr.core.frames import FirewallAction, FrameRole, SignalFrame, TraceLedger


class LoopEntropyDetector:
    """Detects repetitive action loops, semantic stagnation, and goal drift in agent traces."""

    def __init__(
        self,
        window_size: int = 5,
        entropy_warn: float = 0.30,
        entropy_break: float = 0.15,
        drift_break: float = 0.78,
        alpha: float = 0.5,
        beta: float = 0.3,
    ) -> None:
        """Initialize the LoopEntropyDetector.

        Args:
            window_size: Number of trailing active execution frames to analyze.
            entropy_warn: Entropy threshold below which INJECT_CORRECTION is triggered.
            entropy_break: Entropy threshold below which CIRCUIT_BREAK is triggered.
            drift_break: EMA goal drift threshold above which CIRCUIT_BREAK is triggered.
            alpha: Weight balancing discrete action entropy vs semantic stagnation.
            beta: EMA smoothing factor for goal drift.
        """
        if window_size <= 1:
            raise ValueError(f"window_size must be greater than 1, got {window_size}")
        self.window_size = window_size
        self.entropy_warn = entropy_warn
        self.entropy_break = entropy_break
        self.drift_break = drift_break
        self.alpha = alpha
        self.beta = beta

    def _action_signature(self, frame: SignalFrame) -> str:
        """Construct a canonical action signature for a frame."""
        node = frame.node_id
        tool = frame.tool_name or "thought"
        args_keys = ",".join(sorted(frame.tool_args.keys())) if frame.tool_args else ""
        return f"{node}:{tool}:{args_keys}"

    def evaluate(
        self,
        ledger: TraceLedger,
        record: bool = True,
    ) -> tuple[FirewallAction, dict[str, float]]:
        """Evaluate the active window of the trace ledger for loops and drift.

        Args:
            ledger: The active TraceLedger containing trajectory frames and history.
            record: Whether to record the computed entropy and drift into the ledger.

        Returns:
            A tuple of (FirewallAction, metrics_dict).
        """
        # Filter to active reasoning/execution frames
        active_frames = [
            f
            for f in ledger.frames
            if f.role in (FrameRole.TOOL_CALL, FrameRole.ASSISTANT)
        ]

        if len(active_frames) < self.window_size:
            default_metrics = {
                "entropy": 1.0,
                "drift": 0.0,
                "h_discrete": 1.0,
                "semantic_stagnation": 0.0,
            }
            return FirewallAction.ALLOW, default_metrics

        window = active_frames[-self.window_size :]

        # 1. Discrete Action Entropy (H_discrete)
        signatures = [self._action_signature(f) for f in window]
        counts = Counter(signatures)
        denom = float(self.window_size)
        probs = [c / denom for c in counts.values()]
        raw_h = -sum(p * math.log2(p) for p in probs)
        max_h = math.log2(self.window_size)
        h_discrete = max(0.0, min(1.0, raw_h / max_h)) if max_h > 0 else 1.0

        # 2. Semantic Stagnation (S_t)
        similarities: list[float] = []
        for i in range(1, len(window)):
            prev_emb = window[i - 1].embedding
            curr_emb = window[i].embedding
            if prev_emb is not None and curr_emb is not None:
                sim = cosine_similarity(curr_emb, prev_emb)
                similarities.append(sim)

        s_t = sum(similarities) / len(similarities) if similarities else 0.0
        s_t_clamped = max(0.0, min(1.0, s_t))

        # 3. Combined Loop Entropy (E_t)
        entropy = self.alpha * h_discrete + (1.0 - self.alpha) * (1.0 - s_t_clamped)
        entropy = max(0.0, min(1.0, float(entropy)))

        # 4. EMA Goal Drift (D_t)
        latest = window[-1]
        if latest.embedding is not None and ledger.goal_embedding is not None:
            raw_drift = 1.0 - cosine_similarity(latest.embedding, ledger.goal_embedding)
            raw_drift = max(0.0, min(2.0, float(raw_drift)))
        else:
            raw_drift = 0.0

        if ledger.drift_history:
            prev_drift = ledger.drift_history[-1]
            drift = self.beta * raw_drift + (1.0 - self.beta) * prev_drift
        else:
            drift = raw_drift

        drift = max(0.0, min(2.0, float(drift)))

        # Record metrics into ledger if requested
        if record:
            ledger.record_entropy(entropy)
            ledger.record_drift(drift)

        metrics = {
            "entropy": entropy,
            "drift": drift,
            "h_discrete": h_discrete,
            "semantic_stagnation": s_t,
        }

        # Policy evaluation
        if entropy <= self.entropy_break or drift >= self.drift_break:
            return FirewallAction.CIRCUIT_BREAK, metrics
        if entropy <= self.entropy_warn:
            return FirewallAction.INJECT_CORRECTION, metrics

        return FirewallAction.ALLOW, metrics
