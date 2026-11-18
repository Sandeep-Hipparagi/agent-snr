"""AgentSNRFirewall: Orchestrating compaction, entropy detection, budgeting, and routing."""

from __future__ import annotations

import time
from typing import Any

from agent_snr.core.embedder import BaseEmbedder, HashNgramEmbedder
from agent_snr.core.frames import FirewallAction, FrameRole, SignalFrame, TraceLedger
from agent_snr.core.telemetry import TelemetryProfiler
from agent_snr.modules.budgeter import DynamicAttentionBudgeter
from agent_snr.modules.compactor import StateCompactor
from agent_snr.modules.entropy import LoopEntropyDetector
from agent_snr.modules.router import SkipLevelRouter


class SNRCircuitBreakError(RuntimeError):
    """Exception raised when an agent enters an unrecoverable action or drift loop."""

    def __init__(
        self,
        session_id: str,
        node_id: str,
        metrics: dict[str, Any],
        ledger: TraceLedger,
    ) -> None:
        self.session_id = session_id
        self.node_id = node_id
        self.metrics = metrics
        self.ledger = ledger
        entropy = metrics.get("entropy", 0.0)
        drift = metrics.get("drift", 0.0)
        msg = (
            f"SNR Circuit Break triggered on session '{session_id}' at node '{node_id}': "
            f"entropy={entropy:.2f}, drift={drift:.2f}"
        )
        super().__init__(msg)


class AgentSNRFirewall:
    """Core framework-agnostic Agent Signal Firewall orchestrator."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        compactor: StateCompactor | None = None,
        loop_detector: LoopEntropyDetector | None = None,
        budgeter: DynamicAttentionBudgeter | None = None,
        router: SkipLevelRouter | None = None,
        profiler: TelemetryProfiler | None = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize AgentSNRFirewall.

        Args:
            embedder: Embedder instance (defaults to HashNgramEmbedder).
            compactor: State compactor (defaults to StateCompactor).
            loop_detector: Loop entropy & drift detector (defaults to LoopEntropyDetector).
            budgeter: Attention budgeter (defaults to DynamicAttentionBudgeter).
            router: Skip-level router (defaults to SkipLevelRouter).
            profiler: Telemetry profiler (defaults to TelemetryProfiler).
            dry_run: If True, profile and calculate savings without altering message streams.
        """
        self.embedder: BaseEmbedder = embedder if embedder is not None else HashNgramEmbedder()
        self.compactor: StateCompactor = compactor if compactor is not None else StateCompactor()
        self.loop_detector: LoopEntropyDetector = (
            loop_detector if loop_detector is not None else LoopEntropyDetector()
        )
        self.budgeter: DynamicAttentionBudgeter = (
            budgeter if budgeter is not None else DynamicAttentionBudgeter()
        )
        self.router: SkipLevelRouter = router if router is not None else SkipLevelRouter()
        self.profiler: TelemetryProfiler = (
            profiler if profiler is not None else TelemetryProfiler()
        )
        self.dry_run = dry_run
        self.ledgers: dict[str, TraceLedger] = {}

    def get_or_create_ledger(self, session_id: str, goal_text: str = "") -> TraceLedger:
        """Retrieve or create the active TraceLedger for a session."""
        if session_id not in self.ledgers:
            goal_emb = self.embedder.embed(goal_text) if goal_text else None
            self.ledgers[session_id] = TraceLedger(
                session_id=session_id, goal_embedding=goal_emb
            )
        elif goal_text and self.ledgers[session_id].goal_embedding is None:
            self.ledgers[session_id].goal_embedding = self.embedder.embed(goal_text)

        return self.ledgers[session_id]

    def process_outbound(
        self,
        session_id: str,
        node_id: str,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], FirewallAction, dict[str, Any]]:
        """Process and filter outbound LLM message context window.

        Args:
            session_id: Unique identifier for the agent session.
            node_id: Current node/agent executing the step.
            messages: Full list of message dictionaries for the context window.

        Returns:
            Tuple of (budgeted_messages, firewall_action, metrics_dict).

        Raises:
            SNRCircuitBreakError: When loop entropy drops or goal drift exceeds breaker thresholds.
        """
        t0 = time.perf_counter()
        ledger = self.get_or_create_ledger(session_id)

        # 1. Ingest new messages beyond len(ledger.frames)
        start_idx = len(ledger.frames)
        for i in range(start_idx, len(messages)):
            msg = messages[i]
            role_val = msg.get("role", "user")
            content = str(msg.get("content", ""))
            tool_name = msg.get("name") or msg.get("tool_name")
            tool_args = msg.get("tool_args") or msg.get("args")

            # Pin system frames and initial user goal prompt
            role_norm = (
                role_val.value if isinstance(role_val, FrameRole) else str(role_val).lower()
            )
            is_pinned = (role_norm == "system") or (role_norm == "user" and i == 0)

            # Auto-assign goal embedding if first user message
            emb = self.embedder.embed(content)
            if ledger.goal_embedding is None and role_norm == "user" and content:
                ledger.goal_embedding = emb

            raw_tokens = max(1, len(content) // 4)
            frame = SignalFrame(
                step_id=i + 1,
                node_id=node_id,
                role=role_val,
                content=content,
                tool_name=tool_name,
                tool_args=tool_args,
                raw_tokens=raw_tokens,
                embedding=emb,
                is_pinned=is_pinned,
            )

            # Compact frame and record into ledger
            self.compactor.compact(frame, ledger)
            ledger.add_frame(frame)

        # 2. Evaluate Loop Entropy & Drift
        action, metrics = self.loop_detector.evaluate(ledger)

        # 3. Dynamic Attention Budgeting
        allocated_frames = self.budgeter.allocate(ledger)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Record step telemetry
        latest_frame = ledger.frames[-1] if ledger.frames else None
        if latest_frame is not None:
            self.profiler.record_step(
                frame=latest_frame,
                entropy=metrics["entropy"],
                drift=metrics["drift"],
                action_taken=action,
                latency_ms=latency_ms,
            )

        # 5. Handle Dry Run mode
        if self.dry_run:
            dry_metrics = {
                **metrics,
                "dry_run": True,
                "projected_tokens_saved": ledger.tokens_saved,
                "latency_ms": latency_ms,
            }
            return messages, action, dry_metrics

        # 6. Circuit breaker enforcement
        if action == FirewallAction.CIRCUIT_BREAK:
            raise SNRCircuitBreakError(
                session_id=session_id,
                node_id=node_id,
                metrics=metrics,
                ledger=ledger,
            )

        # 7. Serialize budgeted frames back to messages
        output_messages: list[dict[str, Any]] = []
        for f in allocated_frames:
            msg_dict: dict[str, Any] = {
                "role": f.role.value if isinstance(f.role, FrameRole) else str(f.role),
                "content": f.content,
            }
            if f.tool_name:
                msg_dict["name"] = f.tool_name
            if f.tool_args:
                msg_dict["tool_args"] = f.tool_args
            output_messages.append(msg_dict)

        # 8. Inject correction nudge if warning
        if action == FirewallAction.INJECT_CORRECTION:
            nudge = {
                "role": "system",
                "content": (
                    "[FIREWALL WARNING: Repetitive action/semantic stagnation detected. "
                    "Please reassess your strategy, vary your parameters, or report your findings.]"
                ),
            }
            output_messages.append(nudge)

        resp_metrics = {
            **metrics,
            "latency_ms": latency_ms,
            "tokens_saved": ledger.tokens_saved,
        }
        return output_messages, action, resp_metrics

    def resolve_route(
        self,
        session_id: str,
        current_node: str,
        default_next_node: str,
    ) -> tuple[str, bool]:
        """Resolve dynamic skip-level routing for a session."""
        ledger = self.get_or_create_ledger(session_id)
        latest_frame = ledger.frames[-1] if ledger.frames else None
        if latest_frame is None:
            return default_next_node, False
        return self.router.resolve_next_node(
            current_node=current_node,
            default_next_node=default_next_node,
            latest_frame=latest_frame,
            ledger=ledger,
        )
