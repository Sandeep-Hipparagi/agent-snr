"""Unit tests for LoopEntropyDetector in the Agent Signal Firewall."""

from __future__ import annotations

import pytest

from agent_snr import (
    FirewallAction,
    FrameRole,
    HashNgramEmbedder,
    LoopEntropyDetector,
    SignalFrame,
    TraceLedger,
)


@pytest.fixture
def embedder() -> HashNgramEmbedder:
    return HashNgramEmbedder()


class TestLoopEntropyDetector:
    """Tests for loop entropy calculation, semantic stagnation, and drift."""

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError):
            LoopEntropyDetector(window_size=1)
        with pytest.raises(ValueError):
            LoopEntropyDetector(window_size=0)

    def test_fewer_than_window_size_active_frames(self, embedder: HashNgramEmbedder) -> None:
        detector = LoopEntropyDetector(window_size=5)
        ledger = TraceLedger(session_id="test-under-window")

        # Add 2 user frames and 2 active frames (total active = 2 < 5)
        ledger.add_frame(SignalFrame(1, "user", FrameRole.USER, "Do task", raw_tokens=10))
        ledger.add_frame(
            SignalFrame(2, "agent", FrameRole.ASSISTANT, "Thinking...", raw_tokens=20)
        )
        ledger.add_frame(
            SignalFrame(
                3,
                "agent",
                FrameRole.TOOL_CALL,
                "calling tool",
                tool_name="tool_a",
                tool_args={"x": 1},
                raw_tokens=30,
            )
        )
        ledger.add_frame(
            SignalFrame(4, "user", FrameRole.USER, "Next task", raw_tokens=10)
        )

        action, metrics = detector.evaluate(ledger)
        assert action == FirewallAction.ALLOW
        assert metrics["entropy"] == 1.0
        assert metrics["drift"] == 0.0
        assert metrics["h_discrete"] == 1.0
        assert metrics["semantic_stagnation"] == 0.0

    def test_high_diversity_normal_trace(self, embedder: HashNgramEmbedder) -> None:
        detector = LoopEntropyDetector(window_size=5)
        goal = "Analyze quarterly financial revenue reports and growth drivers"
        ledger = TraceLedger(session_id="diverse", goal_embedding=embedder.embed(goal))

        diverse_steps = [
            ("planner", FrameRole.ASSISTANT, "Analyze quarterly income statements", None, None),
            (
                "recon",
                FrameRole.TOOL_CALL,
                "Query SQL database for 2025 revenue",
                "sql_query",
                {"query": "SELECT revenue FROM q4"},
            ),
            (
                "analyst",
                FrameRole.ASSISTANT,
                "Calculate gross margin and operating profit ratio",
                None,
                None,
            ),
            (
                "chart",
                FrameRole.TOOL_CALL,
                "Generate plot of revenue trajectory",
                "plot_chart",
                {"type": "line", "axis": "growth"},
            ),
            (
                "reporter",
                FrameRole.ASSISTANT,
                "Synthesize quarterly financial revenue summary for stakeholders",
                None,
                None,
            ),
        ]

        for idx, (node, role, content, tool, args) in enumerate(diverse_steps, start=1):
            frame = SignalFrame(
                step_id=idx,
                node_id=node,
                role=role,
                content=content,
                tool_name=tool,
                tool_args=args,
                raw_tokens=50,
                embedding=embedder.embed(content),
            )
            ledger.add_frame(frame)

        action, metrics = detector.evaluate(ledger)
        assert action == FirewallAction.ALLOW
        assert metrics["entropy"] > 0.70
        assert metrics["h_discrete"] == 1.0
        assert metrics["drift"] < detector.drift_break

    def test_identical_tool_loop_triggers_circuit_break(
        self,
        embedder: HashNgramEmbedder,
    ) -> None:
        detector = LoopEntropyDetector(
            window_size=5,
            entropy_break=0.15,
            entropy_warn=0.30,
        )
        ledger = TraceLedger(session_id="loop-test")
        content = "Search s3 bucket for configuration files"
        emb = embedder.embed(content)

        # 5 identical tool calls
        for idx in range(1, 6):
            frame = SignalFrame(
                step_id=idx,
                node_id="worker",
                role=FrameRole.TOOL_CALL,
                content=content,
                tool_name="search_tool",
                tool_args={"bucket": "conf", "key": "app.json"},
                raw_tokens=50,
                embedding=emb,
            )
            ledger.add_frame(frame)

        action, metrics = detector.evaluate(ledger)
        assert action == FirewallAction.CIRCUIT_BREAK
        assert metrics["h_discrete"] == 0.0
        assert metrics["semantic_stagnation"] == 1.0
        assert metrics["entropy"] <= 0.15

    def test_mutated_retry_loop_triggers_inject_correction(
        self,
        embedder: HashNgramEmbedder,
    ) -> None:
        detector = LoopEntropyDetector(
            window_size=5,
            entropy_break=0.15,
            entropy_warn=0.30,
            alpha=0.5,
        )
        ledger = TraceLedger(session_id="warn-test")

        # 1 diverse active frame, followed by 4 near-identical retry frames
        frame0 = SignalFrame(
            step_id=1,
            node_id="planner",
            role=FrameRole.ASSISTANT,
            content="Initial high level plan for database indexing.",
            raw_tokens=40,
            embedding=embedder.embed("Initial high level plan for database indexing."),
        )
        ledger.add_frame(frame0)

        for idx in range(2, 6):
            text = f"Retrying query attempt {idx} for customer data"
            frame = SignalFrame(
                step_id=idx,
                node_id="worker",
                role=FrameRole.TOOL_CALL,
                content=text,
                tool_name="sql_exec",
                tool_args={"table": "customers"},
                raw_tokens=40,
                embedding=embedder.embed(text),
            )
            ledger.add_frame(frame)

        action, metrics = detector.evaluate(ledger)
        # Should be below warning threshold (0.30) but above break threshold (0.15)
        assert action == FirewallAction.INJECT_CORRECTION
        assert detector.entropy_break < metrics["entropy"] <= detector.entropy_warn

    def test_off_topic_drift_triggers_circuit_break(
        self,
        embedder: HashNgramEmbedder,
    ) -> None:
        detector = LoopEntropyDetector(
            window_size=5,
            drift_break=0.78,
            beta=0.8,
        )
        goal = "Deploy containerized Kubernetes pod on production cluster"
        ledger = TraceLedger(session_id="drift-test", goal_embedding=embedder.embed(goal))

        # Add 5 active frames that are completely off-topic (cooking recipes)
        off_topic_queries = [
            "Bake chocolate chip cookies with unsalted butter",
            "Whisk flour, baking soda, and granulated white sugar",
            "Preheat kitchen convection oven to 375 degrees Fahrenheit",
            "Fold in dark chocolate chunks into cookie dough batter",
            "Cool freshly baked cookies on a wire rack for twenty minutes",
        ]

        for idx, text in enumerate(off_topic_queries, start=1):
            frame = SignalFrame(
                step_id=idx,
                node_id=f"node_{idx}",
                role=FrameRole.TOOL_CALL,
                content=text,
                tool_name=f"recipe_tool_{idx}",
                tool_args={"step": idx},
                raw_tokens=30,
                embedding=embedder.embed(text),
            )
            ledger.add_frame(frame)

        action, metrics = detector.evaluate(ledger)
        assert action == FirewallAction.CIRCUIT_BREAK
        assert metrics["drift"] >= 0.78

    def test_record_flag_controls_ledger_history(
        self,
        embedder: HashNgramEmbedder,
    ) -> None:
        detector = LoopEntropyDetector(window_size=5)
        ledger = TraceLedger(session_id="record-flag-test")

        for idx in range(1, 6):
            frame = SignalFrame(
                step_id=idx,
                node_id=f"node_{idx}",
                role=FrameRole.TOOL_CALL,
                content=f"Step {idx}",
                embedding=embedder.embed(f"Step {idx}"),
            )
            ledger.add_frame(frame)

        # evaluate with record=False
        detector.evaluate(ledger, record=False)
        assert len(ledger.entropy_history) == 0
        assert len(ledger.drift_history) == 0

        # evaluate with record=True
        detector.evaluate(ledger, record=True)
        assert len(ledger.entropy_history) == 1
        assert len(ledger.drift_history) == 1
