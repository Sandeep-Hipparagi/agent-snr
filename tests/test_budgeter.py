"""Unit tests for DynamicAttentionBudgeter in the Agent Signal Firewall."""

from __future__ import annotations

import math

from agent_snr import (
    DynamicAttentionBudgeter,
    FrameRole,
    HashNgramEmbedder,
    SignalFrame,
    TraceLedger,
)


def test_utility_computation() -> None:
    embedder = HashNgramEmbedder()
    budgeter = DynamicAttentionBudgeter(w_snr=1.0, w_goal=1.0, decay_lambda=0.1)

    goal = "Execute security audit on cloud infrastructure"
    goal_emb = embedder.embed(goal)

    frame = SignalFrame(
        step_id=5,
        node_id="analyst",
        role=FrameRole.ASSISTANT,
        content="Execute cloud infrastructure security audit",
        snr_score=0.9,
        embedding=embedder.embed("Execute cloud infrastructure security audit"),
    )

    # At current_step 5 (age 0), recency factor = 1.0
    u0 = budgeter.compute_utility(frame, current_step=5, goal_embedding=goal_emb)
    # At current_step 10 (age 5), recency factor = exp(-0.5) ~ 0.606
    u5 = budgeter.compute_utility(frame, current_step=10, goal_embedding=goal_emb)

    assert u0 > u5
    assert math.isclose(u5 / u0, math.exp(-0.5), rel_tol=1e-3)


def test_allocate_all_fit_within_budget() -> None:
    budgeter = DynamicAttentionBudgeter(max_context_tokens=1000)
    ledger = TraceLedger(session_id="all_fit")

    for i in range(1, 4):
        ledger.add_frame(
            SignalFrame(
                step_id=i,
                node_id=f"node_{i}",
                role=FrameRole.ASSISTANT,
                content=f"Step {i}",
                raw_tokens=20,
                compacted_tokens=20,
            )
        )

    allocated = budgeter.allocate(ledger)
    assert len(allocated) == 3
    assert [f.step_id for f in allocated] == [1, 2, 3]


def test_allocate_preserves_pinned_and_recency_window() -> None:
    # Set a tight budget of 60 tokens: each frame is 20 tokens
    budgeter = DynamicAttentionBudgeter(max_context_tokens=60, recency_window=2)
    ledger = TraceLedger(session_id="pinned_recency")

    # Frame 1: Pinned system prompt (20 tokens)
    f1 = SignalFrame(
        step_id=1,
        node_id="system",
        role=FrameRole.SYSTEM,
        content="System instructions",
        raw_tokens=20,
        compacted_tokens=20,
        is_pinned=True,
    )
    # Frame 2: Older frame (20 tokens)
    f2 = SignalFrame(
        step_id=2,
        node_id="worker",
        role=FrameRole.TOOL_RESULT,
        content="Old data",
        raw_tokens=20,
        compacted_tokens=20,
        snr_score=0.1,
    )
    # Frame 3: Recent frame 1 (20 tokens) - within recency_window
    f3 = SignalFrame(
        step_id=3,
        node_id="worker",
        role=FrameRole.ASSISTANT,
        content="Recent 1",
        raw_tokens=20,
        compacted_tokens=20,
    )
    # Frame 4: Recent frame 2 (20 tokens) - within recency_window
    f4 = SignalFrame(
        step_id=4,
        node_id="worker",
        role=FrameRole.ASSISTANT,
        content="Recent 2",
        raw_tokens=20,
        compacted_tokens=20,
    )

    ledger.add_frame(f1)
    ledger.add_frame(f2)
    ledger.add_frame(f3)
    ledger.add_frame(f4)

    allocated = budgeter.allocate(ledger)
    # Budget is 60 tokens = 3 frames.
    # Preserved: f1 (pinned) + f3, f4 (trailing 2 recent frames). f2 is dropped!
    allocated_ids = [f.step_id for f in allocated]
    assert allocated_ids == [1, 3, 4]
    assert 2 not in allocated_ids


def test_allocate_packs_by_utility_density() -> None:
    embedder = HashNgramEmbedder()
    goal = "Optimize production database queries"
    goal_emb = embedder.embed(goal)

    # Budget allows 3 frames (60 tokens):
    # Pinned (1) + Recency window (2) = 2 frames (40 tokens).
    # 1 slot remaining (20 tokens) for either frame 2 or frame 3.
    budgeter = DynamicAttentionBudgeter(max_context_tokens=60, recency_window=1)
    ledger = TraceLedger(session_id="utility_density", goal_embedding=goal_emb)

    # Frame 1: Pinned (20 tokens)
    f1 = SignalFrame(
        step_id=1,
        node_id="system",
        role=FrameRole.SYSTEM,
        is_pinned=True,
        raw_tokens=20,
        compacted_tokens=20,
    )
    # Frame 2: Low SNR, unrelated to goal (20 tokens)
    f2 = SignalFrame(
        step_id=2,
        node_id="agent",
        role=FrameRole.ASSISTANT,
        content="Cooking pasta with tomatoes",
        raw_tokens=20,
        compacted_tokens=20,
        snr_score=0.1,
        embedding=embedder.embed("Cooking pasta with tomatoes"),
    )
    # Frame 3: High SNR, highly aligned with goal (20 tokens)
    f3 = SignalFrame(
        step_id=3,
        node_id="agent",
        role=FrameRole.ASSISTANT,
        content="Optimize database indexing strategy and slow queries",
        raw_tokens=20,
        compacted_tokens=20,
        snr_score=0.9,
        embedding=embedder.embed("Optimize database indexing strategy and slow queries"),
    )
    # Frame 4: Recent frame (20 tokens)
    f4 = SignalFrame(
        step_id=4,
        node_id="agent",
        role=FrameRole.ASSISTANT,
        content="Current active step",
        raw_tokens=20,
        compacted_tokens=20,
    )

    ledger.add_frame(f1)
    ledger.add_frame(f2)
    ledger.add_frame(f3)
    ledger.add_frame(f4)

    allocated = budgeter.allocate(ledger)
    allocated_ids = [f.step_id for f in allocated]
    # f3 has much higher utility density than f2, so f3 is selected over f2!
    assert allocated_ids == [1, 3, 4]


def test_allocate_preserves_chronological_order() -> None:
    budgeter = DynamicAttentionBudgeter(max_context_tokens=100)
    ledger = TraceLedger(session_id="order_test")

    for i in range(1, 10):
        ledger.add_frame(
            SignalFrame(
                step_id=i,
                node_id="agent",
                role=FrameRole.ASSISTANT,
                raw_tokens=20,
                compacted_tokens=20,
            )
        )

    allocated = budgeter.allocate(ledger)
    ids = [f.step_id for f in allocated]
    # Output must be strictly ascending
    assert ids == sorted(ids)
