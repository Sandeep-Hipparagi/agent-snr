"""Unit tests for SkipLevelRouter in the Agent Signal Firewall."""

from __future__ import annotations

from pydantic import BaseModel

from agent_snr import (
    FrameRole,
    HashNgramEmbedder,
    SignalFrame,
    SkipLevelRouter,
    SkipRule,
    TraceLedger,
)


class DiagnosticOutput(BaseModel):
    status: str
    confidence: float


def test_skip_rule_immutability() -> None:
    rule = SkipRule(source_node="scan", bypassed_node="reflect", target_node="execute")
    assert rule.source_node == "scan"
    assert rule.bypassed_node == "reflect"
    assert rule.target_node == "execute"


def test_router_default_fallback_without_rules() -> None:
    router = SkipLevelRouter()
    ledger = TraceLedger(session_id="no_rules")
    frame = SignalFrame(1, "scan", FrameRole.TOOL_RESULT, "ok", snr_score=0.9)

    next_node, skipped = router.resolve_next_node(
        current_node="scan",
        default_next_node="reflect",
        latest_frame=frame,
        ledger=ledger,
    )
    assert next_node == "reflect"
    assert not skipped
    assert ledger.hops_skipped == 0


def test_router_snr_threshold() -> None:
    router = SkipLevelRouter()
    rule = SkipRule(
        source_node="analyzer",
        bypassed_node="critique",
        target_node="synthesizer",
        min_snr=0.60,
    )
    router.register_rule(rule)
    ledger = TraceLedger(session_id="snr_check")

    # Low SNR frame (0.40 < 0.60) -> should not skip
    low_snr_frame = SignalFrame(
        1, "analyzer", FrameRole.ASSISTANT, "Low confidence draft", snr_score=0.40
    )
    next_node, skipped = router.resolve_next_node(
        "analyzer", "critique", low_snr_frame, ledger
    )
    assert next_node == "critique"
    assert not skipped

    # High SNR frame (0.85 >= 0.60) -> should skip
    high_snr_frame = SignalFrame(
        2, "analyzer", FrameRole.ASSISTANT, "High confidence draft", snr_score=0.85
    )
    next_node, skipped = router.resolve_next_node(
        "analyzer", "critique", high_snr_frame, ledger
    )
    assert next_node == "synthesizer"
    assert skipped
    assert ledger.hops_skipped == 1


def test_router_pydantic_schema_validation() -> None:
    router = SkipLevelRouter()
    rule = SkipRule(
        source_node="validator",
        bypassed_node="human_review",
        target_node="deployer",
        min_snr=0.40,
        output_schema=DiagnosticOutput,
    )
    router.register_rule(rule)
    ledger = TraceLedger(session_id="pydantic_check")

    # Invalid JSON -> fails schema validation
    invalid_frame = SignalFrame(
        1, "validator", FrameRole.ASSISTANT, "not json", snr_score=0.9
    )
    next_node, skipped = router.resolve_next_node(
        "validator", "human_review", invalid_frame, ledger
    )
    assert next_node == "human_review"
    assert not skipped

    # Valid JSON matching DiagnosticOutput schema -> bypasses human_review
    valid_frame = SignalFrame(
        2,
        "validator",
        FrameRole.ASSISTANT,
        '{"status": "PASSED", "confidence": 0.99}',
        snr_score=0.9,
    )
    next_node, skipped = router.resolve_next_node(
        "validator", "human_review", valid_frame, ledger
    )
    assert next_node == "deployer"
    assert skipped
    assert ledger.hops_skipped == 1


def test_router_goal_similarity_and_custom_predicate() -> None:
    embedder = HashNgramEmbedder()
    goal = "Migrate database to AWS Aurora PostgreSQL"
    goal_emb = embedder.embed(goal)

    router = SkipLevelRouter()
    rule = SkipRule(
        source_node="planner",
        bypassed_node="security_check",
        target_node="migrator",
        min_snr=0.40,
        min_goal_similarity=0.70,
        custom_predicate=lambda frame, _: "authorized" in frame.content,
    )
    router.register_rule(rule)
    ledger = TraceLedger(session_id="goal_and_pred", goal_embedding=goal_emb)

    # Frame 1: Aligned but missing custom predicate keyword
    f1 = SignalFrame(
        1,
        "planner",
        FrameRole.ASSISTANT,
        "Migrate database to Aurora PostgreSQL instance",
        snr_score=0.9,
        embedding=embedder.embed("Migrate database to Aurora PostgreSQL instance"),
    )
    next_node, skipped = router.resolve_next_node(
        "planner", "security_check", f1, ledger
    )
    assert next_node == "security_check"
    assert not skipped

    # Frame 2: Aligned AND has "authorized" keyword
    f2 = SignalFrame(
        2,
        "planner",
        FrameRole.ASSISTANT,
        "Migrate database to Aurora PostgreSQL instance authorized by admin",
        snr_score=0.9,
        embedding=embedder.embed(
            "Migrate database to Aurora PostgreSQL instance authorized by admin"
        ),
    )
    next_node, skipped = router.resolve_next_node(
        "planner", "security_check", f2, ledger
    )
    assert next_node == "migrator"
    assert skipped
    assert ledger.hops_skipped == 1
