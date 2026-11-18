"""Unit and integration tests for AgentSNRFirewall orchestrator."""

from __future__ import annotations

import pytest

from agent_snr import (
    AgentSNRFirewall,
    FirewallAction,
    LoopEntropyDetector,
    SkipRule,
    SNRCircuitBreakError,
    StateCompactor,
)


def test_firewall_initialization_and_ledger() -> None:
    firewall = AgentSNRFirewall()
    ledger = firewall.get_or_create_ledger("sess-1", goal_text="Find vulnerabilities")
    assert ledger.session_id == "sess-1"
    assert ledger.goal_embedding is not None
    assert len(ledger.goal_embedding) == 384


def test_firewall_process_outbound_normal_flow() -> None:
    firewall = AgentSNRFirewall()
    messages = [
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user", "content": "Audit cloud infrastructure."},
        {"role": "assistant", "content": "Running asset inventory."},
    ]

    out_msgs, action, metrics = firewall.process_outbound(
        session_id="test_normal",
        node_id="planner",
        messages=messages,
    )

    assert action == FirewallAction.ALLOW
    assert len(out_msgs) == 3
    assert out_msgs[0]["role"] == "system"
    assert out_msgs[1]["role"] == "user"
    assert out_msgs[2]["role"] == "assistant"
    assert metrics["latency_ms"] < 25.0  # Fast execution


def test_firewall_compaction_integration() -> None:
    compactor = StateCompactor(max_list_items=1)
    firewall = AgentSNRFirewall(compactor=compactor)

    bloated_json = (
        '{"status": "ok", "request_id": "drop-me", "items": [{"id": 1}, {"id": 2}]}'
    )
    messages = [
        {"role": "user", "content": "Check inventory"},
        {
            "role": "tool_result",
            "name": "inventory_api",
            "content": bloated_json,
        },
    ]

    out_msgs, action, metrics = firewall.process_outbound(
        session_id="test_compact",
        node_id="worker",
        messages=messages,
    )

    assert action == FirewallAction.ALLOW
    tool_msg = out_msgs[1]
    assert "request_id" not in tool_msg["content"]
    assert "inventory_api" == tool_msg["name"]
    assert metrics["tokens_saved"] > 0


def test_firewall_dry_run_mode() -> None:
    firewall = AgentSNRFirewall(dry_run=True)
    messages = [
        {"role": "user", "content": "Analyze system"},
        {"role": "assistant", "content": "Analysis plan"},
    ]

    out_msgs, action, metrics = firewall.process_outbound(
        session_id="dry_run_sess",
        node_id="agent",
        messages=messages,
    )

    # In dry-run mode, exact original messages returned
    assert out_msgs == messages
    assert metrics.get("dry_run") is True
    assert "projected_tokens_saved" in metrics


def test_firewall_inject_correction_nudge() -> None:
    # Set detector with high warn threshold and loose drift to test INJECT_CORRECTION
    detector = LoopEntropyDetector(
        window_size=5,
        entropy_warn=0.99,
        entropy_break=0.01,
        drift_break=2.0,
    )
    firewall = AgentSNRFirewall(loop_detector=detector)

    messages = [
        {"role": "user", "content": "Initial user task"},
        {"role": "assistant", "content": "Step 1 execution"},
        {"role": "tool_call", "name": "t1", "content": "Call tool 1"},
        {"role": "assistant", "content": "Step 2 execution"},
        {"role": "tool_call", "name": "t2", "content": "Call tool 2"},
        {"role": "assistant", "content": "Step 3 execution"},
    ]

    out_msgs, action, metrics = firewall.process_outbound(
        session_id="warn_sess",
        node_id="agent",
        messages=messages,
    )

    assert action == FirewallAction.INJECT_CORRECTION
    # The last message must be the injected system nudge
    last_msg = out_msgs[-1]
    assert last_msg["role"] == "system"
    assert "FIREWALL WARNING" in last_msg["content"]


def test_firewall_circuit_break_error() -> None:
    # Set detector break threshold high to force CIRCUIT_BREAK
    detector = LoopEntropyDetector(window_size=5, entropy_break=0.99)
    firewall = AgentSNRFirewall(loop_detector=detector)

    messages = [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": "Step 1"},
        {"role": "tool_call", "name": "t", "content": "Call"},
        {"role": "assistant", "content": "Step 2"},
        {"role": "tool_call", "name": "t", "content": "Call"},
        {"role": "assistant", "content": "Step 3"},
    ]

    with pytest.raises(SNRCircuitBreakError) as exc_info:
        firewall.process_outbound("break_sess", "agent", messages)

    err = exc_info.value
    assert err.session_id == "break_sess"
    assert err.node_id == "agent"
    assert "entropy" in err.metrics
    assert err.ledger is not None


def test_firewall_resolve_route_delegation() -> None:
    firewall = AgentSNRFirewall()
    rule = SkipRule(
        source_node="analyzer",
        bypassed_node="critique",
        target_node="final",
        min_goal_similarity=0.0,
    )
    firewall.router.register_rule(rule)

    # Process a high SNR message first to establish frame
    messages = [
        {"role": "user", "content": "Goal"},
        {"role": "assistant", "content": "High quality analysis output"},
    ]
    firewall.process_outbound("route_sess", "analyzer", messages)

    next_node, skipped = firewall.resolve_route("route_sess", "analyzer", "critique")
    assert next_node == "final"
    assert skipped


def test_firewall_sub_10ms_latency_assertion() -> None:
    firewall = AgentSNRFirewall()
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Execute database search for customer profiles."},
        {"role": "assistant", "content": "Querying customer table on shard 1."},
        {"role": "tool_result", "content": '{"customer_id": 1234, "status": "active"}'},
    ]

    # Warm-up call
    firewall.process_outbound("latency_sess", "worker", messages[:2])

    # Benchmarked turn
    _, _, metrics = firewall.process_outbound("latency_sess", "worker", messages)

    # Sub-10ms requirement
    latency = metrics["latency_ms"]
    assert latency < 10.0, f"Expected latency < 10.0ms, got {latency:.2f}ms"
