"""Unit tests for StateCompactor in the Agent Signal Firewall."""

from __future__ import annotations

import json

from agent_snr import (
    FrameRole,
    HashNgramEmbedder,
    SignalFrame,
    StateCompactor,
    TraceLedger,
)


def test_snr_computation() -> None:
    embedder = HashNgramEmbedder()
    compactor = StateCompactor()
    ledger = TraceLedger(session_id="test_snr")

    # Frame 1: Initial query
    f1 = SignalFrame(
        step_id=1,
        node_id="user",
        role=FrameRole.USER,
        content="Calculate sales revenue for fiscal year 2025",
        raw_tokens=20,
        embedding=embedder.embed("Calculate sales revenue for fiscal year 2025"),
    )
    ledger.add_frame(f1)

    # Frame 2: Novel content
    f2 = SignalFrame(
        step_id=2,
        node_id="analyst",
        role=FrameRole.ASSISTANT,
        content="Deploying automated database query on PostgreSQL database cluster",
        raw_tokens=25,
        embedding=embedder.embed(
            "Deploying automated database query on PostgreSQL database cluster"
        ),
    )
    snr2 = compactor.compute_snr(f2, ledger)
    assert snr2 > 0.40  # High lexical and semantic novelty

    # Frame 3: Near-duplicate content
    f3 = SignalFrame(
        step_id=3,
        node_id="analyst",
        role=FrameRole.ASSISTANT,
        content="Calculate sales revenue for fiscal year 2025",
        raw_tokens=20,
        embedding=embedder.embed("Calculate sales revenue for fiscal year 2025"),
    )
    snr3 = compactor.compute_snr(f3, ledger)
    assert snr3 == 0.0  # Zero lexical novelty compared to f1


def test_pinned_and_system_user_preserve() -> None:
    compactor = StateCompactor()
    ledger = TraceLedger(session_id="preserve")

    system_frame = SignalFrame(
        step_id=1,
        node_id="system",
        role=FrameRole.SYSTEM,
        content="System instructions",
        is_pinned=True,
    )
    compacted = compactor.compact(system_frame, ledger)
    assert compacted.content == "System instructions"
    assert compacted.snr_score == 1.0


def test_gate0_noise_suppression() -> None:
    embedder = HashNgramEmbedder()
    compactor = StateCompactor(snr_drop_threshold=0.10)
    ledger = TraceLedger(session_id="noise_suppression")

    # Add 3 prior frames so len(ledger.frames) > 2
    for i in range(1, 4):
        f = SignalFrame(
            step_id=i,
            node_id="agent",
            role=FrameRole.ASSISTANT,
            content=f"Planning step {i} for deployment",
            embedding=embedder.embed(f"Planning step {i} for deployment"),
        )
        ledger.add_frame(f)

    # Duplicate frame with identical words
    duplicate_frame = SignalFrame(
        step_id=4,
        node_id="agent",
        role=FrameRole.ASSISTANT,
        content="Planning step 1 for deployment",
        embedding=embedder.embed("Planning step 1 for deployment"),
    )
    result = compactor.compact(duplicate_frame, ledger)
    assert "Low-signal duplicate output suppressed" in result.content
    assert result.snr_score < 0.10


def test_tier1a_json_pruning() -> None:
    compactor = StateCompactor(max_list_items=2)
    ledger = TraceLedger(session_id="json_pruning")

    raw_payload = {
        "status": "success",
        "request_id": "req-9999-drop",
        "trace_id": "tr-8888-drop",
        "metadata": {"etag": "xyz", "empty_dict": {}},
        "data": {
            "name": "Production Cluster",
            "empty_str": "",
            "none_val": None,
            "items": [
                {"id": 1, "value": "a"},
                {"id": 2, "value": "b"},
                {"id": 3, "value": "c"},
                {"id": 4, "value": "d"},
            ],
        },
    }

    frame = SignalFrame(
        step_id=1,
        node_id="scanner",
        role=FrameRole.TOOL_RESULT,
        content=json.dumps(raw_payload),
    )
    result = compactor.compact(frame, ledger)

    parsed = json.loads(result.content)
    # Check dropped noise keys
    assert "request_id" not in parsed
    assert "trace_id" not in parsed
    assert "metadata" not in parsed
    # Check empty string and None stripped
    assert "empty_str" not in parsed["data"]
    assert "none_val" not in parsed["data"]
    # Check list truncated to max_list_items (2)
    assert len(parsed["data"]["items"]) == 2
    assert parsed["data"]["name"] == "Production Cluster"


def test_tier1a_invalid_json_passthrough() -> None:
    compactor = StateCompactor()
    ledger = TraceLedger(session_id="invalid_json")

    invalid_json = "{not a valid json string: [unclosed"
    frame = SignalFrame(
        step_id=1,
        node_id="tool",
        role=FrameRole.TOOL_RESULT,
        content=invalid_json,
    )
    result = compactor.compact(frame, ledger)
    assert result.content == invalid_json


def test_tier1b_traceback_and_whitespace_folding() -> None:
    compactor = StateCompactor()
    ledger = TraceLedger(session_id="traceback")

    traceback_content = (
        "Traceback (most recent call last):\n"
        '  File "/env/site-packages/urllib3/poolmanager.py", line 10, in request\n'
        "    return self.urlopen()\n"
        '  File "/site-packages/requests/adapters.py", line 25, in send\n'
        "    raise TimeoutError()\n"
        '  File "/app/service.py", line 42, in fetch_data\n'
        "    return client.get()\n"
        "TimeoutError: Request timed out\n\n\n\n"
        "Ending log message."
    )

    frame = SignalFrame(
        step_id=1,
        node_id="tool",
        role=FrameRole.TOOL_RESULT,
        content=traceback_content,
    )
    result = compactor.compact(frame, ledger)

    assert "[... internal site-packages frames hidden ...]" in result.content
    assert "/app/service.py" in result.content  # Application frame preserved
    assert "\n\n\n\n" not in result.content  # Excessive newlines collapsed


def test_tier2_idempotent_read_deduplication() -> None:
    compactor = StateCompactor()
    ledger = TraceLedger(session_id="idempotent")

    content = "STATUS: ACTIVE, USERS: 42, LOAD: 0.12"
    tool_args = {"endpoint": "/v1/health", "detailed": True}

    frame1 = SignalFrame(
        step_id=1,
        node_id="poller",
        role=FrameRole.TOOL_RESULT,
        content=content,
        tool_name="health_check",
        tool_args=tool_args,
    )
    res1 = compactor.compact(frame1, ledger)
    assert res1.content == content  # First read stored in state_kv

    # Second identical read
    frame2 = SignalFrame(
        step_id=2,
        node_id="poller",
        role=FrameRole.TOOL_RESULT,
        content=content,
        tool_name="health_check",
        tool_args=tool_args,
    )
    res2 = compactor.compact(frame2, ledger)
    assert "[Unchanged state for health_check — identical to prior read]" in res2.content


def test_tier3_slm_distillation() -> None:
    def mock_slm(text: str, budget: int) -> str:
        return f"SLM Summary ({budget} tokens): {text[:30]}..."

    compactor = StateCompactor(max_frame_tokens=20, slm_summarizer=mock_slm)
    ledger = TraceLedger(session_id="slm")

    # Long text exceeding 20 tokens (> 80 characters)
    long_text = "This is a very detailed and verbose explanation of an agent task " * 5
    frame = SignalFrame(
        step_id=1,
        node_id="agent",
        role=FrameRole.ASSISTANT,
        content=long_text,
    )
    result = compactor.compact(frame, ledger)
    assert result.content.startswith("SLM Summary (20 tokens):")
