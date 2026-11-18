"""Unit tests for Agent SNR canonical data structures, token accounting, and embedders."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent_snr import (
    BaseEmbedder,
    FastEmbedAdapter,
    FirewallAction,
    FrameRole,
    HashNgramEmbedder,
    SignalFrame,
    TraceLedger,
    cosine_similarity,
)


class TestEnums:
    """Tests for FrameRole and FirewallAction enumerations."""

    def test_frame_role_values(self) -> None:
        assert FrameRole.SYSTEM == "system"
        assert FrameRole.USER == "user"
        assert FrameRole.ASSISTANT == "assistant"
        assert FrameRole.TOOL_CALL == "tool_call"
        assert FrameRole.TOOL_RESULT == "tool_result"

    def test_frame_role_case_insensitivity(self) -> None:
        assert FrameRole("USER") == FrameRole.USER
        assert FrameRole("user") == FrameRole.USER
        assert FrameRole("SYSTEM") == FrameRole.SYSTEM
        assert FrameRole("TOOL_CALL") == FrameRole.TOOL_CALL
        assert FrameRole("tool_result") == FrameRole.TOOL_RESULT
        assert FrameRole.USER == "USER"
        assert FrameRole.USER == "user"

    def test_invalid_frame_role_raises(self) -> None:
        with pytest.raises(ValueError):
            FrameRole("unknown_role")

    def test_firewall_action_values(self) -> None:
        assert FirewallAction.ALLOW == "allow"
        assert FirewallAction.COMPACT == "compact"
        assert FirewallAction.SKIP_LEVEL == "skip_level"
        assert FirewallAction.INJECT_CORRECTION == "inject_correction"
        assert FirewallAction.CIRCUIT_BREAK == "circuit_break"

    def test_firewall_action_case_insensitivity(self) -> None:
        assert FirewallAction("ALLOW") == FirewallAction.ALLOW
        assert FirewallAction("allow") == FirewallAction.ALLOW
        assert FirewallAction("CIRCUIT_BREAK") == FirewallAction.CIRCUIT_BREAK
        assert FirewallAction("circuit_break") == FirewallAction.CIRCUIT_BREAK
        assert FirewallAction("SKIP_LEVEL") == FirewallAction.SKIP_LEVEL
        assert FirewallAction.ALLOW == "ALLOW"
        assert FirewallAction.ALLOW == "allow"

    def test_invalid_firewall_action_raises(self) -> None:
        with pytest.raises(ValueError):
            FirewallAction("invalid_action")


class TestSignalFrame:
    """Tests for SignalFrame creation, slots, and properties."""

    def test_frame_creation_defaults(self) -> None:
        frame = SignalFrame(step_id=1, node_id="agent_node", role=FrameRole.USER)
        assert frame.step_id == 1
        assert frame.node_id == "agent_node"
        assert frame.role == FrameRole.USER
        assert frame.content == ""
        assert frame.tool_name is None
        assert frame.tool_args is None
        assert frame.raw_tokens == 0
        assert frame.compacted_tokens == 0
        assert frame.snr_score == 0.0
        assert frame.embedding is None
        assert isinstance(frame.timestamp, float)
        assert frame.is_pinned is False

    def test_frame_role_string_auto_conversion(self) -> None:
        frame = SignalFrame(step_id=1, node_id="planner", role="user")
        assert isinstance(frame.role, FrameRole)
        assert frame.role == FrameRole.USER

        frame_upper = SignalFrame(step_id=2, node_id="executor", role="TOOL_CALL")
        assert isinstance(frame_upper.role, FrameRole)
        assert frame_upper.role == FrameRole.TOOL_CALL

    def test_frame_has_slots(self) -> None:
        frame = SignalFrame(step_id=1, node_id="test", role=FrameRole.SYSTEM)
        assert hasattr(frame, "__slots__")
        # In slotted dataclasses without __dict__, setting undeclared attributes fails
        with pytest.raises(AttributeError):
            frame.arbitrary_dynamic_field = "not_allowed"  # type: ignore[attr-defined]

    def test_frame_serialization_round_trip(self) -> None:
        original = SignalFrame(
            step_id=42,
            node_id="analyst",
            role=FrameRole.ASSISTANT,
            content="Plan executed successfully.",
            tool_name="database_query",
            tool_args={"query": "SELECT 1"},
            raw_tokens=150,
            compacted_tokens=45,
            snr_score=0.92,
            embedding=[0.1, 0.2, 0.3],
            is_pinned=True,
        )

        serialized = original.to_dict()
        assert serialized["step_id"] == 42
        assert serialized["role"] == "assistant"
        assert serialized["tokens_saved"] == 105
        assert serialized["effective_tokens"] == 45

        deserialized = SignalFrame.from_dict(serialized)
        assert deserialized.step_id == original.step_id
        assert deserialized.node_id == original.node_id
        assert deserialized.role == original.role
        assert deserialized.content == original.content
        assert deserialized.tool_name == original.tool_name
        assert deserialized.tool_args == original.tool_args
        assert deserialized.raw_tokens == original.raw_tokens
        assert deserialized.compacted_tokens == original.compacted_tokens
        assert deserialized.snr_score == original.snr_score
        assert deserialized.embedding == original.embedding
        assert deserialized.is_pinned == original.is_pinned


class TestTokenAccounting:
    """Tests for token accounting on SignalFrame and TraceLedger."""

    def test_uncompacted_frame_accounting(self) -> None:
        frame = SignalFrame(
            step_id=1,
            node_id="search",
            role=FrameRole.TOOL_RESULT,
            raw_tokens=500,
            compacted_tokens=0,
        )
        assert frame.tokens_saved == 0
        assert frame.effective_tokens == 500

    def test_compacted_frame_accounting(self) -> None:
        frame = SignalFrame(
            step_id=2,
            node_id="search",
            role=FrameRole.TOOL_RESULT,
            raw_tokens=500,
            compacted_tokens=120,
        )
        assert frame.tokens_saved == 380
        assert frame.effective_tokens == 120

    def test_frame_compact_method(self) -> None:
        frame = SignalFrame(
            step_id=3,
            node_id="summarizer",
            role=FrameRole.ASSISTANT,
            content="Extremely long verbose response that needs compaction.",
            raw_tokens=250,
        )
        saved = frame.compact("Short summary.", compacted_tokens=30)
        assert saved == 220
        assert frame.content == "Short summary."
        assert frame.compacted_tokens == 30
        assert frame.tokens_saved == 220
        assert frame.effective_tokens == 30

    def test_compacted_tokens_equal_or_greater_than_raw(self) -> None:
        frame = SignalFrame(
            step_id=4,
            node_id="test",
            role=FrameRole.USER,
            raw_tokens=100,
            compacted_tokens=100,
        )
        assert frame.tokens_saved == 0

        frame_overflow = SignalFrame(
            step_id=5,
            node_id="test",
            role=FrameRole.USER,
            raw_tokens=100,
            compacted_tokens=150,
        )
        assert frame_overflow.tokens_saved == 0

    def test_trace_ledger_token_accumulation(self) -> None:
        ledger = TraceLedger(session_id="session-xyz")

        frame1 = SignalFrame(
            step_id=1,
            node_id="input",
            role=FrameRole.USER,
            raw_tokens=80,
            compacted_tokens=0,
        )
        frame2 = SignalFrame(
            step_id=2,
            node_id="tool",
            role=FrameRole.TOOL_RESULT,
            raw_tokens=400,
            compacted_tokens=100,
        )
        frame3 = SignalFrame(
            step_id=3,
            node_id="agent",
            role=FrameRole.ASSISTANT,
            raw_tokens=200,
            compacted_tokens=50,
        )

        ledger.add_frame(frame1)
        ledger.add_frame(frame2)
        ledger.add_frame(frame3)

        assert ledger.total_raw_tokens == 80 + 400 + 200  # 680
        assert ledger.total_compacted_tokens == 0 + 100 + 50  # 150
        assert ledger.total_effective_tokens == 80 + 100 + 50  # 230
        assert ledger.tokens_saved == 300 + 150  # 450

        # Recalculate test
        assert ledger.recalculate_tokens_saved() == 450


class TestTraceLedger:
    """Tests for TraceLedger operations, metrics, and serialization."""

    def test_ledger_creation(self) -> None:
        ledger = TraceLedger(session_id="sess-001")
        assert ledger.session_id == "sess-001"
        assert ledger.goal_embedding is None
        assert ledger.frames == []
        assert ledger.state_kv == {}
        assert ledger.entropy_history == []
        assert ledger.drift_history == []
        assert ledger.tokens_saved == 0
        assert ledger.hops_skipped == 0

    def test_ledger_metrics_tracking(self) -> None:
        ledger = TraceLedger(session_id="sess-002")

        ledger.record_hop_skipped(2)
        assert ledger.hops_skipped == 2

        ledger.record_drift(0.12)
        ledger.record_drift(0.25)
        assert ledger.drift_history == [0.12, 0.25]

        ledger.record_entropy(1.85)
        assert ledger.entropy_history == [1.85]

    def test_ledger_serialization_round_trip(self) -> None:
        ledger = TraceLedger(
            session_id="sess-serialize",
            goal_embedding=[0.5, -0.5, 0.0],
            state_kv={"target": "api_endpoint", "retries": 1},
            entropy_history=[1.2, 1.4],
            drift_history=[0.05, 0.09],
            tokens_saved=120,
            hops_skipped=3,
        )
        ledger.add_frame(
            SignalFrame(
                step_id=1,
                node_id="root",
                role=FrameRole.USER,
                content="Execute task",
                raw_tokens=50,
            )
        )

        data = ledger.to_dict()
        assert data["session_id"] == "sess-serialize"
        assert len(data["frames"]) == 1
        assert data["hops_skipped"] == 3

        reconstructed = TraceLedger.from_dict(data)
        assert reconstructed.session_id == ledger.session_id
        assert reconstructed.goal_embedding == ledger.goal_embedding
        assert reconstructed.state_kv == ledger.state_kv
        assert reconstructed.entropy_history == ledger.entropy_history
        assert reconstructed.drift_history == ledger.drift_history
        assert len(reconstructed.frames) == 1
        assert reconstructed.frames[0].step_id == 1
        assert reconstructed.frames[0].content == "Execute task"


class TestHashNgramEmbedder:
    """Tests for HashNgramEmbedder and BaseEmbedder protocol."""

    def test_protocol_conformance(self) -> None:
        embedder = HashNgramEmbedder()
        assert isinstance(embedder, BaseEmbedder)

    def test_output_dimension(self) -> None:
        embedder = HashNgramEmbedder(dim=384)
        vec = embedder.embed("Test agent planning workflow")
        assert len(vec) == 384
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)

    def test_custom_dimension(self) -> None:
        embedder = HashNgramEmbedder(dim=128)
        vec = embedder.embed("Testing dimension configuration")
        assert len(vec) == 128

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            HashNgramEmbedder(dim=0)
        with pytest.raises(ValueError):
            HashNgramEmbedder(dim=-10)

    def test_determinism(self) -> None:
        embedder = HashNgramEmbedder()
        text = "Deterministic feature hashing verification across multiple runs."
        vec1 = embedder.embed(text)
        vec2 = embedder.embed(text)
        assert vec1 == vec2

    def test_l2_normalization(self) -> None:
        embedder = HashNgramEmbedder()
        text = "Check vector L2 normalization property"
        vec = embedder.embed(text)
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_string_handling(self) -> None:
        embedder = HashNgramEmbedder()
        vec_empty = embedder.embed("")
        assert len(vec_empty) == 384
        assert all(x == 0.0 for x in vec_empty)

        vec_whitespace = embedder.embed("    \n\t  ")
        assert all(x == 0.0 for x in vec_whitespace)

    def test_batch_embedding(self) -> None:
        embedder = HashNgramEmbedder()
        texts = ["Text one", "Text two", "Text three"]
        batch_vecs = embedder.embed_batch(texts)
        assert len(batch_vecs) == 3
        for i, text in enumerate(texts):
            assert batch_vecs[i] == embedder.embed(text)


class TestCosineSimilarityProperties:
    """Tests for cosine similarity mathematical and semantic properties."""

    def test_self_similarity_is_unity(self) -> None:
        embedder = HashNgramEmbedder()
        vec = embedder.embed("Agent firewall signal detection")
        sim = cosine_similarity(vec, vec)
        assert math.isclose(sim, 1.0, rel_tol=1e-5)

    def test_symmetry(self) -> None:
        embedder = HashNgramEmbedder()
        v1 = embedder.embed("Autonomous agent plan verification")
        v2 = embedder.embed("Agent planning and policy evaluation")
        assert math.isclose(cosine_similarity(v1, v2), cosine_similarity(v2, v1), rel_tol=1e-6)

    def test_bounded_range(self) -> None:
        embedder = HashNgramEmbedder()
        v1 = embedder.embed("Query database for security alerts")
        v2 = embedder.embed("Completely unrelated text about cooking pancakes")
        sim = cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    def test_zero_vector_similarity(self) -> None:
        zero_vec = [0.0] * 384
        embedder = HashNgramEmbedder()
        vec = embedder.embed("Valid content")
        assert cosine_similarity(zero_vec, vec) == 0.0
        assert cosine_similarity(zero_vec, zero_vec) == 0.0

    def test_semantic_lexical_similarity_contrast(self) -> None:
        embedder = HashNgramEmbedder()
        base = "Deploying autonomous container workloads to Kubernetes cluster"
        similar = "Deploy autonomous containers to Kubernetes clusters"
        unrelated = "Delicious chocolate brownie recipe made with dark cocoa"

        v_base = embedder.embed(base)
        v_similar = embedder.embed(similar)
        v_unrelated = embedder.embed(unrelated)

        sim_similar = cosine_similarity(v_base, v_similar)
        sim_unrelated = cosine_similarity(v_base, v_unrelated)

        # High similarity for lexically/semantically related phrases
        assert sim_similar > 0.60
        # Very low/orthogonal similarity for completely unrelated domain
        assert sim_unrelated < 0.15
        assert sim_similar > sim_unrelated


class TestFastEmbedAdapter:
    """Tests for FastEmbedAdapter lazy loading, protocol conformance, and mock injection."""

    def test_protocol_conformance(self) -> None:
        adapter = FastEmbedAdapter()
        assert isinstance(adapter, BaseEmbedder)

    def test_import_error_guidance_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if "fastembed" in name:
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        adapter = FastEmbedAdapter()
        with pytest.raises(ImportError) as exc_info:
            adapter.embed("Test text requiring fastembed")

        assert "fastembed is required" in str(exc_info.value)
        assert "agent-snr[embeddings]" in str(exc_info.value)

    def test_mock_model_injection(self) -> None:
        mock_model: Any = MagicMock()
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mock_model.embed.return_value = [mock_embedding]

        adapter = FastEmbedAdapter(model=mock_model)
        result = adapter.embed("Hello fastembed")

        expected = [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3), pytest.approx(0.4)]
        assert result == expected
        mock_model.embed.assert_called_once_with(["Hello fastembed"])

    def test_mock_batch_embedding(self) -> None:
        mock_model: Any = MagicMock()
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        mock_model.embed.return_value = [v1, v2]

        adapter = FastEmbedAdapter(model=mock_model)
        batch_results = adapter.embed_batch(["text1", "text2"])

        assert len(batch_results) == 2
        assert batch_results[0] == [1.0, 0.0]
        assert batch_results[1] == [0.0, 1.0]

    def test_empty_batch_embedding(self) -> None:
        adapter = FastEmbedAdapter()
        assert adapter.embed_batch([]) == []
