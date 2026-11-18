"""Unit tests for TelemetryProfiler, StepTelemetry, and CLI diagnostic commands."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from rich.console import Console

from agent_snr import (
    FirewallAction,
    FrameRole,
    SignalFrame,
    StepTelemetry,
    TelemetryProfiler,
    profile_step,
)
from agent_snr.cli import main, run_demo


class TestStepTelemetry:
    """Tests for StepTelemetry data structure and serialization."""

    def test_step_telemetry_creation(self) -> None:
        step = StepTelemetry(
            step_id=1,
            node_id="analyst",
            role=FrameRole.ASSISTANT,
            raw_tokens=200,
            compacted_tokens=50,
            tokens_saved=150,
            snr_score=0.88,
            entropy=0.92,
            drift=0.04,
            action_taken=FirewallAction.ALLOW,
            latency_ms=1.45,
        )
        assert step.step_id == 1
        assert step.node_id == "analyst"
        assert step.role == FrameRole.ASSISTANT
        assert step.raw_tokens == 200
        assert step.compacted_tokens == 50
        assert step.tokens_saved == 150
        assert step.snr_score == 0.88
        assert step.action_taken == FirewallAction.ALLOW
        assert step.latency_ms == 1.45

    def test_step_telemetry_has_slots(self) -> None:
        step = StepTelemetry(
            step_id=1,
            node_id="test",
            role=FrameRole.USER,
            raw_tokens=10,
            compacted_tokens=0,
            tokens_saved=0,
            snr_score=1.0,
            entropy=1.0,
            drift=0.0,
            action_taken=FirewallAction.ALLOW,
            latency_ms=0.5,
        )
        assert hasattr(step, "__slots__")
        with pytest.raises(AttributeError):
            step.undeclared_attr = 123  # type: ignore[attr-defined]

    def test_step_telemetry_dict_round_trip(self) -> None:
        original = StepTelemetry(
            step_id=5,
            node_id="search_worker",
            role=FrameRole.TOOL_CALL,
            raw_tokens=300,
            compacted_tokens=60,
            tokens_saved=240,
            snr_score=0.12,
            entropy=0.14,
            drift=0.85,
            action_taken=FirewallAction.CIRCUIT_BREAK,
            latency_ms=2.1,
        )
        data = original.to_dict()
        assert data["step_id"] == 5
        assert data["role"] == "tool_call"
        assert data["action_taken"] == "circuit_break"

        restored = StepTelemetry.from_dict(data)
        assert restored.step_id == original.step_id
        assert restored.node_id == original.node_id
        assert restored.role == original.role
        assert restored.action_taken == original.action_taken
        assert restored.snr_score == original.snr_score
        assert restored.tokens_saved == original.tokens_saved


class TestTelemetryProfiler:
    """Tests for TelemetryProfiler metric aggregation, report printing, and JSON export."""

    def test_record_step_via_kwargs(self) -> None:
        profiler = TelemetryProfiler(cost_per_million_input=3.0)
        s1 = profiler.record_step(
            node_id="node_a",
            role=FrameRole.USER,
            raw_tokens=100,
            compacted_tokens=0,
            snr_score=1.0,
        )
        assert s1.step_id == 1
        assert s1.raw_tokens == 100
        assert s1.tokens_saved == 0
        assert len(profiler.steps) == 1

    def test_record_step_via_frame(self) -> None:
        profiler = TelemetryProfiler()
        frame = SignalFrame(
            step_id=10,
            node_id="compressor",
            role=FrameRole.TOOL_RESULT,
            raw_tokens=1000,
            compacted_tokens=200,
            snr_score=0.10,
        )
        s = profiler.record_step(
            frame=frame,
            entropy=0.85,
            drift=0.10,
            action_taken=FirewallAction.COMPACT,
            latency_ms=1.5,
        )
        assert s.step_id == 10
        assert s.raw_tokens == 1000
        assert s.compacted_tokens == 200
        assert s.tokens_saved == 800
        assert s.action_taken == FirewallAction.COMPACT

    def test_summary_calculation(self) -> None:
        profiler = TelemetryProfiler(cost_per_million_input=5.0)

        # Step 1: Normal user query
        profiler.record_step(
            node_id="user",
            role=FrameRole.USER,
            raw_tokens=200,
            compacted_tokens=0,
            snr_score=1.0,
            latency_ms=1.0,
        )
        # Step 2: Compacted tool output
        profiler.record_step(
            node_id="tool",
            role=FrameRole.TOOL_RESULT,
            raw_tokens=2000,
            compacted_tokens=500,
            snr_score=0.25,
            action_taken=FirewallAction.COMPACT,
            latency_ms=2.0,
        )
        # Step 3: Loop warning step
        profiler.record_step(
            node_id="agent",
            role=FrameRole.TOOL_CALL,
            raw_tokens=100,
            compacted_tokens=0,
            snr_score=0.20,
            action_taken=FirewallAction.INJECT_CORRECTION,
            latency_ms=3.0,
        )
        # Step 4: Circuit break step
        profiler.record_step(
            node_id="agent",
            role=FrameRole.TOOL_CALL,
            raw_tokens=100,
            compacted_tokens=0,
            snr_score=0.05,
            action_taken=FirewallAction.CIRCUIT_BREAK,
            latency_ms=4.0,
        )

        s = profiler.summary()
        assert s["total_steps"] == 4
        assert s["total_raw_tokens"] == 2400
        assert s["total_compacted_tokens"] == 500
        assert s["total_tokens_saved"] == 1500
        # 1500 / 2400 = 62.5%
        assert s["signal_efficiency_pct"] == 62.5
        assert s["loop_warning_count"] == 1
        assert s["circuit_break_count"] == 1
        assert s["loop_interventions"] == 2
        # (1.0 + 0.25 + 0.20 + 0.05) / 4 = 0.375
        assert s["mean_snr"] == 0.375
        # (1500 / 1M) * $5.00 = $0.0075
        assert s["estimated_usd_saved"] == 0.0075
        # (1 + 2 + 3 + 4) / 4 = 2.5 ms
        assert s["avg_overhead_ms"] == 2.5

    def test_json_export_and_import_round_trip(self) -> None:
        import tempfile

        profiler = TelemetryProfiler(cost_per_million_input=4.0)
        profiler.record_step(
            step_id=1,
            node_id="worker",
            role=FrameRole.ASSISTANT,
            raw_tokens=500,
            compacted_tokens=100,
            snr_score=0.75,
            entropy=0.65,
            drift=0.15,
            action_taken=FirewallAction.ALLOW,
            latency_ms=0.8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sub" / "trace.json"
            profiler.export_json(file_path)
            assert file_path.is_file()

            restored = TelemetryProfiler.from_json(file_path)
            assert restored.cost_per_million_input == 4.0
            assert len(restored.steps) == 1
            assert restored.steps[0].step_id == 1
            assert restored.steps[0].raw_tokens == 500
            assert restored.steps[0].tokens_saved == 400
            assert restored.steps[0].action_taken == FirewallAction.ALLOW

    def test_print_report_rendering(self) -> None:
        profiler = TelemetryProfiler()
        profiler.record_step(
            node_id="test_node",
            role=FrameRole.TOOL_CALL,
            raw_tokens=1000,
            compacted_tokens=100,
            snr_score=0.08,  # Red color coding (< 0.15)
            action_taken=FirewallAction.CIRCUIT_BREAK,
        )
        profiler.record_step(
            node_id="planner",
            role=FrameRole.ASSISTANT,
            raw_tokens=150,
            compacted_tokens=0,
            snr_score=0.30,  # Yellow color coding (< 0.40)
            action_taken=FirewallAction.INJECT_CORRECTION,
        )
        profiler.record_step(
            node_id="user",
            role=FrameRole.USER,
            raw_tokens=50,
            compacted_tokens=0,
            snr_score=0.95,  # Green color coding (>= 0.40)
            action_taken=FirewallAction.ALLOW,
        )

        console = Console(record=True, width=120)
        profiler.print_report(console=console)
        output = console.export_text()

        assert "Agent Signal Firewall Diagnostic Report" in output
        assert "Signal Efficiency" in output
        assert "Step-Level Signal Telemetry" in output
        assert "test_node" in output
        assert "planner" in output


class TestProfileStepHelper:
    """Tests for @profile_step context manager and decorator."""

    def test_profile_step_as_context_manager(self) -> None:
        profiler = TelemetryProfiler()

        with profile_step(
            profiler=profiler,
            node_id="vector_db",
            role=FrameRole.TOOL_CALL,
            snr_score=0.90,
        ) as ctx:
            time.sleep(0.01)  # 10ms
            ctx.set_tokens(raw=600, compacted=150)

        assert len(profiler.steps) == 1
        step = profiler.steps[0]
        assert step.node_id == "vector_db"
        assert step.raw_tokens == 600
        assert step.compacted_tokens == 150
        assert step.tokens_saved == 450
        assert step.latency_ms >= 5.0  # Measured sleep

    def test_profile_step_as_decorator(self) -> None:
        profiler = TelemetryProfiler()

        @profile_step(
            profiler=profiler,
            node_id="decorated_tool",
            role=FrameRole.TOOL_CALL,
            raw_tokens=300,
            compacted_tokens=50,
            snr_score=0.85,
        )
        def run_computation(a: int, b: int) -> int:
            return a + b

        result = run_computation(10, 20)
        assert result == 30
        assert len(profiler.steps) == 1
        step = profiler.steps[0]
        assert step.node_id == "decorated_tool"
        assert step.raw_tokens == 300
        assert step.compacted_tokens == 50
        assert step.tokens_saved == 250


class TestCliCommands:
    """Tests for CLI main entrypoint with demo and inspect."""

    def test_cli_demo_execution(self) -> None:
        console = Console(record=True)
        profiler = run_demo(console=console)
        assert len(profiler.steps) == 10
        s = profiler.summary()
        assert s["total_tokens_saved"] == 4150
        assert s["loop_warning_count"] == 1
        assert s["circuit_break_count"] == 1

    def test_cli_main_demo_and_inspect(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "cli_demo.json"
            exit_code = main(["demo", "--out", str(out_file), "--cost", "2.5"])
            assert exit_code == 0
            assert out_file.is_file()

            # Test inspecting exported trace
            inspect_code = main(["inspect", str(out_file)])
            assert inspect_code == 0
