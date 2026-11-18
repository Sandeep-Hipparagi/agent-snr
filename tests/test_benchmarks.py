"""Unit and integration tests for benchmarks and adversarial loop stress suite."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.eval_harness import evaluate_trajectory_file
from benchmarks.run_all import main as run_all_cli
from benchmarks.synthetic_loops import run_synthetic_benchmarks


class TestBenchmarksSuite:
    """Validation test suite for Phase 4 benchmarks and evaluation harness."""

    def test_benchmark_determinism(self) -> None:
        """Verify that running synthetic benchmarks with seed=42 produces bit-identical metrics."""
        run1 = run_synthetic_benchmarks(trials=5, seed=42)
        run2 = run_synthetic_benchmarks(trials=5, seed=42)

        assert run1.total_baseline_tokens == run2.total_baseline_tokens
        assert run1.total_snr_tokens == run2.total_snr_tokens
        assert run1.overall_token_reduction_pct == run2.overall_token_reduction_pct
        assert run1.overall_loop_interception_pct == run2.overall_loop_interception_pct
        assert run1.overall_false_positive_pct == run2.overall_false_positive_pct

        for s1, s2 in zip(run1.scenarios, run2.scenarios, strict=True):
            assert s1.scenario_id == s2.scenario_id
            assert s1.baseline_avg_tokens == s2.baseline_avg_tokens
            assert s1.snr_avg_tokens == s2.snr_avg_tokens
            assert s1.snr_avg_steps == s2.snr_avg_steps
            assert s1.loop_interception_rate_pct == s2.loop_interception_rate_pct

    def test_token_reduction_on_bloated_traces(self) -> None:
        """Verify that bloated context scenario achieves at least 40% token reduction."""
        results = run_synthetic_benchmarks(trials=5, seed=42)
        bloat_scenario = next(
            s for s in results.scenarios if s.scenario_id == "bloated_context"
        )
        assert bloat_scenario.token_reduction_pct >= 40.0
        assert bloat_scenario.baseline_avg_tokens > bloat_scenario.snr_avg_tokens * 2

    def test_adversarial_loop_interception_rate(self) -> None:
        """Verify that all 3 adversarial loop scenarios achieve >= 95% interception rate."""
        results = run_synthetic_benchmarks(trials=10, seed=42)
        assert results.overall_loop_interception_pct >= 95.0

        for sc_id in ("arg_mutation_loop", "ping_pong_loop", "silent_goal_drift"):
            sc = next(s for s in results.scenarios if s.scenario_id == sc_id)
            assert sc.loop_interception_rate_pct >= 95.0
            assert sc.snr_avg_steps < sc.baseline_avg_steps

    def test_control_workflow_zero_false_positives(self) -> None:
        """Verify that valid control workflow achieves 0% false positive breaks."""
        results = run_synthetic_benchmarks(trials=10, seed=42)
        control_sc = next(
            s for s in results.scenarios if s.scenario_id == "valid_control_workflow"
        )
        assert control_sc.false_positive_break_rate_pct == 0.0
        assert control_sc.snr_avg_steps == control_sc.baseline_avg_steps
        assert control_sc.token_reduction_pct > 0.0

    def test_firewall_sub_10ms_overhead(self) -> None:
        """Verify that per-step firewall latency overhead is well under 10ms p95."""
        summary = evaluate_trajectory_file("benchmarks/sample_trajectories.jsonl")
        assert summary.latency_p50_ms < 5.0
        assert summary.latency_p95_ms < 10.0

    def test_trajectory_eval_harness_metrics(self) -> None:
        """Verify trajectory replay A/B evaluation across 20 representative tasks."""
        summary = evaluate_trajectory_file("benchmarks/sample_trajectories.jsonl")

        assert summary.tasks_evaluated == 20
        assert summary.overall_token_reduction_pct > 0.0
        assert summary.peak_context_reduction_pct > 0.0
        assert summary.total_hops_saved == summary.total_hops_eligible
        assert summary.overall_hops_saved_pct == 100.0
        assert summary.overall_fact_recall_pct >= 90.0

    def test_results_json_schema(self) -> None:
        """Verify that benchmarks/results.json contains all required schema keys."""
        results_file = Path("benchmarks/results.json")
        assert results_file.is_file(), "benchmarks/results.json must exist"

        with open(results_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "config" in data
        assert "synthetic_benchmarks" in data
        assert "trajectory_evaluations" in data

        synth = data["synthetic_benchmarks"]
        assert "scenarios" in synth
        assert len(synth["scenarios"]) == 5
        assert synth["overall_token_reduction_pct"] >= 40.0
        assert synth["overall_loop_interception_pct"] >= 95.0
        assert synth["overall_false_positive_pct"] == 0.0

        traj = data["trajectory_evaluations"]
        assert traj["tasks_evaluated"] == 20
        assert traj["overall_hops_saved_pct"] >= 95.0
        assert traj["overall_fact_recall_pct"] >= 90.0
        assert traj["latency_p95_ms"] < 10.0

    def test_run_all_cli_execution(self, tmp_path: Path) -> None:
        """Verify that benchmarks.run_all CLI runs with exit code 0."""
        temp_out = tmp_path / "test_results.json"
        exit_code = run_all_cli(["--trials", "2", "--output", str(temp_out)])
        assert exit_code == 0
        assert temp_out.is_file()
