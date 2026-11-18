"""Trajectory replay A/B evaluation harness for the Agent Signal Firewall."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_snr.core.firewall import AgentSNRFirewall
from agent_snr.modules.router import SkipRule


@dataclass(slots=True)
class TrajectoryEvaluationResult:
    """Evaluation result for a single multi-step trajectory."""

    task_id: str
    baseline_prompt_tokens: int
    snr_prompt_tokens: int
    token_reduction_pct: float
    baseline_peak_context_tokens: int
    snr_peak_context_tokens: int
    peak_token_reduction_pct: float
    hops_saved: int
    hops_eligible: int
    hops_saved_pct: float
    critical_facts_retained: int
    critical_facts_total: int
    fact_retention_recall_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class HarnessEvaluationSummary:
    """Aggregated evaluation metrics across all benchmark trajectories."""

    tasks_evaluated: int
    total_baseline_tokens: int
    total_snr_tokens: int
    overall_token_reduction_pct: float
    avg_baseline_peak_context: float
    avg_snr_peak_context: float
    peak_context_reduction_pct: float
    total_hops_saved: int
    total_hops_eligible: int
    overall_hops_saved_pct: float
    overall_fact_recall_pct: float
    latency_p50_ms: float
    latency_p95_ms: float
    task_results: list[TrajectoryEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "tasks_evaluated": self.tasks_evaluated,
            "total_baseline_tokens": self.total_baseline_tokens,
            "total_snr_tokens": self.total_snr_tokens,
            "overall_token_reduction_pct": self.overall_token_reduction_pct,
            "avg_baseline_peak_context": self.avg_baseline_peak_context,
            "avg_snr_peak_context": self.avg_snr_peak_context,
            "peak_context_reduction_pct": self.peak_context_reduction_pct,
            "total_hops_saved": self.total_hops_saved,
            "total_hops_eligible": self.total_hops_eligible,
            "overall_hops_saved_pct": self.overall_hops_saved_pct,
            "overall_fact_recall_pct": self.overall_fact_recall_pct,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "task_results": [r.to_dict() for r in self.task_results],
        }


def evaluate_trajectory_file(
    file_path: str | Path,
) -> HarnessEvaluationSummary:
    """Replay and evaluate multi-step trajectories from a JSONL file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    tasks: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                tasks.append(json.loads(line_str))

    task_results: list[TrajectoryEvaluationResult] = []
    all_latencies_ms: list[float] = []

    for task in tasks:
        task_id = str(task.get("task_id", "unknown_task"))
        goal = str(task.get("goal", ""))
        critical_facts: list[str] = list(task.get("critical_facts", []))
        steps: list[dict[str, Any]] = list(task.get("steps", []))
        src_node = task.get("source_node")
        bypassed_node = task.get("bypassed_node")
        tgt_node = task.get("target_node")

        # ----------------------------------------------------------------------
        # 1. Baseline Replay (Unprotected cumulative context)
        # ----------------------------------------------------------------------
        baseline_messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
        baseline_tokens = 0
        baseline_peak = 0

        for s in steps:
            baseline_messages.append(
                {
                    "role": s.get("role", "assistant"),
                    "content": s.get("content", ""),
                    "name": s.get("tool_name"),
                    "tool_args": s.get("tool_args"),
                }
            )
            step_tok = sum(max(1, len(m.get("content", "")) // 4) for m in baseline_messages)
            baseline_tokens += step_tok
            if step_tok > baseline_peak:
                baseline_peak = step_tok

        # ----------------------------------------------------------------------
        # 2. AgentSNRFirewall Replay
        # ----------------------------------------------------------------------
        firewall = AgentSNRFirewall()
        if src_node and bypassed_node and tgt_node:
            firewall.router.register_rule(
                SkipRule(
                    source_node=src_node,
                    bypassed_node=bypassed_node,
                    target_node=tgt_node,
                    min_snr=0.40,
                    min_goal_similarity=0.0,
                )
            )

        snr_messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
        snr_tokens = 0
        snr_peak = 0
        hops_saved = 0
        hops_eligible = 0
        final_budgeted: list[dict[str, Any]] = []

        import time

        for s in steps:
            node_id = str(s.get("node_id", "agent"))
            msg = {
                "role": s.get("role", "assistant"),
                "content": s.get("content", ""),
                "name": s.get("tool_name"),
                "tool_args": s.get("tool_args"),
            }
            snr_messages.append(msg)

            t0 = time.perf_counter()
            budgeted, _action, _metrics = firewall.process_outbound(
                session_id=task_id,
                node_id=node_id,
                messages=snr_messages,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            all_latencies_ms.append(elapsed_ms)
            final_budgeted = budgeted

            step_tok = sum(max(1, len(m.get("content", "")) // 4) for m in budgeted)
            snr_tokens += step_tok
            if step_tok > snr_peak:
                snr_peak = step_tok

            if s.get("expected_skip") and bypassed_node and tgt_node:
                hops_eligible += 1
                next_node, skipped = firewall.resolve_route(
                    session_id=task_id,
                    current_node=node_id,
                    default_next_node=bypassed_node,
                )
                if skipped and next_node == tgt_node:
                    hops_saved += 1

        # ----------------------------------------------------------------------
        # 3. Critical Fact Retention Recall Check
        # ----------------------------------------------------------------------
        retained_count = 0
        all_budgeted_content = " ".join(
            str(m.get("content", "")).lower() for m in final_budgeted
        )
        for fact in critical_facts:
            if fact.lower() in all_budgeted_content:
                retained_count += 1

        recall_pct = (
            (retained_count / len(critical_facts)) * 100.0 if critical_facts else 100.0
        )
        tok_red_pct = (
            (1.0 - (snr_tokens / baseline_tokens)) * 100.0 if baseline_tokens > 0 else 0.0
        )
        peak_red_pct = (
            (1.0 - (snr_peak / baseline_peak)) * 100.0 if baseline_peak > 0 else 0.0
        )
        hops_pct = (
            (hops_saved / hops_eligible) * 100.0 if hops_eligible > 0 else 100.0
        )

        task_results.append(
            TrajectoryEvaluationResult(
                task_id=task_id,
                baseline_prompt_tokens=baseline_tokens,
                snr_prompt_tokens=snr_tokens,
                token_reduction_pct=round(tok_red_pct, 1),
                baseline_peak_context_tokens=baseline_peak,
                snr_peak_context_tokens=snr_peak,
                peak_token_reduction_pct=round(peak_red_pct, 1),
                hops_saved=hops_saved,
                hops_eligible=hops_eligible,
                hops_saved_pct=round(hops_pct, 1),
                critical_facts_retained=retained_count,
                critical_facts_total=len(critical_facts),
                fact_retention_recall_pct=round(recall_pct, 1),
            )
        )

    # --------------------------------------------------------------------------
    # Aggregated Summary
    # --------------------------------------------------------------------------
    tot_base_tok = sum(r.baseline_prompt_tokens for r in task_results)
    tot_snr_tok = sum(r.snr_prompt_tokens for r in task_results)
    overall_red_pct = (
        (1.0 - (tot_snr_tok / tot_base_tok)) * 100.0 if tot_base_tok > 0 else 0.0
    )

    avg_base_peak = (
        sum(r.baseline_peak_context_tokens for r in task_results) / len(task_results)
        if task_results
        else 0.0
    )
    avg_snr_peak = (
        sum(r.snr_peak_context_tokens for r in task_results) / len(task_results)
        if task_results
        else 0.0
    )
    peak_red_pct = (
        (1.0 - (avg_snr_peak / avg_base_peak)) * 100.0 if avg_base_peak > 0 else 0.0
    )

    tot_hops_saved = sum(r.hops_saved for r in task_results)
    tot_hops_eligible = sum(r.hops_eligible for r in task_results)
    overall_hops_pct = (
        (tot_hops_saved / tot_hops_eligible) * 100.0 if tot_hops_eligible > 0 else 100.0
    )

    tot_retained = sum(r.critical_facts_retained for r in task_results)
    tot_critical = sum(r.critical_facts_total for r in task_results)
    overall_recall_pct = (
        (tot_retained / tot_critical) * 100.0 if tot_critical > 0 else 100.0
    )

    if all_latencies_ms:
        sorted_lats = sorted(all_latencies_ms)
        p50_idx = int(len(sorted_lats) * 0.50)
        p95_idx = min(len(sorted_lats) - 1, int(len(sorted_lats) * 0.95))
        p50 = sorted_lats[p50_idx]
        p95 = sorted_lats[p95_idx]
    else:
        p50 = 0.0
        p95 = 0.0

    return HarnessEvaluationSummary(
        tasks_evaluated=len(task_results),
        total_baseline_tokens=tot_base_tok,
        total_snr_tokens=tot_snr_tok,
        overall_token_reduction_pct=round(overall_red_pct, 1),
        avg_baseline_peak_context=round(avg_base_peak, 1),
        avg_snr_peak_context=round(avg_snr_peak, 1),
        peak_context_reduction_pct=round(peak_red_pct, 1),
        total_hops_saved=tot_hops_saved,
        total_hops_eligible=tot_hops_eligible,
        overall_hops_saved_pct=round(overall_hops_pct, 1),
        overall_fact_recall_pct=round(overall_recall_pct, 1),
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        task_results=task_results,
    )
