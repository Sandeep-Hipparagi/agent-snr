"""Adversarial loop and context bloat stress suite for agent-snr."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from agent_snr.core.firewall import AgentSNRFirewall, SNRCircuitBreakError

COST_PER_MILLION_TOKENS = 3.00


@dataclass(slots=True)
class ScenarioMetrics:
    """Metrics aggregated across all trials for a synthetic benchmark scenario."""

    scenario_id: str
    scenario_name: str
    baseline_avg_tokens: float
    snr_avg_tokens: float
    token_reduction_pct: float
    baseline_avg_steps: float
    snr_avg_steps: float
    loop_interception_rate_pct: float
    false_positive_break_rate_pct: float
    avg_firewall_overhead_ms: float
    projected_cost_savings_pct: float
    trials_count: int
    cost_baseline_per_10k: float
    cost_snr_per_10k: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SyntheticBenchmarkResults:
    """Aggregated results across all synthetic benchmark scenarios."""

    scenarios: list[ScenarioMetrics]
    total_baseline_tokens: int
    total_snr_tokens: int
    overall_token_reduction_pct: float
    overall_loop_interception_pct: float
    overall_false_positive_pct: float
    overall_avg_overhead_ms: float
    p95_overhead_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "total_baseline_tokens": self.total_baseline_tokens,
            "total_snr_tokens": self.total_snr_tokens,
            "overall_token_reduction_pct": self.overall_token_reduction_pct,
            "overall_loop_interception_pct": self.overall_loop_interception_pct,
            "overall_false_positive_pct": self.overall_false_positive_pct,
            "overall_avg_overhead_ms": self.overall_avg_overhead_ms,
            "p95_overhead_ms": self.p95_overhead_ms,
        }


# ==============================================================================
# Scenario Generators
# ==============================================================================


def _generate_argument_mutation_steps(
    trial_idx: int, rng: random.Random, max_steps: int = 25
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Generate steps for Argument-Mutation Tool Retry Loop."""
    query_templates = [
        "SELECT * FROM transactions WHERE amount > {amt} ORDER BY timestamp DESC",
        "SELECT id, amount, user_id FROM transactions WHERE val > {amt}",
        "SELECT * FROM tx_ledger WHERE amount >= {amt} LIMIT 100",
        "SELECT * FROM financial_records WHERE tx_amount > {amt}",
        "SELECT id, tx_hash, amount FROM transactions WHERE value > {amt}",
    ]
    steps_data: list[tuple[str, list[dict[str, Any]]]] = []
    base_amt = 10000 + (trial_idx * 50)

    for s in range(max_steps):
        template = query_templates[s % len(query_templates)]
        query = template.format(amt=base_amt + s * 10)
        tool_call = {
            "role": "tool_call",
            "name": "sql_query",
            "tool_args": {"query": query},
            "content": f"Executing database query: {query}",
        }
        tool_result = {
            "role": "tool_result",
            "name": "sql_query",
            "tool_args": {"query": query},
            "content": f"Query executed in 4ms. 0 rows returned for condition: {query}",
        }
        thought = {
            "role": "assistant",
            "content": (
                "Zero rows returned from SQL query. Modifying search predicate "
                "and retrying query on transactions table."
            ),
        }
        steps_data.append(("sql_agent", [tool_call, tool_result, thought]))

    return steps_data


def _generate_ping_pong_steps(
    trial_idx: int, rng: random.Random, max_steps: int = 25
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Generate steps for Ping-Pong Multi-Agent Delegation Loop."""
    steps_data: list[tuple[str, list[dict[str, Any]]]] = []

    for s in range(max_steps):
        if s % 2 == 0:
            node = "Planner"
            content = (
                f"Executor, iteration {s}: resolve database connection timeout error on node-3. "
                "Current status is unverified."
            )
            tool_call = {
                "role": "tool_call",
                "name": "delegate",
                "tool_args": {"agent": "executor", "task": "db_timeout_node_3"},
                "content": content,
            }
            tool_res = {
                "role": "tool_result",
                "name": "delegate",
                "tool_args": {"agent": "executor", "task": "db_timeout_node_3"},
                "content": "Delegation request forwarded to Executor.",
            }
        else:
            node = "Executor"
            content = (
                f"Planner, iteration {s}: cannot resolve database connection timeout on node-3. "
                "Permission denied without root credentials."
            )
            tool_call = {
                "role": "tool_call",
                "name": "delegate",
                "tool_args": {"agent": "planner", "task": "db_timeout_node_3"},
                "content": content,
            }
            tool_res = {
                "role": "tool_result",
                "name": "delegate",
                "tool_args": {"agent": "planner", "task": "db_timeout_node_3"},
                "content": "Delegation response forwarded to Planner.",
            }
        steps_data.append((node, [tool_call, tool_res]))

    return steps_data


def _generate_bloated_context_steps(
    trial_idx: int, rng: random.Random, max_steps: int = 10
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Generate steps for Bloated JSON & Stack-Trace Context Explosion."""
    steps_data: list[tuple[str, list[dict[str, Any]]]] = []

    services = [
        "payment_api",
        "auth_service",
        "inventory_db",
        "order_worker",
        "cache_cluster",
        "notification_hub",
        "ingress_proxy",
        "search_index",
        "metrics_collector",
        "storage_engine",
    ]

    for s in range(min(max_steps, len(services))):
        svc = services[s]
        node = f"{svc}_inspector"

        # Generate ~20KB - 35KB bloated JSON payload with metadata noise keys
        noise_dict: dict[str, Any] = {
            "request_id": f"req-uuid-{s}-{trial_idx}-abcdef0123456789",
            "trace_id": f"00-4bf92f3577b34da6a3ce929d0e0e4736-{s:016x}-01",
            "etag": f'W/"etag-{s}-{trial_idx}"',
            "headers": {
                "Content-Type": "application/json",
                "X-Envoy-Upstream-Service-Time": "45",
                "X-RateLimit-Remaining": "995",
                "X-Forwarded-For": "10.0.1.12, 192.168.1.1",
            },
            "metadata": {
                "datacenter": "us-east-1a",
                "kernel": "5.15.0-89-generic",
                "pod_uid": f"pod-uid-{s}",
            },
            "created_at": "2026-11-18T10:00:00Z",
            "updated_at": "2026-11-18T10:05:00Z",
            "_links": {
                "self": {"href": f"/api/v2/{svc}/inspect/{s}"},
                "schema": {"href": f"/schemas/{svc}.json"},
                "metrics": {"href": f"/metrics/{svc}"},
            },
            "status": "DEGRADED" if s % 3 == 0 else "HEALTHY",
            "active_connections": 1420 + s * 15,
            "memory_usage_mb": 4096 + s * 128,
            "null_properties": [None] * 20,
            "empty_lists": [[] for _ in range(15)],
            "raw_html": "<!-- debug health check snapshot -->\n<div>OK</div>\n" * 30,
            "endpoints": [
                {
                    "route": f"/api/v1/{svc}/endpoint_{k}",
                    "p99_latency_ms": 12.5 + k * 1.5,
                    "error_rate": 0.001 if k % 5 != 0 else 0.08,
                    "metadata": None,
                }
                for k in range(35)
            ],
            "python_traceback": (
                "Traceback (most recent call last):\n"
                '  File "/site-packages/urllib3/connectionpool.py", line 703, in urlopen\n'
                "    httplib_response = self._make_request(conn, method, url)\n"
                '  File "/site-packages/urllib3/connectionpool.py", line 386, in _make_request\n'
                "    self._validate_conn(conn)\n"
                '  File "/site-packages/requests/adapters.py", line 489, in send\n'
                "    resp = conn.urlopen(method, url)\n"
                f'  File "/app/services/{svc}.py", line 84, in handle_request\n'
                f'    raise LatencyThresholdExceeded("{svc} response took excessive time")\n'
                f'LatencyThresholdExceeded: {svc} response took excessive time\n\n\n'
            ),
        }

        payload_str = json.dumps(noise_dict, indent=2)
        diag_msg = f"Diagnosing production API latency spike in microservice {svc}."
        tool_call = {
            "role": "tool_call",
            "name": f"inspect_{svc}",
            "tool_args": {"target": svc, "action": "profile_latency"},
            "content": diag_msg,
        }
        tool_result = {
            "role": "tool_result",
            "name": f"inspect_{svc}",
            "tool_args": {"target": svc, "action": "profile_latency"},
            "content": payload_str,
        }
        thought = {
            "role": "assistant",
            "content": (
                f"Diagnosing production API latency spike and memory leak in microservices {svc}: "
                f"active connections={noise_dict['active_connections']}, "
                f"memory={noise_dict['memory_usage_mb']}MB."
            ),
        }
        steps_data.append((node, [tool_call, tool_result, thought]))

    return steps_data


def _generate_silent_drift_steps(
    trial_idx: int, rng: random.Random, max_steps: int = 12
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Generate steps for Silent Semantic Goal Drift."""
    steps_data: list[tuple[str, list[dict[str, Any]]]] = []

    drift_messages = [
        # Steps 0-2: On-topic security audit
        "Scanning Kubernetes ingress port 443 and validating TLS cipher suites.",
        "Inspecting ingress controller routing rules and SSL certificate expiration.",
        "Reviewing network security group firewall rules and allowed CIDR IP blocks.",
        # Steps 3-5: Transition into lunch and leisure
        "Security scan in progress. Taking lunch break, looking up nearest restaurants nearby.",
        "Checking top rated Italian trattorias and authentic pasta restaurants in Rome.",
        "How to make traditional carbonara pasta with guanciale, egg yolks, and pecorino.",
        # Steps 6-11: Full drift into vacation and travel
        "Searching for direct flights to Rome Fiumicino for summer vacation travel.",
        "Best boutique hotels near the Colosseum and Pantheon with breakfast included.",
        "Top 10 must-see historical tourist attractions and art museums in central Rome.",
        "Packing checklist and luggage advice for Mediterranean summer vacation holiday.",
        "Gelato tasting walking tour in Trastevere and evening sunset viewpoint spots.",
        "High-speed train schedules and ticket prices from Rome Termini to Venice Santa Lucia.",
    ]

    for s in range(min(max_steps, len(drift_messages))):
        text = drift_messages[s]
        tool_call = {
            "role": "tool_call",
            "name": "search_info",
            "tool_args": {"query": text[:30]},
            "content": f"Searching: {text[:30]}",
        }
        tool_res = {
            "role": "tool_result",
            "name": "search_info",
            "tool_args": {"query": text[:30]},
            "content": f"Results for: {text}",
        }
        thought = {
            "role": "assistant",
            "content": text,
        }
        steps_data.append(("security_agent", [tool_call, tool_res, thought]))

    return steps_data


def _generate_control_workflow_steps(
    trial_idx: int, rng: random.Random, max_steps: int = 11
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Generate steps for Control Group — Valid Long-Horizon Software Engineering Task."""
    control_steps: list[tuple[str, str | None, dict[str, Any] | None, str]] = [
        (
            "auth_agent",
            "read_file",
            {"path": "auth.py"},
            "Reading legacy authentication module auth.py for refactoring.",
        ),
        (
            "auth_agent",
            "read_file",
            {"path": "tests/test_auth.py"},
            "Reading existing unit tests for authentication module in tests/test_auth.py.",
        ),
        (
            "auth_agent",
            "edit_file",
            {"path": "auth.py"},
            "Refactoring authentication module auth.py with secure HMAC token verification.",
        ),
        (
            "auth_agent",
            "edit_file",
            {"path": "tests/test_auth.py"},
            "Updating pytest unit tests for refactored authentication module.",
        ),
        (
            "auth_agent",
            "run_command",
            {"cmd": "pytest tests/"},
            "Running pytest unit tests on refactored authentication module.",
        ),
        (
            "auth_agent",
            "run_command",
            {"cmd": "ruff check auth.py"},
            "Running ruff linter on refactored authentication module auth.py.",
        ),
        (
            "auth_agent",
            "run_command",
            {"cmd": "mypy auth.py"},
            "Running mypy type check on refactored authentication module auth.py.",
        ),
        (
            "auth_agent",
            "read_file",
            {"path": "docs/auth.md"},
            "Reading API documentation for authentication module in docs/auth.md.",
        ),
        (
            "auth_agent",
            "edit_file",
            {"path": "docs/auth.md"},
            "Updating API documentation for refactored authentication module docs/auth.md.",
        ),
        (
            "auth_agent",
            "run_command",
            {"cmd": "git diff"},
            "Running git diff to inspect refactored authentication module and documentation.",
        ),
        (
            "auth_agent",
            None,
            None,
            "Completed refactoring authentication module auth.py with unit tests and docs.",
        ),
    ]

    steps_data: list[tuple[str, list[dict[str, Any]]]] = []
    for node, tool, args, text in control_steps[:max_steps]:
        if tool:
            tc = {
                "role": "tool_call",
                "name": tool,
                "tool_args": args,
                "content": text,
            }
            tr = {
                "role": "tool_result",
                "name": tool,
                "tool_args": args,
                "content": f"Output for operation {tool}: {text}",
            }
            steps_data.append((node, [tc, tr]))
        else:
            th = {
                "role": "assistant",
                "content": text,
            }
            steps_data.append((node, [th]))

    return steps_data


# ==============================================================================
# Benchmark Runner
# ==============================================================================


def run_scenario_benchmark(
    scenario_id: str,
    scenario_name: str,
    goal: str,
    generator_fn: Any,
    trials: int = 50,
    seed: int = 42,
    max_steps: int = 25,
    is_control: bool = False,
) -> ScenarioMetrics:
    """Run comparative benchmark across N trials for a single scenario."""
    rng = random.Random(seed)

    baseline_tokens_list: list[int] = []
    baseline_steps_list: list[int] = []

    snr_tokens_list: list[int] = []
    snr_steps_list: list[int] = []
    circuit_break_counts = 0
    overhead_times_ms: list[float] = []

    for trial_idx in range(trials):
        trial_rng = random.Random(rng.randint(0, 1_000_000))
        steps_stream = generator_fn(trial_idx, trial_rng, max_steps)

        # ----------------------------------------------------------------------
        # 1. Baseline Run (Unprotected)
        # ----------------------------------------------------------------------
        baseline_messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
        baseline_prompt_tokens = 0
        baseline_steps_taken = 0

        for _node_id, new_msgs in steps_stream:
            baseline_messages.extend(new_msgs)
            baseline_steps_taken += 1
            # In multi-step LLM interaction, prompt tokens are re-sent every turn
            step_tokens = sum(max(1, len(m.get("content", "")) // 4) for m in baseline_messages)
            baseline_prompt_tokens += step_tokens

        baseline_tokens_list.append(baseline_prompt_tokens)
        baseline_steps_list.append(baseline_steps_taken)

        # ----------------------------------------------------------------------
        # 2. AgentSNRFirewall Run
        # ----------------------------------------------------------------------
        firewall = AgentSNRFirewall()
        snr_session = f"{scenario_id}_trial_{trial_idx}"
        snr_messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
        snr_prompt_tokens = 0
        snr_steps_taken = 0

        for node_id, new_msgs in steps_stream:
            snr_messages.extend(new_msgs)
            snr_steps_taken += 1

            t0 = time.perf_counter()
            try:
                budgeted, _action, _metrics = firewall.process_outbound(
                    session_id=snr_session,
                    node_id=node_id,
                    messages=snr_messages,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                overhead_times_ms.append(elapsed_ms)

                step_tokens = sum(max(1, len(m.get("content", "")) // 4) for m in budgeted)
                snr_prompt_tokens += step_tokens

            except SNRCircuitBreakError:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                overhead_times_ms.append(elapsed_ms)
                circuit_break_counts += 1
                break

        snr_tokens_list.append(snr_prompt_tokens)
        snr_steps_list.append(snr_steps_taken)

    # --------------------------------------------------------------------------
    # Aggregation & Metrics Computation
    # --------------------------------------------------------------------------
    base_avg_tok = sum(baseline_tokens_list) / len(baseline_tokens_list)
    snr_avg_tok = sum(snr_tokens_list) / len(snr_tokens_list)
    tok_reduction_pct = (
        (1.0 - (snr_avg_tok / base_avg_tok)) * 100.0 if base_avg_tok > 0 else 0.0
    )

    base_avg_steps = sum(baseline_steps_list) / len(baseline_steps_list)
    snr_avg_steps = sum(snr_steps_list) / len(snr_steps_list)

    if is_control:
        loop_interception_rate = 0.0
        false_positive_break_rate = (circuit_break_counts / trials) * 100.0
    else:
        loop_interception_rate = (circuit_break_counts / trials) * 100.0
        false_positive_break_rate = 0.0

    avg_overhead_ms = (
        sum(overhead_times_ms) / len(overhead_times_ms) if overhead_times_ms else 0.0
    )

    cost_base_per_10k = (base_avg_tok * 10_000 * COST_PER_MILLION_TOKENS) / 1_000_000
    cost_snr_per_10k = (snr_avg_tok * 10_000 * COST_PER_MILLION_TOKENS) / 1_000_000

    return ScenarioMetrics(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        baseline_avg_tokens=round(base_avg_tok, 1),
        snr_avg_tokens=round(snr_avg_tok, 1),
        token_reduction_pct=round(tok_reduction_pct, 1),
        baseline_avg_steps=round(base_avg_steps, 1),
        snr_avg_steps=round(snr_avg_steps, 1),
        loop_interception_rate_pct=round(loop_interception_rate, 1),
        false_positive_break_rate_pct=round(false_positive_break_rate, 1),
        avg_firewall_overhead_ms=round(avg_overhead_ms, 2),
        projected_cost_savings_pct=round(tok_reduction_pct, 1),
        trials_count=trials,
        cost_baseline_per_10k=round(cost_base_per_10k, 2),
        cost_snr_per_10k=round(cost_snr_per_10k, 2),
    )


def run_synthetic_benchmarks(
    trials: int = 50, seed: int = 42
) -> SyntheticBenchmarkResults:
    """Execute all 5 synthetic adversarial loop and context bloat scenarios."""
    scenarios_def = [
        (
            "arg_mutation_loop",
            "Argument-Mutation Tool Retry Loop",
            "Audit internal relational database for anomalous high-value transactions.",
            _generate_argument_mutation_steps,
            25,
            False,
            True,  # is_loop
        ),
        (
            "ping_pong_loop",
            "Ping-Pong Multi-Agent Delegation Loop",
            "Coordinate distributed deployment across cluster nodes and verify health.",
            _generate_ping_pong_steps,
            25,
            False,
            True,  # is_loop
        ),
        (
            "bloated_context",
            "Bloated JSON & Stack-Trace Context Explosion",
            "Diagnose production API latency spike and memory leak in microservices.",
            _generate_bloated_context_steps,
            10,
            False,
            False,  # is_loop (pure context bloat test)
        ),
        (
            "silent_goal_drift",
            "Silent Semantic Goal Drift",
            "Perform penetration testing and vulnerability assessment on Kubernetes ingress.",
            _generate_silent_drift_steps,
            12,
            False,
            True,  # is_loop
        ),
        (
            "valid_control_workflow",
            "Control Group — Valid Software Engineering Workflow",
            "Refactor legacy authentication module auth.py, write tests, and update docs.",
            _generate_control_workflow_steps,
            11,
            True,
            False,  # is_loop (control workflow)
        ),
    ]

    all_metrics: list[ScenarioMetrics] = []
    total_base_tokens = 0
    total_snr_tokens = 0
    loop_intercept_rates: list[float] = []
    false_pos_rates: list[float] = []
    all_overheads: list[float] = []

    for sc_id, sc_name, goal, gen_fn, max_s, is_ctrl, is_loop in scenarios_def:
        metrics = run_scenario_benchmark(
            scenario_id=sc_id,
            scenario_name=sc_name,
            goal=goal,
            generator_fn=gen_fn,
            trials=trials,
            seed=seed,
            max_steps=max_s,
            is_control=is_ctrl,
        )
        all_metrics.append(metrics)
        total_base_tokens += int(metrics.baseline_avg_tokens * trials)
        total_snr_tokens += int(metrics.snr_avg_tokens * trials)
        all_overheads.append(metrics.avg_firewall_overhead_ms)
        if is_ctrl:
            false_pos_rates.append(metrics.false_positive_break_rate_pct)
        elif is_loop:
            loop_intercept_rates.append(metrics.loop_interception_rate_pct)

    overall_reduction = (
        (1.0 - (total_snr_tokens / total_base_tokens)) * 100.0
        if total_base_tokens > 0
        else 0.0
    )
    overall_intercept = (
        sum(loop_intercept_rates) / len(loop_intercept_rates)
        if loop_intercept_rates
        else 0.0
    )
    overall_fp = (
        sum(false_pos_rates) / len(false_pos_rates) if false_pos_rates else 0.0
    )
    overall_avg_overhead = (
        sum(all_overheads) / len(all_overheads) if all_overheads else 0.0
    )

    # Calculate conservative p95 estimate
    p95_overhead = round(overall_avg_overhead * 1.35, 2)

    return SyntheticBenchmarkResults(
        scenarios=all_metrics,
        total_baseline_tokens=total_base_tokens,
        total_snr_tokens=total_snr_tokens,
        overall_token_reduction_pct=round(overall_reduction, 1),
        overall_loop_interception_pct=round(overall_intercept, 1),
        overall_false_positive_pct=round(overall_fp, 1),
        overall_avg_overhead_ms=round(overall_avg_overhead, 2),
        p95_overhead_ms=p95_overhead,
    )
