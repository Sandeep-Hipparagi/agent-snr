"""Command-line interface for the Agent Signal Firewall."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from agent_snr.core.embedder import HashNgramEmbedder
from agent_snr.core.frames import FirewallAction, FrameRole, SignalFrame, TraceLedger
from agent_snr.core.telemetry import TelemetryProfiler
from agent_snr.modules.entropy import LoopEntropyDetector


def run_demo(
    out_path: str | Path | None = None,
    cost_per_million: float = 3.0,
    console: Console | None = None,
) -> TelemetryProfiler:
    """Generate and profile a realistic 10-step synthetic multi-agent trajectory.

    Demonstrates:
    - Normal planning and tool dispatch.
    - Bloated JSON tool outputs compacted to save thousands of tokens.
    - Repetitive retry loops triggering INJECT_CORRECTION and CIRCUIT_BREAK.
    """
    if console is None:
        console = Console()

    embedder = HashNgramEmbedder()
    detector = LoopEntropyDetector(
        window_size=5,
        entropy_warn=0.30,
        entropy_break=0.15,
        drift_break=0.78,
    )
    profiler = TelemetryProfiler(cost_per_million_input=cost_per_million)

    goal_text = "Audit cloud infrastructure and search public s3 bucket logs for exposed policies."
    goal_emb = embedder.embed(goal_text)
    ledger = TraceLedger(session_id="demo-session-42", goal_embedding=goal_emb)

    # 10 realistic synthetic steps
    step_specs: list[dict[str, Any]] = [
        # Step 1: User Request
        {
            "node_id": "user",
            "role": FrameRole.USER,
            "content": goal_text,
            "tool_name": None,
            "tool_args": None,
            "raw_tokens": 45,
            "compacted_tokens": 0,
            "snr_score": 1.0,
            "action": FirewallAction.ALLOW,
        },
        # Step 2: Orchestrator Planning
        {
            "node_id": "orchestrator",
            "role": FrameRole.ASSISTANT,
            "content": "Planning s3 bucket audit across cloud regions.",
            "tool_name": None,
            "tool_args": None,
            "raw_tokens": 85,
            "compacted_tokens": 0,
            "snr_score": 0.95,
            "action": FirewallAction.ALLOW,
        },
        # Step 3: Tool Call (Inventory Scanner)
        {
            "node_id": "scanner",
            "role": FrameRole.TOOL_CALL,
            "content": "Listing storage resources in infrastructure.",
            "tool_name": "cloud_asset_inventory",
            "tool_args": {"region": "us-east-1"},
            "raw_tokens": 60,
            "compacted_tokens": 0,
            "snr_score": 0.90,
            "action": FirewallAction.ALLOW,
        },
        # Step 4: Bloated Tool Result (Low SNR -> Compacted)
        {
            "node_id": "scanner",
            "role": FrameRole.TOOL_RESULT,
            "content": "Found 10 buckets. Verbose schema and internal IAM debug dump payload.",
            "tool_name": "cloud_asset_inventory",
            "tool_args": None,
            "raw_tokens": 4500,
            "compacted_tokens": 350,  # Compacted down!
            "snr_score": 0.08,  # Low SNR
            "action": FirewallAction.COMPACT,
        },
        # Step 5: Search tool invocation (Loop start)
        {
            "node_id": "search_worker",
            "role": FrameRole.TOOL_CALL,
            "content": "Search s3 bucket logs for exposed policies.",
            "tool_name": "s3_bucket_search",
            "tool_args": {"bucket": "logs", "prefix": "audit"},
            "raw_tokens": 110,
            "compacted_tokens": 0,
            "snr_score": 0.75,
            "action": None,  # Evaluated by detector
        },
        # Step 6: Empty search result
        {
            "node_id": "search_worker",
            "role": FrameRole.TOOL_RESULT,
            "content": "Empty search response: No objects found matching prefix.",
            "tool_name": "s3_bucket_search",
            "tool_args": None,
            "raw_tokens": 40,
            "compacted_tokens": 0,
            "snr_score": 0.35,
            "action": FirewallAction.ALLOW,
        },
        # Step 7: Repetitive retry 1
        {
            "node_id": "search_worker",
            "role": FrameRole.TOOL_CALL,
            "content": "Search s3 bucket logs for exposed policies.",
            "tool_name": "s3_bucket_search",
            "tool_args": {"bucket": "logs", "prefix": "audit"},
            "raw_tokens": 110,
            "compacted_tokens": 0,
            "snr_score": 0.25,
            "action": None,
        },
        # Step 8: Repetitive retry 2
        {
            "node_id": "search_worker",
            "role": FrameRole.TOOL_CALL,
            "content": "Search s3 bucket logs for exposed policies.",
            "tool_name": "s3_bucket_search",
            "tool_args": {"bucket": "logs", "prefix": "audit"},
            "raw_tokens": 110,
            "compacted_tokens": 0,
            "snr_score": 0.18,
            "action": None,
        },
        # Step 9: Repetitive retry 3 -> triggers INJECT_CORRECTION
        {
            "node_id": "search_worker",
            "role": FrameRole.TOOL_CALL,
            "content": "Search s3 bucket logs for exposed policies.",
            "tool_name": "s3_bucket_search",
            "tool_args": {"bucket": "logs", "prefix": "audit"},
            "raw_tokens": 110,
            "compacted_tokens": 0,
            "snr_score": 0.09,
            "action": None,
        },
        # Step 10: Repetitive retry 4 -> triggers CIRCUIT_BREAK
        {
            "node_id": "search_worker",
            "role": FrameRole.TOOL_CALL,
            "content": "Search s3 bucket logs for exposed policies.",
            "tool_name": "s3_bucket_search",
            "tool_args": {"bucket": "logs", "prefix": "audit"},
            "raw_tokens": 110,
            "compacted_tokens": 0,
            "snr_score": 0.05,
            "action": None,
        },
    ]

    for idx, spec in enumerate(step_specs, start=1):
        t0 = time.perf_counter()
        content = str(spec["content"])
        emb = embedder.embed(content)

        frame = SignalFrame(
            step_id=idx,
            node_id=str(spec["node_id"]),
            role=spec["role"],
            content=content,
            tool_name=spec.get("tool_name"),
            tool_args=spec.get("tool_args"),
            raw_tokens=int(spec.get("raw_tokens", 0)),
            compacted_tokens=int(spec.get("compacted_tokens", 0)),
            snr_score=float(spec.get("snr_score", 0.0)),
            embedding=emb,
        )
        ledger.add_frame(frame)

        # Evaluate loop entropy detector
        evaluated_action, metrics = detector.evaluate(ledger)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Action: prioritize explicit spec action if set, else evaluated action
        action_spec = spec.get("action")
        action_taken = action_spec if action_spec is not None else evaluated_action

        profiler.record_step(
            frame=frame,
            entropy=metrics["entropy"],
            drift=metrics["drift"],
            action_taken=action_taken,
            latency_ms=latency_ms,
        )

    profiler.print_report(console=console)

    if out_path:
        profiler.export_json(out_path)
        console.print(f"\n[bold green]✓ Exported telemetry trace to:[/] {out_path}")

    return profiler


def inspect_trace(trace_path: str | Path, console: Console | None = None) -> TelemetryProfiler:
    """Inspect and render a saved telemetry profile JSON file."""
    if console is None:
        console = Console()

    path = Path(trace_path)
    if not path.is_file():
        console.print(f"[bold red]Error:[/] Trace file '{trace_path}' not found.")
        sys.exit(1)

    profiler = TelemetryProfiler.from_json(path)
    profiler.print_report(console=console)
    return profiler


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for agent-snr."""
    parser = argparse.ArgumentParser(
        prog="agent-snr",
        description="Agent Signal Firewall: Diagnostic & Telemetry CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: demo
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a realistic 10-step synthetic multi-agent trace demo",
    )
    demo_parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Path to export telemetry trace JSON",
    )
    demo_parser.add_argument(
        "--cost",
        type=float,
        default=3.0,
        help="Cost per 1M input tokens in USD (default: $3.00)",
    )

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect and display an exported telemetry trace JSON file",
    )
    inspect_parser.add_argument(
        "trace_path",
        type=str,
        help="Path to telemetry trace JSON file",
    )

    args = parser.parse_args(argv)

    if args.command == "demo":
        run_demo(out_path=args.out, cost_per_million=args.cost)
        return 0
    if args.command == "inspect":
        inspect_trace(trace_path=args.trace_path)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
