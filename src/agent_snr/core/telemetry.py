"""Telemetry profiling and Rich terminal reporting for the Agent Signal Firewall."""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_snr.core.frames import FirewallAction, FrameRole, SignalFrame

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(slots=True)
class StepTelemetry:
    """Telemetry data captured for a single execution step in an agent trajectory."""

    step_id: int
    node_id: str
    role: FrameRole | str
    raw_tokens: int
    compacted_tokens: int
    tokens_saved: int
    snr_score: float
    entropy: float
    drift: float
    action_taken: FirewallAction | str
    latency_ms: float

    def __post_init__(self) -> None:
        if isinstance(self.role, str) and not isinstance(self.role, FrameRole):
            self.role = FrameRole(self.role)
        if isinstance(self.action_taken, str) and not isinstance(self.action_taken, FirewallAction):
            self.action_taken = FirewallAction(self.action_taken)

    def to_dict(self) -> dict[str, Any]:
        """Convert step telemetry into a serializable dictionary."""
        return {
            "step_id": self.step_id,
            "node_id": self.node_id,
            "role": self.role.value if isinstance(self.role, FrameRole) else str(self.role),
            "raw_tokens": self.raw_tokens,
            "compacted_tokens": self.compacted_tokens,
            "tokens_saved": self.tokens_saved,
            "snr_score": self.snr_score,
            "entropy": self.entropy,
            "drift": self.drift,
            "action_taken": (
                self.action_taken.value
                if isinstance(self.action_taken, FirewallAction)
                else str(self.action_taken)
            ),
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepTelemetry:
        """Construct StepTelemetry from a serialized dictionary."""
        return cls(
            step_id=int(data["step_id"]),
            node_id=str(data["node_id"]),
            role=FrameRole(data["role"]),
            raw_tokens=int(data["raw_tokens"]),
            compacted_tokens=int(data["compacted_tokens"]),
            tokens_saved=int(data["tokens_saved"]),
            snr_score=float(data["snr_score"]),
            entropy=float(data["entropy"]),
            drift=float(data["drift"]),
            action_taken=FirewallAction(data["action_taken"]),
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


class TelemetryProfiler:
    """Profiles step-level agent token usage, signal quality, and firewall actions."""

    def __init__(
        self,
        cost_per_million_input: float = 3.0,
        steps: list[StepTelemetry] | None = None,
    ) -> None:
        """Initialize TelemetryProfiler.

        Args:
            cost_per_million_input: Estimated cost in USD per 1M input tokens (default: $3.00).
            steps: Optional initial list of StepTelemetry objects.
        """
        self.cost_per_million_input = cost_per_million_input
        self.steps: list[StepTelemetry] = steps if steps is not None else []

    def record_step(
        self,
        frame: SignalFrame | None = None,
        *,
        step: StepTelemetry | None = None,
        step_id: int | None = None,
        node_id: str = "",
        role: FrameRole | str = FrameRole.ASSISTANT,
        raw_tokens: int = 0,
        compacted_tokens: int = 0,
        tokens_saved: int | None = None,
        snr_score: float = 1.0,
        entropy: float = 1.0,
        drift: float = 0.0,
        action_taken: FirewallAction | str = FirewallAction.ALLOW,
        latency_ms: float = 0.0,
    ) -> StepTelemetry:
        """Record a single step's telemetry metrics.

        Can be called with an explicit StepTelemetry object, a SignalFrame, or individual kwargs.
        """
        if step is not None:
            self.steps.append(step)
            return step

        if frame is not None:
            actual_step_id = frame.step_id if step_id is None else step_id
            actual_node_id = frame.node_id if not node_id else node_id
            actual_role = frame.role
            actual_raw = frame.raw_tokens if raw_tokens == 0 else raw_tokens
            actual_compacted = (
                frame.compacted_tokens if compacted_tokens == 0 else compacted_tokens
            )
            actual_saved = frame.tokens_saved if tokens_saved is None else tokens_saved
            actual_snr = frame.snr_score if snr_score == 1.0 else snr_score
        else:
            actual_step_id = (len(self.steps) + 1) if step_id is None else step_id
            actual_node_id = node_id
            actual_role = role
            actual_raw = raw_tokens
            actual_compacted = compacted_tokens
            if tokens_saved is None:
                actual_saved = (
                    max(0, actual_raw - actual_compacted) if actual_compacted > 0 else 0
                )
            else:
                actual_saved = tokens_saved
            actual_snr = snr_score

        entry = StepTelemetry(
            step_id=actual_step_id,
            node_id=actual_node_id,
            role=actual_role,
            raw_tokens=actual_raw,
            compacted_tokens=actual_compacted,
            tokens_saved=actual_saved,
            snr_score=actual_snr,
            entropy=entropy,
            drift=drift,
            action_taken=action_taken,
            latency_ms=latency_ms,
        )
        self.steps.append(entry)
        return entry

    def summary(self) -> dict[str, Any]:
        """Compute aggregated telemetry summary metrics."""
        total_steps = len(self.steps)
        total_raw = sum(s.raw_tokens for s in self.steps)
        total_compacted = sum(s.compacted_tokens for s in self.steps)
        total_saved = sum(s.tokens_saved for s in self.steps)

        signal_eff = (total_saved / total_raw * 100.0) if total_raw > 0 else 0.0

        loop_warnings = sum(
            1 for s in self.steps if s.action_taken == FirewallAction.INJECT_CORRECTION
        )
        circuit_breaks = sum(
            1 for s in self.steps if s.action_taken == FirewallAction.CIRCUIT_BREAK
        )
        mean_snr = (sum(s.snr_score for s in self.steps) / total_steps) if total_steps > 0 else 0.0
        usd_saved = (total_saved / 1_000_000.0) * self.cost_per_million_input
        avg_overhead = (
            (sum(s.latency_ms for s in self.steps) / total_steps) if total_steps > 0 else 0.0
        )

        return {
            "total_steps": total_steps,
            "total_raw_tokens": total_raw,
            "total_compacted_tokens": total_compacted,
            "total_tokens_saved": total_saved,
            "signal_efficiency_pct": round(signal_eff, 2),
            "loop_warning_count": loop_warnings,
            "circuit_break_count": circuit_breaks,
            "loop_interventions": loop_warnings + circuit_breaks,
            "mean_snr": round(mean_snr, 4),
            "estimated_usd_saved": round(usd_saved, 4),
            "avg_overhead_ms": round(avg_overhead, 2),
            "cost_per_million_input": self.cost_per_million_input,
        }

    def export_json(self, path: str | Path) -> None:
        """Export telemetry steps and summary to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "cost_per_million_input": self.cost_per_million_input,
            "steps": [s.to_dict() for s in self.steps],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> TelemetryProfiler:
        """Load TelemetryProfiler from an exported JSON file."""
        target = Path(path)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        cost = float(data.get("cost_per_million_input", 3.0))
        steps = [StepTelemetry.from_dict(s) for s in data.get("steps", [])]
        return cls(cost_per_million_input=cost, steps=steps)

    def print_report(self, console: Console | None = None) -> None:
        """Render diagnostic report to the terminal using Rich."""
        if console is None:
            console = Console()

        s = self.summary()

        # 1. Executive Summary Panel
        eff_str = f"[bold green]{s['signal_efficiency_pct']:.1f}%[/]"
        saved_str = f"[bold green]{s['total_tokens_saved']:,}[/] / {s['total_raw_tokens']:,}"
        cost_str = f"[bold yellow]${s['estimated_usd_saved']:.4f}[/]"
        rate_str = f"(@ ${s['cost_per_million_input']:.2f}/1M tokens)"
        warn_str = f"[bold yellow]{s['loop_warning_count']} warnings[/]"
        break_str = f"[bold red]{s['circuit_break_count']} breaks[/]"
        overhead_str = f"[dim]{s['avg_overhead_ms']:.2f} ms[/]"

        lines = [
            f"[bold cyan]Signal Efficiency:[/] {eff_str}   [bold cyan]Tokens Saved:[/] {saved_str}",
            f"[bold cyan]Estimated Cost Reduction:[/] {cost_str} {rate_str}",
            f"[bold cyan]Loop Interventions:[/] {warn_str}, {break_str}   "
            f"[bold cyan]Avg Profiler Overhead:[/] {overhead_str}",
        ]
        panel = Panel(
            "\n".join(lines),
            title="[bold blue]🛡️  Agent Signal Firewall Diagnostic Report[/]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)

        # 2. Per-Step Telemetry Table
        table = Table(
            title="Step-Level Signal Telemetry",
            header_style="bold magenta",
            show_header=True,
            show_lines=False,
        )
        table.add_column("Step", justify="right", style="cyan")
        table.add_column("Node", justify="left")
        table.add_column("Role", justify="left", style="dim")
        table.add_column("Raw", justify="right")
        table.add_column("Compacted", justify="right")
        table.add_column("Saved", justify="right")
        table.add_column("SNR", justify="right")
        table.add_column("Entropy", justify="right")
        table.add_column("Drift", justify="right")
        table.add_column("Action", justify="center")
        table.add_column("Latency", justify="right")

        for step in self.steps:
            # Color code SNR
            if step.snr_score < 0.15:
                snr_display = f"[bold red]{step.snr_score:.2f}[/]"
            elif step.snr_score < 0.40:
                snr_display = f"[yellow]{step.snr_score:.2f}[/]"
            else:
                snr_display = f"[green]{step.snr_score:.2f}[/]"

            # Highlight Action
            action_val = (
                step.action_taken.value
                if isinstance(step.action_taken, FirewallAction)
                else str(step.action_taken)
            )
            if action_val == FirewallAction.CIRCUIT_BREAK.value:
                action_display = "[bold white on red] 🚨 BREAK [/]"
            elif action_val == FirewallAction.INJECT_CORRECTION.value:
                action_display = "[bold yellow]⚠️ CORRECTION[/]"
            elif action_val == FirewallAction.COMPACT.value:
                action_display = "[cyan]COMPACT[/]"
            elif action_val == FirewallAction.SKIP_LEVEL.value:
                action_display = "[blue]SKIP[/]"
            else:
                action_display = "[green]ALLOW[/]"

            table.add_row(
                f"#{step.step_id}",
                step.node_id,
                step.role.value if isinstance(step.role, FrameRole) else str(step.role),
                f"{step.raw_tokens:,}",
                f"{step.compacted_tokens:,}",
                f"{step.tokens_saved:,}",
                snr_display,
                f"{step.entropy:.2f}",
                f"{step.drift:.2f}",
                action_display,
                f"{step.latency_ms:.1f}ms",
            )

        console.print(table)


class StepProfileContext:
    """Context manager and decorator helper for profiling agent steps."""

    def __init__(
        self,
        profiler: TelemetryProfiler,
        node_id: str = "execution",
        role: FrameRole | str = FrameRole.ASSISTANT,
        step_id: int | None = None,
        raw_tokens: int = 0,
        compacted_tokens: int = 0,
        snr_score: float = 1.0,
        entropy: float = 1.0,
        drift: float = 0.0,
        action_taken: FirewallAction | str = FirewallAction.ALLOW,
    ) -> None:
        self.profiler = profiler
        self.node_id = node_id
        self.role = role
        self.step_id = step_id
        self.raw_tokens = raw_tokens
        self.compacted_tokens = compacted_tokens
        self.snr_score = snr_score
        self.entropy = entropy
        self.drift = drift
        self.action_taken = action_taken
        self.start_time: float = 0.0
        self.latency_ms: float = 0.0
        self.result: StepTelemetry | None = None

    def __enter__(self) -> StepProfileContext:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.latency_ms = (time.perf_counter() - self.start_time) * 1000.0
        saved = (
            max(0, self.raw_tokens - self.compacted_tokens) if self.compacted_tokens > 0 else 0
        )
        self.result = self.profiler.record_step(
            step_id=self.step_id,
            node_id=self.node_id,
            role=self.role,
            raw_tokens=self.raw_tokens,
            compacted_tokens=self.compacted_tokens,
            tokens_saved=saved,
            snr_score=self.snr_score,
            entropy=self.entropy,
            drift=self.drift,
            action_taken=self.action_taken,
            latency_ms=self.latency_ms,
        )

    def set_tokens(self, raw: int, compacted: int = 0, snr_score: float = 1.0) -> None:
        """Update token counts and SNR for this step during execution."""
        self.raw_tokens = raw
        self.compacted_tokens = compacted
        self.snr_score = snr_score

    def __call__(self, fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


def profile_step(
    profiler: TelemetryProfiler,
    node_id: str = "execution",
    role: FrameRole | str = FrameRole.ASSISTANT,
    step_id: int | None = None,
    raw_tokens: int = 0,
    compacted_tokens: int = 0,
    snr_score: float = 1.0,
    entropy: float = 1.0,
    drift: float = 0.0,
    action_taken: FirewallAction | str = FirewallAction.ALLOW,
) -> StepProfileContext:
    """Create a profile context or decorator to record latency and metrics."""
    return StepProfileContext(
        profiler=profiler,
        node_id=node_id,
        role=role,
        step_id=step_id,
        raw_tokens=raw_tokens,
        compacted_tokens=compacted_tokens,
        snr_score=snr_score,
        entropy=entropy,
        drift=drift,
        action_taken=action_taken,
    )
