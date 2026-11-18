"""Self-contained runnable example of hierarchical skip-level routing with LangGraphSNRAdapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console

from agent_snr import (
    AgentSNRFirewall,
    LangGraphSNRAdapter,
    SkipRule,
)

console = Console()


# 1. Target Pydantic Schema for Structured Skip-Level Validation
class ResearchBrief(BaseModel):
    """Verified structured output schema for research synthesis."""

    topic: str
    key_findings: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


# 2. Setup Firewall, SkipRule, and LangGraphSNRAdapter
firewall = AgentSNRFirewall()
skip_rule = SkipRule(
    source_node="Researcher",
    bypassed_node="ManagerReviewer",
    target_node="FinalSynthesizer",
    min_snr=0.40,
    min_goal_similarity=0.40,
    output_schema=ResearchBrief,
)
firewall.router.register_rule(skip_rule)
adapter = LangGraphSNRAdapter(firewall=firewall, session_key="thread_id")


# 3. Define Decorated Graph Node Functions
@adapter.guard_node("Researcher")
def researcher_node(state: dict[str, Any]) -> dict[str, Any]:
    console.print("  [cyan]↳ Executing Researcher Node...[/]")
    # Emits research payload stored in state
    return {"messages": state["messages"]}


@adapter.guard_node("ManagerReviewer")
def manager_node(state: dict[str, Any]) -> dict[str, Any]:
    console.print("  [yellow]↳ Executing ManagerReviewer Node (Manual Quality Review)...[/]")
    state["messages"].append(
        {"role": "assistant", "content": "Manager Review: Approved output."}
    )
    return {"messages": state["messages"]}


@adapter.guard_node("FinalSynthesizer")
def synthesizer_node(state: dict[str, Any]) -> dict[str, Any]:
    console.print("  [green]↳ Executing FinalSynthesizer Node (Delivery)...[/]")
    state["messages"].append(
        {"role": "assistant", "content": "Final Synthesizer: Delivered to stakeholder."}
    )
    return {"messages": state["messages"]}


# 4. Define Conditional Edge Router with Skip Router Interceptor
def default_review_router(state: dict[str, Any]) -> str:
    """Default fallback route without firewall skip-level optimization."""
    return "ManagerReviewer"


guarded_router = adapter.skip_router("Researcher", default_review_router)


def simulate_graph_execution(
    scenario_title: str,
    thread_id: str,
    research_content: str,
) -> None:
    """Simulate LangGraph execution loop for a scenario."""
    console.print(f"\n[bold]{scenario_title}[/]")
    goal = "Investigate quantum computing fault tolerance and summarize findings."

    state: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": research_content},
        ]
    }
    config = {"configurable": {"thread_id": thread_id}}

    # Step A: Run Researcher node
    state = researcher_node(state, config=config)

    # Step B: Evaluate conditional edge router
    next_node = guarded_router(state, config=config)
    console.print(f"  [bold magenta]Dynamic Route Resolution:[/] -> [bold]{next_node}[/]")

    # Step C: Dispatch based on resolved route
    if next_node == "ManagerReviewer":
        state = manager_node(state, config=config)
        state = synthesizer_node(state, config=config)
    elif next_node == "FinalSynthesizer":
        console.print(
            "  [bold green]⚡ Skip-Level Optimization:[/] Bypassed ManagerReviewer successfully!"
        )
        state = synthesizer_node(state, config=config)


def main() -> int:
    console.print("[bold blue]🛡️  LangGraph Hierarchical Skip-Level Routing Demo[/]")

    # Scenario 1: High-SNR Structured Pydantic Payload -> Triggers Direct Skip
    valid_brief = ResearchBrief(
        topic="Quantum Computing Fault Tolerance",
        key_findings=[
            "Surface code error thresholds achieved at 1.0%",
            "Logical qubit coherence times exceeded physical coherence by 4x",
        ],
        confidence=0.96,
    )
    simulate_graph_execution(
        scenario_title="Scenario 1: High-SNR Valid ResearchBrief Schema Output",
        thread_id="thread_structured_001",
        research_content=valid_brief.model_dump_json(),
    )

    # Scenario 2: Unstructured Rough Draft -> Fails Schema Check -> Routes to Manager
    unstructured_draft = (
        "Rough notes on quantum computing: lots of noise, maybe some error correction happened, "
        "not really sure about confidence or final measurements."
    )
    simulate_graph_execution(
        scenario_title="Scenario 2: Low-Structure Draft Output (Schema Mismatch)",
        thread_id="thread_unstructured_002",
        research_content=unstructured_draft,
    )

    console.print("\n[bold blue]Telemetry Diagnostic Report Across Sessions:[/]")
    adapter.profiler.print_report(console=console)
    return 0


if __name__ == "__main__":
    main()
