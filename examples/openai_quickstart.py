"""Runnable example demonstrating SNRWrappedOpenAI with mock and optional live OpenAI clients."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from rich.console import Console

from agent_snr import AgentSNRFirewall, SNRCircuitBreakError, SNRWrappedOpenAI

console = Console()


class MockChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class MockChoice:
    def __init__(self, content: str) -> None:
        self.message = MockChoiceMessage(content)


class MockChatCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [MockChoice(content)]


class MockCompletions:
    def create(self, *args: Any, **kwargs: Any) -> MockChatCompletion:
        messages = kwargs.get("messages", [])
        last_msg = messages[-1]["content"] if messages else ""
        return MockChatCompletion(f"Processed input: {last_msg[:60]}...")


class MockChat:
    def __init__(self) -> None:
        self.completions = MockCompletions()


class MockOpenAIClient:
    def __init__(self) -> None:
        self.chat = MockChat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent SNR OpenAI Quickstart Example")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real OpenAI client (requires OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    if args.live:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            console.print("[bold red]Error:[/] OPENAI_API_KEY environment variable not set.")
            return 1
        import openai

        base_client: Any = openai.OpenAI(api_key=api_key)
        console.print("[bold green]Using Live OpenAI API[/]")
    else:
        base_client = MockOpenAIClient()
        console.print("[bold cyan]Using Mock OpenAI Client (pass --live for real API)[/]")

    # Initialize firewall and wrapped client
    firewall = AgentSNRFirewall()
    client = SNRWrappedOpenAI(base_client, firewall=firewall, session_id="openai_demo_session")

    console.print("\n[bold]1. Sending Initial User Request...[/]")
    conv_history = [
        {"role": "user", "content": "Analyze AWS EC2 instances and report open security groups."}
    ]
    resp1 = client.chat.completions.create(model="gpt-4o", messages=conv_history)
    conv_history.append({"role": "assistant", "content": resp1.choices[0].message.content})

    console.print("\n[bold]2. Injecting Bloated 5KB JSON Tool Result...[/]")
    # Generate large noisy JSON payload
    bloated_payload = {
        "request_id": "req-9999-abcdef",
        "trace_id": "trace-8888-123456",
        "metadata": {"etag": "xyz123", "cache_hit": False, "datacenter": "us-east-1"},
        "headers": {"x-amz-id-2": "abc", "server": "AmazonS3"},
        "reservations": [
            {
                "id": f"inst-{i}",
                "state": "running",
                "empty_field": "",
                "none_field": None,
                "raw_html": "<p>noise</p>",
            }
            for i in range(50)  # Exceeds max_list_items (15)
        ],
    }
    conv_history.append(
        {
            "role": "tool_result",
            "name": "ec2_describe_instances",
            "content": json.dumps(bloated_payload),
        }
    )
    resp2 = client.chat.completions.create(model="gpt-4o", messages=conv_history)
    conv_history.append({"role": "assistant", "content": resp2.choices[0].message.content})

    console.print("\n[bold]3. Simulating Repetitive Search Retry Loop...[/]")
    retry_query = {
        "role": "tool_call",
        "name": "security_group_search",
        "tool_args": {"group_id": "sg-12345"},
        "content": "Searching for security group sg-12345",
    }
    retry_result = {
        "role": "tool_result",
        "name": "security_group_search",
        "content": "No ingress rules found for sg-12345",
    }

    # Simulate loop until circuit break or warnings
    broken = False
    for loop_idx in range(1, 6):
        console.print(f"  Executing Retry Step {loop_idx}...")
        conv_history.append(retry_query)
        conv_history.append(retry_result)
        try:
            client.chat.completions.create(model="gpt-4o", messages=conv_history)
        except SNRCircuitBreakError as err:
            console.print(f"\n[bold red]🚨 Circuit Breaker Triggered at loop {loop_idx}![/]")
            console.print(f"[dim]{err}[/]")
            broken = True
            break

    if not broken:
        console.print("[yellow]Loop completed without break.[/]")

    console.print("\n[bold blue]Final Diagnostic Telemetry Report:[/]")
    client.profiler.print_report(console=console)
    return 0


if __name__ == "__main__":
    sys.exit(main())
