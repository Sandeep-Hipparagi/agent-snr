"""CrewAI step callback hook adapter for the Agent Signal Firewall."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_snr.core.firewall import AgentSNRFirewall
from agent_snr.core.telemetry import TelemetryProfiler


class CrewAISNRHook:
    """Callback hook integrating AgentSNRFirewall into CrewAI step_callback execution."""

    def __init__(
        self,
        firewall: AgentSNRFirewall | None = None,
        session_id: str = "crew_session",
    ) -> None:
        """Initialize CrewAISNRHook.

        Args:
            firewall: AgentSNRFirewall instance (created if None).
            session_id: Session identifier used for telemetry and trajectory ledger.
        """
        self._firewall = firewall if firewall is not None else AgentSNRFirewall()
        self.session_id = session_id
        self._histories: dict[str, list[dict[str, Any]]] = {}

    @property
    def firewall(self) -> AgentSNRFirewall:
        """Access the underlying AgentSNRFirewall."""
        return self._firewall

    @property
    def profiler(self) -> TelemetryProfiler:
        """Access the firewall's TelemetryProfiler."""
        return self._firewall.profiler

    def reset(self, session_id: str | None = None) -> None:
        """Reset step message history for a session."""
        target = session_id or self.session_id
        self._histories.pop(target, None)

    def _extract_step_info(
        self, step_output: Any
    ) -> tuple[str, str | None, dict[str, Any] | None, str]:
        """Extract node_id, tool_name, tool_args, and content from arbitrary CrewAI outputs."""
        # 1. Agent / Node identifier
        node_id = "crew_agent"
        if hasattr(step_output, "agent"):
            agent_obj = step_output.agent
            node_id = str(
                getattr(agent_obj, "role", getattr(agent_obj, "name", str(agent_obj)))
            )
        elif isinstance(step_output, dict) and "agent" in step_output:
            node_id = str(step_output["agent"])
        elif hasattr(step_output, "role"):
            node_id = str(step_output.role)

        # 2. Tool information
        tool_name: str | None = None
        tool_args: dict[str, Any] | None = None

        action_obj = getattr(step_output, "action", None)
        if action_obj is not None:
            tool_name = getattr(action_obj, "tool", None)
            raw_input = getattr(action_obj, "tool_input", None)
            tool_args = raw_input if isinstance(raw_input, dict) else {"input": str(raw_input)}
        elif isinstance(step_output, dict):
            tool_name = step_output.get("tool")
            raw_input = step_output.get("tool_input")
            tool_args = raw_input if isinstance(raw_input, dict) else {"input": str(raw_input)}
        elif isinstance(step_output, (tuple, list)) and len(step_output) > 1:
            action_part = step_output[0]
            tool_name = str(getattr(action_part, "tool", str(action_part)))
            raw_input = getattr(action_part, "tool_input", None)
            tool_args = raw_input if isinstance(raw_input, dict) else {"input": str(raw_input)}

        # 3. Observation / Content
        if hasattr(step_output, "observation"):
            content = str(step_output.observation)
        elif hasattr(step_output, "result"):
            content = str(step_output.result)
        elif hasattr(step_output, "output"):
            content = str(step_output.output)
        elif isinstance(step_output, dict):
            content = str(
                step_output.get(
                    "output",
                    step_output.get(
                        "result", step_output.get("observation", str(step_output))
                    ),
                )
            )
        elif isinstance(step_output, (tuple, list)) and len(step_output) > 1:
            content = str(step_output[1])
        else:
            content = str(step_output)

        return node_id, tool_name, tool_args, content

    def __call__(self, step_output: Any) -> None:
        """Execute firewall telemetry and loop detection on each CrewAI step."""
        node_id, tool_name, tool_args, content = self._extract_step_info(step_output)

        messages: list[dict[str, Any]] = []
        if tool_name:
            messages.append(
                {
                    "role": "tool_call",
                    "name": tool_name,
                    "tool_args": tool_args,
                    "content": f"Calling tool {tool_name}",
                }
            )
            messages.append(
                {
                    "role": "tool_result",
                    "name": tool_name,
                    "tool_args": tool_args,
                    "content": content,
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

        # Accumulate in step history for the session so incremental frames are ingested
        if self.session_id not in self._histories:
            self._histories[self.session_id] = []
        self._histories[self.session_id].extend(messages)

        # Process through firewall (raises SNRCircuitBreakError if threshold exceeded)
        self.firewall.process_outbound(
            session_id=self.session_id,
            node_id=node_id,
            messages=self._histories[self.session_id],
        )

    def as_callback(self) -> Callable[[Any], None]:
        """Return a callable reference for CrewAI step_callback."""
        return self.__call__
