"""OpenAI and LiteLLM drop-in client wrappers for the Agent Signal Firewall."""

from __future__ import annotations

from typing import Any

from agent_snr.core.firewall import AgentSNRFirewall
from agent_snr.core.telemetry import TelemetryProfiler


class _ChatCompletionsProxy:
    """Synchronous proxy intercepting client.chat.completions calls."""

    def __init__(self, target: Any, wrapper: SNRWrappedOpenAI) -> None:
        self._target = target
        self._wrapper = wrapper

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Filter outbound messages through AgentSNRFirewall before inference."""
        session_id = kwargs.pop("snr_session_id", None) or self._wrapper.session_id
        node_id = kwargs.pop("snr_node_id", "openai_chat")
        messages = kwargs.get("messages")

        if messages is not None:
            optimized_messages, _action, _metrics = (
                self._wrapper.firewall.process_outbound(
                    session_id=session_id,
                    node_id=node_id,
                    messages=messages,
                )
            )
            kwargs["messages"] = optimized_messages

        return self._target.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ChatProxy:
    """Synchronous proxy intercepting client.chat."""

    def __init__(self, target: Any, wrapper: SNRWrappedOpenAI) -> None:
        self._target = target
        self._wrapper = wrapper
        target_completions = getattr(target, "completions", target)
        self.completions = _ChatCompletionsProxy(target_completions, wrapper)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class SNRWrappedOpenAI:
    """Drop-in wrapper for sync OpenAI-compatible clients (OpenAI, LiteLLM, vLLM, Ollama)."""

    def __init__(
        self,
        client: Any,
        firewall: AgentSNRFirewall | None = None,
        session_id: str = "default",
    ) -> None:
        """Initialize SNRWrappedOpenAI.

        Args:
            client: Native OpenAI or LiteLLM client instance.
            firewall: AgentSNRFirewall instance (created if None).
            session_id: Default session identifier for telemetry and ledger history.
        """
        self.client = client
        self._firewall = firewall if firewall is not None else AgentSNRFirewall()
        self.session_id = session_id

        target_chat = getattr(client, "chat", client)
        self.chat = _ChatProxy(target_chat, self)

    @property
    def firewall(self) -> AgentSNRFirewall:
        """Access the underlying AgentSNRFirewall instance."""
        return self._firewall

    @property
    def profiler(self) -> TelemetryProfiler:
        """Access the firewall's TelemetryProfiler."""
        return self._firewall.profiler

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


class _AsyncChatCompletionsProxy:
    """Asynchronous proxy intercepting client.chat.completions calls."""

    def __init__(self, target: Any, wrapper: AsyncSNRWrappedOpenAI) -> None:
        self._target = target
        self._wrapper = wrapper

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """Filter outbound messages through AgentSNRFirewall before async inference."""
        session_id = kwargs.pop("snr_session_id", None) or self._wrapper.session_id
        node_id = kwargs.pop("snr_node_id", "openai_chat")
        messages = kwargs.get("messages")

        if messages is not None:
            optimized_messages, _action, _metrics = (
                self._wrapper.firewall.process_outbound(
                    session_id=session_id,
                    node_id=node_id,
                    messages=messages,
                )
            )
            kwargs["messages"] = optimized_messages

        return await self._target.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _AsyncChatProxy:
    """Asynchronous proxy intercepting client.chat."""

    def __init__(self, target: Any, wrapper: AsyncSNRWrappedOpenAI) -> None:
        self._target = target
        self._wrapper = wrapper
        target_completions = getattr(target, "completions", target)
        self.completions = _AsyncChatCompletionsProxy(target_completions, wrapper)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class AsyncSNRWrappedOpenAI:
    """Drop-in wrapper for async OpenAI-compatible clients (AsyncOpenAI, async LiteLLM)."""

    def __init__(
        self,
        client: Any,
        firewall: AgentSNRFirewall | None = None,
        session_id: str = "default",
    ) -> None:
        """Initialize AsyncSNRWrappedOpenAI.

        Args:
            client: Native AsyncOpenAI or async LiteLLM client instance.
            firewall: AgentSNRFirewall instance (created if None).
            session_id: Default session identifier for telemetry and ledger history.
        """
        self.client = client
        self._firewall = firewall if firewall is not None else AgentSNRFirewall()
        self.session_id = session_id

        target_chat = getattr(client, "chat", client)
        self.chat = _AsyncChatProxy(target_chat, self)

    @property
    def firewall(self) -> AgentSNRFirewall:
        """Access the underlying AgentSNRFirewall instance."""
        return self._firewall

    @property
    def profiler(self) -> TelemetryProfiler:
        """Access the firewall's TelemetryProfiler."""
        return self._firewall.profiler

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
