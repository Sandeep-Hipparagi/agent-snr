"""LangGraph node and conditional edge interceptor adapter for the Agent Signal Firewall."""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from agent_snr.core.firewall import AgentSNRFirewall
from agent_snr.core.telemetry import TelemetryProfiler

F = TypeVar("F", bound=Callable[..., Any])

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "tool": "tool_result",
    "tool_result": "tool_result",
    "function": "tool_result",
    "system": "system",
}

INVERSE_ROLE_MAP = {
    "user": "human",
    "assistant": "ai",
    "tool_result": "tool",
    "system": "system",
}


class LangGraphSNRAdapter:
    """Zero-lock-in interceptor for LangGraph node functions and conditional edge routers."""

    def __init__(
        self,
        firewall: AgentSNRFirewall | None = None,
        session_key: str = "thread_id",
        messages_key: str = "messages",
    ) -> None:
        """Initialize LangGraphSNRAdapter.

        Args:
            firewall: AgentSNRFirewall instance (created if None).
            session_key: Configurable key used to extract session ID (default: 'thread_id').
            messages_key: State dictionary key containing messages (default: 'messages').
        """
        self._firewall = firewall if firewall is not None else AgentSNRFirewall()
        self.session_key = session_key
        self.messages_key = messages_key

    @property
    def firewall(self) -> AgentSNRFirewall:
        """Access the underlying AgentSNRFirewall instance."""
        return self._firewall

    @property
    def profiler(self) -> TelemetryProfiler:
        """Access the firewall's TelemetryProfiler."""
        return self._firewall.profiler

    def _normalize_messages(self, raw_messages: Sequence[Any]) -> list[dict[str, Any]]:
        """Normalize both LangChain BaseMessage duck-types and dicts into canonical dicts."""
        normalized: list[dict[str, Any]] = []
        for item in raw_messages:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "role": str(item.get("role", "user")),
                        "content": str(item.get("content", "")),
                        "name": item.get("name"),
                        "tool_args": item.get("tool_args"),
                    }
                )
            else:
                # Duck-type LangChain BaseMessage (has .content and .type/.role)
                raw_type = getattr(item, "type", getattr(item, "role", "user"))
                mapped_role = ROLE_MAP.get(str(raw_type).lower(), "user")
                normalized.append(
                    {
                        "role": mapped_role,
                        "content": str(getattr(item, "content", "")),
                        "name": getattr(item, "name", None),
                        "tool_args": getattr(
                            item,
                            "tool_args",
                            getattr(item, "additional_kwargs", None),
                        ),
                    }
                )
        return normalized

    def _reconstruct_messages(
        self,
        original_raw: Sequence[Any],
        optimized_dicts: list[dict[str, Any]],
    ) -> list[Any]:
        """Restore messages into the original format (dicts or BaseMessage duck types)."""
        if not original_raw or isinstance(original_raw[0], dict):
            return optimized_dicts

        # Attempt to preserve or reconstruct objects matching original type
        first_obj = original_raw[0]
        reconstructed: list[Any] = []

        for d in optimized_dicts:
            role = d["role"]
            content = d["content"]
            # Look for an existing message in original_raw to clone
            matched_orig = None
            for orig in original_raw:
                orig_type = getattr(orig, "type", getattr(orig, "role", "user"))
                if ROLE_MAP.get(str(orig_type).lower(), "user") == role:
                    matched_orig = orig
                    break

            if matched_orig is not None:
                try:
                    cloned = copy.copy(matched_orig)
                    cloned.content = content
                    if d.get("name"):
                        cloned.name = d["name"]
                    reconstructed.append(cloned)
                    continue
                except Exception:
                    pass

            # Fallback to duck-type instantiation or dict
            try:
                msg_cls = type(first_obj)
                new_msg = msg_cls(content=content)
                if hasattr(new_msg, "type"):
                    new_msg.type = INVERSE_ROLE_MAP.get(role, role)
                reconstructed.append(new_msg)
            except Exception:
                reconstructed.append(d)

        return reconstructed

    def _extract_session_id(self, *args: Any, **kwargs: Any) -> str:
        """Extract session_id from LangGraph config dictionary."""
        config = None
        if len(args) > 1 and isinstance(args[1], dict):
            config = args[1]
        elif "config" in kwargs and isinstance(kwargs["config"], dict):
            config = kwargs["config"]

        if config and isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict) and self.session_key in configurable:
                return str(configurable[self.session_key])

        return "default"

    def guard_node(self, node_id: str) -> Callable[[F], F]:
        """Decorate a LangGraph node function (sync or async) with firewall protection."""

        def decorator(fn: F) -> F:
            is_async = inspect.iscoroutinefunction(fn)
            sig = inspect.signature(fn)
            params = sig.parameters
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            accepts_multiple_pos = len(params) > 1 or any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
            )

            def _prepare_guarded_state(args: tuple[Any, ...]) -> tuple[dict[str, Any], Any]:
                state = args[0]
                state_copy = copy.copy(state) if isinstance(state, dict) else dict(state)
                raw_msgs = state_copy.get(self.messages_key, [])
                return state_copy, raw_msgs

            def _filter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
                if accepts_var_kwargs:
                    return kwargs
                return {k: v for k, v in kwargs.items() if k in params}

            if is_async:

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if not args:
                        return await fn(*args, **kwargs)

                    session_id = self._extract_session_id(*args, **kwargs)
                    state_copy, raw_msgs = _prepare_guarded_state(args)

                    if raw_msgs:
                        norm_msgs = self._normalize_messages(raw_msgs)
                        opt_msgs, _action, _metrics = self.firewall.process_outbound(
                            session_id=session_id,
                            node_id=node_id,
                            messages=norm_msgs,
                        )
                        state_copy[self.messages_key] = self._reconstruct_messages(
                            raw_msgs, opt_msgs
                        )

                    filtered_kw = _filter_kwargs(kwargs)
                    if accepts_multiple_pos and len(args) > 1:
                        return await fn(state_copy, *args[1:], **filtered_kw)
                    return await fn(state_copy, **filtered_kw)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not args:
                    return fn(*args, **kwargs)

                session_id = self._extract_session_id(*args, **kwargs)
                state_copy, raw_msgs = _prepare_guarded_state(args)

                if raw_msgs:
                    norm_msgs = self._normalize_messages(raw_msgs)
                    opt_msgs, _action, _metrics = self.firewall.process_outbound(
                        session_id=session_id,
                        node_id=node_id,
                        messages=norm_msgs,
                    )
                    state_copy[self.messages_key] = self._reconstruct_messages(
                        raw_msgs, opt_msgs
                    )

                filtered_kw = _filter_kwargs(kwargs)
                if accepts_multiple_pos and len(args) > 1:
                    return fn(state_copy, *args[1:], **filtered_kw)
                return fn(state_copy, **filtered_kw)

            return sync_wrapper  # type: ignore[return-value]

        return decorator

    def skip_router(
        self,
        current_node: str,
        default_router: Callable[..., str],
    ) -> Callable[..., Any]:
        """Wrap a LangGraph conditional edge routing function with skip-level bypasses."""
        is_async = inspect.iscoroutinefunction(default_router)
        sig = inspect.signature(default_router)
        params = sig.parameters
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        accepts_multiple_pos = len(params) > 1 or any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
        )

        def _call_router(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            filtered_kw = (
                kwargs
                if accepts_var_kwargs
                else {k: v for k, v in kwargs.items() if k in params}
            )
            if accepts_multiple_pos and len(args) > 1:
                return default_router(*args, **filtered_kw)
            if args:
                return default_router(args[0], **filtered_kw)
            return default_router(**filtered_kw)

        if is_async:

            @functools.wraps(default_router)
            async def async_router_wrapper(*args: Any, **kwargs: Any) -> str:
                default_next = await _call_router(args, kwargs)
                session_id = self._extract_session_id(*args, **kwargs)
                next_node, _skipped = self.firewall.resolve_route(
                    session_id=session_id,
                    current_node=current_node,
                    default_next_node=default_next,
                )
                return next_node

            return async_router_wrapper

        @functools.wraps(default_router)
        def sync_router_wrapper(*args: Any, **kwargs: Any) -> str:
            default_next = _call_router(args, kwargs)
            session_id = self._extract_session_id(*args, **kwargs)
            next_node, _skipped = self.firewall.resolve_route(
                session_id=session_id,
                current_node=current_node,
                default_next_node=default_next,
            )
            return next_node

        return sync_router_wrapper
