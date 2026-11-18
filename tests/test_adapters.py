"""Unit and integration tests for zero-lock-in ecosystem adapters (OpenAI, LangGraph, CrewAI)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agent_snr import (
    AgentSNRFirewall,
    AsyncSNRWrappedOpenAI,
    CrewAISNRHook,
    LangGraphSNRAdapter,
    SkipRule,
    SNRCircuitBreakError,
    SNRWrappedOpenAI,
    StateCompactor,
)
from examples.langgraph_hierarchical_skip import main as run_langgraph_example
from examples.openai_quickstart import main as run_openai_example


class MockMessageDuck:
    """Duck-typed LangChain BaseMessage object."""

    def __init__(self, content: str, type: str, name: str | None = None) -> None:
        self.content = content
        self.type = type
        self.name = name


class StructuredDiagnostic(BaseModel):
    status: str
    code: int


class TestOpenAIAdapter:
    """Tests for SNRWrappedOpenAI and AsyncSNRWrappedOpenAI wrappers."""

    def test_sync_openai_wrapper_intercepts_messages(self) -> None:
        mock_client = MagicMock()
        mock_completions = MagicMock()
        mock_client.chat.completions = mock_completions
        mock_completions.create.return_value = "response_obj"

        firewall = AgentSNRFirewall(compactor=StateCompactor(max_list_items=1))
        wrapper = SNRWrappedOpenAI(mock_client, firewall=firewall, session_id="test_sess")

        assert wrapper.firewall is firewall
        assert wrapper.profiler is firewall.profiler

        bloated_json = '{"status": "ok", "request_id": "drop-me", "items": [{"id": 1}, {"id": 2}]}'
        messages = [
            {"role": "user", "content": "Fetch data"},
            {"role": "tool_result", "content": bloated_json},
        ]

        result = wrapper.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            snr_session_id="custom_sess",
            snr_node_id="custom_node",
        )

        assert result == "response_obj"
        mock_completions.create.assert_called_once()
        call_kwargs = mock_completions.create.call_args.kwargs

        # Verify custom SNR kwargs were popped
        assert "snr_session_id" not in call_kwargs
        assert "snr_node_id" not in call_kwargs

        # Verify messages were compacted before calling client
        passed_msgs = call_kwargs["messages"]
        assert "request_id" not in passed_msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_async_openai_wrapper_intercepts_messages(self) -> None:
        mock_client = MagicMock()
        mock_completions = MagicMock()
        mock_client.chat.completions = mock_completions
        mock_completions.create = AsyncMock(return_value="async_response")

        firewall = AgentSNRFirewall()
        wrapper = AsyncSNRWrappedOpenAI(mock_client, firewall=firewall, session_id="async_sess")

        messages = [
            {"role": "user", "content": "Async prompt"},
            {"role": "assistant", "content": "Async answer"},
        ]

        result = await wrapper.chat.completions.create(model="gpt-4o", messages=messages)
        assert result == "async_response"
        mock_completions.create.assert_called_once()


class TestLangGraphAdapter:
    """Tests for LangGraphSNRAdapter node guards and skip routers."""

    def test_normalize_messages_with_dicts_and_duck_messages(self) -> None:
        adapter = LangGraphSNRAdapter()

        raw_msgs = [
            {"role": "user", "content": "Hello dict"},
            MockMessageDuck(content="Hello human duck", type="human"),
            MockMessageDuck(content="Hello AI duck", type="ai"),
            MockMessageDuck(content="Tool result", type="tool", name="calculator"),
            MockMessageDuck(content="System instruction", type="system"),
        ]

        normalized = adapter._normalize_messages(raw_msgs)
        assert len(normalized) == 5
        assert normalized[0]["role"] == "user"
        assert normalized[1]["role"] == "user"
        assert normalized[2]["role"] == "assistant"
        assert normalized[3]["role"] == "tool_result"
        assert normalized[3]["name"] == "calculator"
        assert normalized[4]["role"] == "system"

    def test_guard_node_sync_execution(self) -> None:
        adapter = LangGraphSNRAdapter()

        @adapter.guard_node("planner_node")
        def planner(state: dict[str, Any]) -> dict[str, Any]:
            # Verify guarded messages exist
            assert len(state["messages"]) == 1
            return {"plan": "execute"}

        initial_state = {"messages": [{"role": "user", "content": "Goal: Plan task"}]}
        result = planner(initial_state)
        assert result == {"plan": "execute"}

    def test_guard_node_sync_with_config(self) -> None:
        adapter = LangGraphSNRAdapter(session_key="thread_id")

        @adapter.guard_node("agent_node")
        def agent(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            assert config["configurable"]["thread_id"] == "thread_123"
            return {"status": "done"}

        state = {"messages": [{"role": "user", "content": "Hello"}]}
        config = {"configurable": {"thread_id": "thread_123"}}
        res = agent(state, config)
        assert res == {"status": "done"}
        assert "thread_123" in adapter.firewall.ledgers

    @pytest.mark.asyncio
    async def test_guard_node_async_execution(self) -> None:
        adapter = LangGraphSNRAdapter(session_key="thread_id")

        @adapter.guard_node("async_node")
        async def async_agent(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            return {"async_status": "complete"}

        state = {"messages": [{"role": "user", "content": "Async task"}]}
        config = {"configurable": {"thread_id": "async_thread"}}
        res = await async_agent(state, config)
        assert res == {"async_status": "complete"}

    def test_skip_router_sync_bypass_and_fallback(self) -> None:
        firewall = AgentSNRFirewall()
        rule = SkipRule(
            source_node="Validator",
            bypassed_node="ManualReview",
            target_node="Deployer",
            min_snr=0.40,
            min_goal_similarity=0.0,
            output_schema=StructuredDiagnostic,
        )
        firewall.router.register_rule(rule)
        adapter = LangGraphSNRAdapter(firewall=firewall, session_key="thread_id")

        def default_router(state: dict[str, Any]) -> str:
            return "ManualReview"

        guarded_router = adapter.skip_router("Validator", default_router)

        # 1. High SNR valid structured JSON -> bypasses ManualReview
        valid_json = json.dumps({"status": "PASS", "code": 200})
        state_pass = {
            "messages": [
                {"role": "user", "content": "Run validation check"},
                {"role": "assistant", "content": valid_json},
            ]
        }
        config_pass = {"configurable": {"thread_id": "thread_pass"}}
        # Guard node records the frame
        guarded_node = adapter.guard_node("Validator")(lambda s: s)
        guarded_node(state_pass, config=config_pass)

        route_pass = guarded_router(state_pass, config=config_pass)
        assert route_pass == "Deployer"  # Bypassed!

        # 2. Invalid text -> falls back to ManualReview
        state_fail = {
            "messages": [
                {"role": "user", "content": "Run validation check"},
                {"role": "assistant", "content": "Unstructured error message"},
            ]
        }
        config_fail = {"configurable": {"thread_id": "thread_fail"}}
        guarded_node(state_fail, config=config_fail)

        route_fail = guarded_router(state_fail, config=config_fail)
        assert route_fail == "ManualReview"  # Default fallback!

    @pytest.mark.asyncio
    async def test_skip_router_async(self) -> None:
        firewall = AgentSNRFirewall()
        adapter = LangGraphSNRAdapter(firewall=firewall)

        async def async_default_router(state: dict[str, Any]) -> str:
            return "DefaultNode"

        router_fn = adapter.skip_router("SourceNode", async_default_router)
        res = await router_fn({"messages": []})
        assert res == "DefaultNode"


class TestCrewAIAdapter:
    """Tests for CrewAISNRHook callback integration."""

    def test_crewai_hook_step_processing(self) -> None:
        firewall = AgentSNRFirewall()
        hook = CrewAISNRHook(firewall=firewall, session_id="crew_test")

        assert hook.firewall is firewall
        assert hook.profiler is firewall.profiler

        # Simulate CrewAI step output object
        step_obj = MagicMock()
        step_obj.agent.role = "Security Researcher"
        step_obj.action.tool = "nmap_scanner"
        step_obj.action.tool_input = {"host": "10.0.0.1"}
        step_obj.observation = "Port 80 and 443 open"

        hook(step_obj)

        ledger = firewall.get_or_create_ledger("crew_test")
        assert len(ledger.frames) == 2
        assert ledger.frames[0].role == "tool_call"
        assert ledger.frames[1].role == "tool_result"
        assert ledger.frames[1].node_id == "Security Researcher"

    def test_crewai_hook_circuit_break_on_loop(self) -> None:
        firewall = AgentSNRFirewall()
        hook = CrewAISNRHook(firewall=firewall, session_id="crew_loop")

        step_dict = {
            "agent": "worker",
            "tool": "search_db",
            "tool_input": {"q": "identical"},
            "output": "No records found",
        }

        # Repetitive steps driving loop entropy down
        with pytest.raises(SNRCircuitBreakError):
            for _ in range(12):
                hook(step_dict)

    def test_crewai_as_callback_helper(self) -> None:
        hook = CrewAISNRHook()
        cb = hook.as_callback()
        assert callable(cb)


class TestExamplesExecution:
    """Verify that both runnable example scripts execute with exit code 0."""

    def test_openai_quickstart_example_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["openai_quickstart.py"])
        exit_code = run_openai_example()
        assert exit_code == 0

    def test_langgraph_hierarchical_skip_example_runs(self) -> None:
        exit_code = run_langgraph_example()
        assert exit_code == 0
