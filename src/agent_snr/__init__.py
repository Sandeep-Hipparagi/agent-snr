"""Agent SNR: An open-source, lightweight, framework-agnostic Agent Signal Firewall."""

from agent_snr.adapters.crewai import CrewAISNRHook
from agent_snr.adapters.langgraph import LangGraphSNRAdapter
from agent_snr.adapters.openai import AsyncSNRWrappedOpenAI, SNRWrappedOpenAI
from agent_snr.core.embedder import (
    BaseEmbedder,
    FastEmbedAdapter,
    HashNgramEmbedder,
    cosine_similarity,
)
from agent_snr.core.firewall import (
    AgentSNRFirewall,
    SNRCircuitBreakError,
)
from agent_snr.core.frames import (
    FirewallAction,
    FrameRole,
    SignalFrame,
    TraceLedger,
)
from agent_snr.core.telemetry import (
    StepTelemetry,
    TelemetryProfiler,
    profile_step,
)
from agent_snr.modules.budgeter import DynamicAttentionBudgeter
from agent_snr.modules.compactor import StateCompactor
from agent_snr.modules.entropy import LoopEntropyDetector
from agent_snr.modules.router import SkipLevelRouter, SkipRule

__version__ = "0.1.0"

__all__ = [
    "AgentSNRFirewall",
    "AsyncSNRWrappedOpenAI",
    "BaseEmbedder",
    "CrewAISNRHook",
    "DynamicAttentionBudgeter",
    "FastEmbedAdapter",
    "FirewallAction",
    "FrameRole",
    "HashNgramEmbedder",
    "LangGraphSNRAdapter",
    "LoopEntropyDetector",
    "SNRCircuitBreakError",
    "SNRWrappedOpenAI",
    "SignalFrame",
    "SkipLevelRouter",
    "SkipRule",
    "StateCompactor",
    "StepTelemetry",
    "TelemetryProfiler",
    "TraceLedger",
    "__version__",
    "cosine_similarity",
    "profile_step",
]
