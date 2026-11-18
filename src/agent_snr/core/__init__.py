"""Core data structures, telemetry profiler, and embedding algorithms for Agent SNR."""

from agent_snr.core.embedder import (
    BaseEmbedder,
    FastEmbedAdapter,
    HashNgramEmbedder,
    cosine_similarity,
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

__all__ = [
    "BaseEmbedder",
    "FastEmbedAdapter",
    "FirewallAction",
    "FrameRole",
    "HashNgramEmbedder",
    "SignalFrame",
    "StepTelemetry",
    "TelemetryProfiler",
    "TraceLedger",
    "cosine_similarity",
    "profile_step",
]
