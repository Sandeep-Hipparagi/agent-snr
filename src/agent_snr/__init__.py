"""Agent SNR: An open-source, lightweight, framework-agnostic Agent Signal Firewall."""

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
from agent_snr.modules.entropy import LoopEntropyDetector

__version__ = "0.1.0"

__all__ = [
    "BaseEmbedder",
    "FastEmbedAdapter",
    "FirewallAction",
    "FrameRole",
    "HashNgramEmbedder",
    "LoopEntropyDetector",
    "SignalFrame",
    "StepTelemetry",
    "TelemetryProfiler",
    "TraceLedger",
    "__version__",
    "cosine_similarity",
    "profile_step",
]
