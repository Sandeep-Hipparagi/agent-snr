"""Core data structures and embedding algorithms for Agent SNR."""

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

__all__ = [
    "BaseEmbedder",
    "FastEmbedAdapter",
    "FirewallAction",
    "FrameRole",
    "HashNgramEmbedder",
    "SignalFrame",
    "TraceLedger",
    "cosine_similarity",
]
