"""Canonical data structures and enumerations for the Agent Signal Firewall."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FrameRole(str, Enum):
    """Role associated with a SignalFrame in an agent interaction trajectory."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    @classmethod
    def _missing_(cls, value: object) -> FrameRole | None:
        if isinstance(value, str):
            val_norm = value.strip().lower()
            for member in cls:
                if member.value == val_norm or member.name.lower() == val_norm:
                    return member
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other.lower() or self.name.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


class FirewallAction(str, Enum):
    """Action determined by the Agent Signal Firewall for a given frame or transition."""

    ALLOW = "allow"
    COMPACT = "compact"
    SKIP_LEVEL = "skip_level"
    INJECT_CORRECTION = "inject_correction"
    CIRCUIT_BREAK = "circuit_break"

    @classmethod
    def _missing_(cls, value: object) -> FirewallAction | None:
        if isinstance(value, str):
            val_norm = value.strip().lower()
            for member in cls:
                if member.value == val_norm or member.name.lower() == val_norm:
                    return member
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other.lower() or self.name.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(slots=True)
class SignalFrame:
    """Canonical frame representing a single step or observation in an agent trajectory."""

    step_id: int
    node_id: str
    role: FrameRole | str
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    raw_tokens: int = 0
    compacted_tokens: int = 0
    snr_score: float = 0.0
    embedding: list[float] | None = None
    timestamp: float = field(default_factory=time.time)
    is_pinned: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.role, str) and not isinstance(self.role, FrameRole):
            self.role = FrameRole(self.role)

    @property
    def tokens_saved(self) -> int:
        """Calculate the number of tokens saved through compaction for this frame."""
        if self.compacted_tokens > 0 and self.compacted_tokens < self.raw_tokens:
            return self.raw_tokens - self.compacted_tokens
        return 0

    @property
    def effective_tokens(self) -> int:
        """Effective token count consumed by this frame (compacted if set, else raw)."""
        if self.compacted_tokens > 0:
            return self.compacted_tokens
        return self.raw_tokens

    def compact(self, compacted_content: str, compacted_tokens: int) -> int:
        """Apply compaction to this frame and return the number of tokens saved.

        Args:
            compacted_content: The compacted or summarized text content.
            compacted_tokens: The token count of the compacted content.

        Returns:
            The number of tokens saved by compaction.
        """
        self.content = compacted_content
        self.compacted_tokens = compacted_tokens
        return self.tokens_saved

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame to a standard dictionary."""
        return {
            "step_id": self.step_id,
            "node_id": self.node_id,
            "role": self.role.value if isinstance(self.role, FrameRole) else str(self.role),
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "raw_tokens": self.raw_tokens,
            "compacted_tokens": self.compacted_tokens,
            "snr_score": self.snr_score,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "is_pinned": self.is_pinned,
            "tokens_saved": self.tokens_saved,
            "effective_tokens": self.effective_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalFrame:
        """Deserialize frame from a dictionary."""
        return cls(
            step_id=int(data["step_id"]),
            node_id=str(data.get("node_id", "")),
            role=FrameRole(data["role"]),
            content=str(data.get("content", "")),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            raw_tokens=int(data.get("raw_tokens", 0)),
            compacted_tokens=int(data.get("compacted_tokens", 0)),
            snr_score=float(data.get("snr_score", 0.0)),
            embedding=data.get("embedding"),
            timestamp=float(data.get("timestamp", time.time())),
            is_pinned=bool(data.get("is_pinned", False)),
        )


@dataclass
class TraceLedger:
    """Ledger recording session history, metrics, and firewall state."""

    session_id: str
    goal_embedding: list[float] | None = None
    frames: list[SignalFrame] = field(default_factory=list)
    state_kv: dict[str, Any] = field(default_factory=dict)
    entropy_history: list[float] = field(default_factory=list)
    drift_history: list[float] = field(default_factory=list)
    tokens_saved: int = 0
    hops_skipped: int = 0

    def add_frame(self, frame: SignalFrame) -> None:
        """Append a frame to the ledger and update tokens saved."""
        self.frames.append(frame)
        self.tokens_saved += frame.tokens_saved

    @property
    def total_raw_tokens(self) -> int:
        """Sum of all raw tokens across all recorded frames."""
        return sum(f.raw_tokens for f in self.frames)

    @property
    def total_compacted_tokens(self) -> int:
        """Sum of compacted tokens across all recorded frames."""
        return sum(f.compacted_tokens for f in self.frames)

    @property
    def total_effective_tokens(self) -> int:
        """Sum of effective tokens across all frames."""
        return sum(f.effective_tokens for f in self.frames)

    def recalculate_tokens_saved(self) -> int:
        """Recalculate total tokens saved across all frames and update the ledger counter."""
        self.tokens_saved = sum(f.tokens_saved for f in self.frames)
        return self.tokens_saved

    def record_hop_skipped(self, count: int = 1) -> None:
        """Increment the count of skipped hops/nodes."""
        self.hops_skipped += count

    def record_drift(self, drift: float) -> None:
        """Record a semantic drift measurement."""
        self.drift_history.append(float(drift))

    def record_entropy(self, entropy: float) -> None:
        """Record an entropy measurement."""
        self.entropy_history.append(float(entropy))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ledger to a dictionary."""
        return {
            "session_id": self.session_id,
            "goal_embedding": self.goal_embedding,
            "frames": [f.to_dict() for f in self.frames],
            "state_kv": self.state_kv,
            "entropy_history": list(self.entropy_history),
            "drift_history": list(self.drift_history),
            "tokens_saved": self.tokens_saved,
            "hops_skipped": self.hops_skipped,
            "total_raw_tokens": self.total_raw_tokens,
            "total_effective_tokens": self.total_effective_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceLedger:
        """Deserialize a ledger from a dictionary."""
        frames_data = data.get("frames", [])
        frames = [SignalFrame.from_dict(f) for f in frames_data]
        return cls(
            session_id=str(data["session_id"]),
            goal_embedding=data.get("goal_embedding"),
            frames=frames,
            state_kv=dict(data.get("state_kv", {})),
            entropy_history=[float(e) for e in data.get("entropy_history", [])],
            drift_history=[float(d) for d in data.get("drift_history", [])],
            tokens_saved=int(data.get("tokens_saved", 0)),
            hops_skipped=int(data.get("hops_skipped", 0)),
        )
