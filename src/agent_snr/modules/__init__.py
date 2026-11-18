"""Modular detection, filtering, compaction, and routing algorithms for agent-snr."""

from agent_snr.modules.budgeter import DynamicAttentionBudgeter
from agent_snr.modules.compactor import StateCompactor
from agent_snr.modules.entropy import LoopEntropyDetector
from agent_snr.modules.router import SkipLevelRouter, SkipRule

__all__ = [
    "DynamicAttentionBudgeter",
    "LoopEntropyDetector",
    "SkipLevelRouter",
    "SkipRule",
    "StateCompactor",
]
