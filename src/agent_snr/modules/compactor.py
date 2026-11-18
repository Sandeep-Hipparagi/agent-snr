"""State compaction module: SNR scoring, JSON pruning, traceback folding, and deduplication."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from agent_snr.core.embedder import cosine_similarity
from agent_snr.core.frames import FrameRole, SignalFrame, TraceLedger

NOISE_KEYS = frozenset(
    {
        "request_id",
        "trace_id",
        "etag",
        "headers",
        "metadata",
        "created_at",
        "updated_at",
        "_links",
        "raw_html",
    }
)


class StateCompactor:
    """Multi-tier context compactor for agent execution frames."""

    def __init__(
        self,
        max_frame_tokens: int = 512,
        snr_drop_threshold: float = 0.08,
        max_json_depth: int = 5,
        max_list_items: int = 15,
        slm_summarizer: Callable[[str, int], str] | None = None,
    ) -> None:
        """Initialize StateCompactor.

        Args:
            max_frame_tokens: Maximum allowed token budget before SLM distillation.
            snr_drop_threshold: SNR score below which redundant frames are suppressed.
            max_json_depth: Maximum recursion depth when pruning JSON payloads.
            max_list_items: Maximum items retained per list in JSON payloads.
            slm_summarizer: Optional callable (content, budget) -> summarized_content.
        """
        self.max_frame_tokens = max_frame_tokens
        self.snr_drop_threshold = snr_drop_threshold
        self.max_json_depth = max_json_depth
        self.max_list_items = max_list_items
        self.slm_summarizer = slm_summarizer

    def compute_snr(self, frame: SignalFrame, ledger: TraceLedger) -> float:
        """Compute Signal-to-Noise Ratio (SNR) for a frame relative to ledger history.

        Formula: round(lexical_novelty * semantic_novelty, 4)
        """
        tokens = set(re.findall(r"\w+", frame.content.lower()))
        if not tokens:
            return 0.0

        # Prior tokens over trailing 10 frames (excluding the current frame if already added)
        prior_frames = [f for f in ledger.frames if f.step_id != frame.step_id][-10:]
        prior_tokens: set[str] = set()
        for f in prior_frames:
            prior_tokens.update(re.findall(r"\w+", f.content.lower()))

        novel_tokens = tokens - prior_tokens
        lexical_novelty = len(novel_tokens) / len(tokens)

        # Semantic novelty: 1.0 - max cosine similarity to prior frames
        max_sim = 0.0
        if frame.embedding is not None:
            sims = [
                cosine_similarity(frame.embedding, prev.embedding)
                for prev in prior_frames
                if prev.embedding is not None
            ]
            if sims:
                max_sim = max(sims)

        semantic_novelty = max(0.05, 1.0 - max_sim)
        return round(float(lexical_novelty * semantic_novelty), 4)

    def _prune_json_recursive(self, data: Any, depth: int) -> Any:
        """Recursively prune noise keys, empty values, and truncate lists/strings."""
        if depth >= self.max_json_depth:
            return "[Max depth reached]"

        if isinstance(data, dict):
            cleaned: dict[str, Any] = {}
            for k, v in data.items():
                if str(k).lower() in NOISE_KEYS:
                    continue
                pruned_v = self._prune_json_recursive(v, depth + 1)
                # Strip None, "", [], {}
                if pruned_v is None or pruned_v == "" or pruned_v == [] or pruned_v == {}:
                    continue
                cleaned[str(k)] = pruned_v
            return cleaned

        if isinstance(data, list):
            truncated = data[: self.max_list_items]
            cleaned_list: list[Any] = []
            for item in truncated:
                pruned_item = self._prune_json_recursive(item, depth + 1)
                if (
                    pruned_item is not None
                    and pruned_item != ""
                    and pruned_item != []
                    and pruned_item != {}
                ):
                    cleaned_list.append(pruned_item)
            return cleaned_list

        if isinstance(data, str):
            if len(data) > 1000:
                return data[:1000] + "... [truncated]"
            return data

        return data

    def _prune_json(self, content: str) -> str:
        """Attempt deterministic JSON pruning if content looks like JSON."""
        stripped = content.strip()
        if not ((stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        )):
            return content

        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return content

        pruned = self._prune_json_recursive(parsed, depth=0)
        return json.dumps(pruned, separators=(",", ":"), ensure_ascii=False)

    def _fold_tracebacks_and_whitespace(self, content: str) -> str:
        """Fold internal site-packages stack frames and collapse excessive newlines."""
        lines = content.splitlines()
        folded_lines: list[str] = []
        i = 0
        n = len(lines)
        internal_pkg_pattern = re.compile(
            r'^[ \t]*File "[^"]*(?:site-packages|dist-packages|[\\/]usr[\\/]lib)'
            r'[^"]*", line \d+'
        )
        file_line_pattern = re.compile(r'^[ \t]*File "[^"]*", line \d+')

        while i < n:
            line = lines[i]
            if internal_pkg_pattern.match(line):
                # Found the start of an internal package frame
                has_internal = False
                while i < n and internal_pkg_pattern.match(lines[i]):
                    has_internal = True
                    i += 1
                    # Skip code lines for this frame (indented lines that don't start with File ")
                    while (
                        i < n
                        and lines[i].startswith(" ")
                        and not file_line_pattern.match(lines[i])
                    ):
                        i += 1
                if has_internal:
                    folded_lines.append("  [... internal site-packages frames hidden ...]")
            else:
                folded_lines.append(line)
                i += 1

        folded = "\n".join(folded_lines)
        return re.sub(r"\n{3,}", "\n\n", folded).strip()

    def compact(self, frame: SignalFrame, ledger: TraceLedger) -> SignalFrame:
        """Apply multi-tier compaction to a frame."""
        # Unconditionally preserve pinned frames and system/user messages
        if frame.is_pinned or frame.role in (FrameRole.SYSTEM, FrameRole.USER):
            if frame.snr_score == 0.0:
                frame.snr_score = 1.0
            if frame.raw_tokens == 0:
                frame.raw_tokens = max(1, len(frame.content) // 4)
            return frame

        # Compute SNR score
        frame.snr_score = self.compute_snr(frame, ledger)

        if frame.raw_tokens == 0:
            frame.raw_tokens = max(1, len(frame.content) // 4)

        # Tier 2: Idempotent Read Deduplication check for TOOL_RESULT
        if frame.role == FrameRole.TOOL_RESULT and frame.tool_name:
            key = f"{frame.tool_name}:{json.dumps(frame.tool_args or {}, sort_keys=True)}"
            if ledger.state_kv.get(key) == frame.content:
                dedup_content = (
                    f"[Unchanged state for {frame.tool_name} — identical to prior read]"
                )
                comp_tokens = max(1, len(dedup_content) // 4)
                frame.compact(dedup_content, comp_tokens)
                return frame
            ledger.state_kv[key] = frame.content

        # Gate 0: Low-signal noise suppression
        if frame.snr_score < self.snr_drop_threshold and len(ledger.frames) > 2:
            suppressed = f"[SNR={frame.snr_score:.2f} | Low-signal duplicate output suppressed]"
            comp_tokens = max(1, len(suppressed) // 4)
            frame.compact(suppressed, comp_tokens)
            return frame

        content = frame.content

        # Tier 1A: Deterministic JSON pruning
        content = self._prune_json(content)

        # Tier 1B: Traceback & whitespace folding
        content = self._fold_tracebacks_and_whitespace(content)

        # Tier 3: Optional SLM Distillation
        est_tokens = max(1, len(content) // 4)
        if est_tokens > self.max_frame_tokens and self.slm_summarizer is not None:
            content = self.slm_summarizer(content, self.max_frame_tokens)

        compacted_tokens = max(1, len(content) // 4)
        frame.compact(content, compacted_tokens)
        return frame
