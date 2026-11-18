"""Embedding protocols, deterministic hashing embedder, and FastEmbed adapter."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol for embedding providers in agent-snr."""

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: Input text string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...


class HashNgramEmbedder:
    """Zero-dependency deterministic feature-hashing embedder.

    Produces 384-dimensional L2-normalized vectors using word unigrams, bigrams,
    and character n-grams with Murmur/MD5 signed feature hashing. Enables agent-snr
    to work out-of-the-box with zero model downloads or GPU requirements.
    """

    def __init__(
        self,
        dim: int = 384,
        char_ngram_range: tuple[int, int] = (3, 4),
        include_bigrams: bool = True,
    ) -> None:
        """Initialize the HashNgramEmbedder.

        Args:
            dim: Dimension of the output embedding vector (default: 384).
            char_ngram_range: Tuple of (min_n, max_n) for character n-grams.
            include_bigrams: Whether to include adjacent word bigrams as features.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.char_ngram_range = char_ngram_range
        self.include_bigrams = include_bigrams
        self._word_pattern = re.compile(r"\b[a-zA-Z0-9_\-']+\b")

    def _extract_features(self, text: str) -> list[str]:
        """Extract word unigrams, bigrams, and character n-grams from text."""
        cleaned = text.lower().strip()
        if not cleaned:
            return []

        words = self._word_pattern.findall(cleaned)
        if not words:
            return []

        features: list[str] = []

        # 1. Word unigrams
        for w in words:
            features.append(f"w:{w}")

        # 2. Word bigrams
        if self.include_bigrams and len(words) > 1:
            for i in range(len(words) - 1):
                features.append(f"bi:{words[i]}_{words[i+1]}")

        # 3. Subword character n-grams with boundary markers
        min_n, max_n = self.char_ngram_range
        for w in words:
            padded = f"^{w}$"
            padded_len = len(padded)
            for n in range(min_n, max_n + 1):
                if padded_len >= n:
                    for i in range(padded_len - n + 1):
                        features.append(f"c:{padded[i : i + n]}")

        return features

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic 384-d L2-normalized embedding for text.

        Args:
            text: The text to embed.

        Returns:
            A list of 384 floats representing the L2-normalized embedding.
        """
        features = self._extract_features(text)
        vec = np.zeros(self.dim, dtype=np.float32)

        if not features:
            return vec.tolist()

        for feat in features:
            # Deterministic MD5 hash of UTF-8 encoded feature bytes
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            # First 4 bytes determine bucket index
            idx = int.from_bytes(digest[:4], byteorder="little") % self.dim
            # 5th byte determines sign (+1 or -1) to remove collision bias
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm

        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings in batch.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(t) for t in texts]


class FastEmbedAdapter:
    """Adapter for fastembed.TextEmbedding, lazy-loading models on first embed call.

    Default model: 'BAAI/bge-small-en-v1.5' (384 dimensions).
    Requires `agent-snr[embeddings]` or `fastembed>=0.3.0`.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FastEmbedAdapter.

        Args:
            model_name: FastEmbed model identifier. Defaults to 'BAAI/bge-small-en-v1.5'.
            model: Optional pre-instantiated model (useful for testing/mocking).
            **kwargs: Extra arguments passed to TextEmbedding initialization.
        """
        self.model_name = model_name
        self._model = model
        self.kwargs = kwargs

    def _get_model(self) -> Any:
        """Lazy-load the underlying fastembed model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as err:
                raise ImportError(
                    "fastembed is required to use FastEmbedAdapter. "
                    "Install it via `pip install agent-snr[embeddings]` or `pip install fastembed`."
                ) from err
            self._model = TextEmbedding(model_name=self.model_name, **self.kwargs)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string using fastembed.

        Args:
            text: Input text string.

        Returns:
            List of floats representing the embedding.
        """
        model = self._get_model()
        results = list(model.embed([text]))
        vec = results[0]
        if hasattr(vec, "tolist"):
            return vec.tolist()  # type: ignore[no-any-return]
        return list(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings using fastembed.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        model = self._get_model()
        results = list(model.embed(texts))
        return [r.tolist() if hasattr(r, "tolist") else list(r) for r in results]


def cosine_similarity(
    a: Sequence[float] | np.ndarray[Any, Any],
    b: Sequence[float] | np.ndarray[Any, Any],
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in the range [-1.0, 1.0].
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)

    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = float(np.dot(va, vb) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))
