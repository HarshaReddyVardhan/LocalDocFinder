"""Shared test doubles for the indexing and search pipelines."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import xxhash

from vector_embed.core.providers.base import EmbedKind
from vector_embed.core.providers.ollama import Interrupted

DIM = 16


def text_vector(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic pseudo-embedding with word overlap giving similarity (bag of hashed words)."""
    vector = np.zeros(dim, dtype=np.float32)
    for word in text.lower().split():
        slot = xxhash.xxh32_intdigest(word.encode()) % dim
        vector[slot] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return vector / norm


@dataclass
class FakeEmbedder:
    """Embedder with no model: records every batch so tests can count embedded chunks."""

    dim: int = DIM
    calls: list[tuple[list[str], str]] = field(default_factory=list)

    def embed(
        self,
        texts: list[str],
        kind: EmbedKind = "doc",
        cpu: bool = False,
        stop_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        if stop_check is not None and stop_check():
            raise Interrupted
        self.calls.append((list(texts), kind))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([text_vector(t, self.dim) for t in texts])

    @property
    def embedded_texts(self) -> list[str]:
        return [t for batch, _ in self.calls for t in batch]
