"""Shared test doubles for the indexing and search pipelines."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xxhash

from localdoc_finder.core.providers.base import (
    ChatChunk,
    ChatOptions,
    EmbedKind,
    JsonResult,
    Message,
    ModelInfo,
    Usage,
)
from localdoc_finder.core.providers.ollama import Interrupted

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


class FakeCloudInner:
    """A scripted cloud provider that records exactly what it was sent."""

    name = "openrouter"
    label = "OpenRouter"

    def __init__(self) -> None:
        self.sent: list[list[Message]] = []
        self.reply = ["Contact ", "[NAME_1] at [EMAIL", "_1]."]
        self.usage = Usage(1000, 500)
        self.json_data: Any = {"summary": "[NAME_1] is a fit"}
        self.json_fn: Callable[[list[Message]], Any] | None = None
        self.error: Exception | None = None  # raised by every call, before anything is produced

    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]:
        self.sent.append(messages)
        if self.error is not None:
            raise self.error
        for piece in self.reply:
            yield ChatChunk(piece)
        yield ChatChunk("", self.usage)

    def chat_json(
        self,
        messages: list[Message],
        model: str,
        schema: dict[str, Any],
        options: ChatOptions | None = None,
    ) -> JsonResult:
        self.sent.append(messages)
        if self.error is not None:
            raise self.error
        data = self.json_fn(messages) if self.json_fn else self.json_data
        return JsonResult(data, self.usage)

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo("m", "openrouter")]

    def capabilities(self, model: str) -> frozenset[str]:
        return frozenset({"completion"})

    def estimate_cost(self, model: str, usage: Usage) -> float:
        return (usage.prompt_tokens + usage.completion_tokens) / 1000.0  # $1 per 1k tokens
