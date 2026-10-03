"""Provider contracts. Skills depend on these protocols, never on ollama or openai directly."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

EmbedKind = Literal["query", "doc"]

CAP_EMBEDDING = "embedding"
CAP_COMPLETION = "completion"
CAP_VISION = "vision"
CAP_TOOLS = "tools"
CAP_THINKING = "thinking"


class ProviderError(RuntimeError):
    """A provider call failed."""


class ProviderUnavailableError(ProviderError):
    """The provider cannot be reached (server down, no key, offline)."""


class ModelNotFoundError(ProviderError):
    """The requested model is not installed or not offered by the provider."""


class InvalidJsonError(ProviderError):
    """The model answered, but not with parseable JSON (worth one more try)."""


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ChatOptions:
    num_ctx: int = 8192
    temperature: float = 0.2
    keep_alive: str | int = "10m"
    cpu: bool = False  # run on CPU (num_gpu=0) so the GPU stays free
    max_tokens: int | None = None  # cap on the reply length (a cloud reply is billed per token)


@dataclass(frozen=True)
class ChatChunk:
    """One streamed piece. ``usage`` is set on the final chunk only."""

    text: str = ""
    usage: Usage | None = None


@dataclass(frozen=True)
class JsonResult:
    data: Any
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    provider: str
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    capabilities: frozenset[str] = frozenset()

    @property
    def is_embedding(self) -> bool:
        return CAP_EMBEDDING in self.capabilities


@dataclass(frozen=True)
class PullProgress:
    status: str
    completed: int = 0
    total: int = 0

    @property
    def fraction(self) -> float:
        return self.completed / self.total if self.total else 0.0


@runtime_checkable
class ChatProvider(Protocol):
    name: str

    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]: ...

    def chat_json(
        self,
        messages: list[Message],
        model: str,
        schema: dict[str, Any],
        options: ChatOptions | None = None,
    ) -> JsonResult: ...

    def list_models(self) -> list[ModelInfo]: ...

    def capabilities(self, model: str) -> frozenset[str]: ...

    def estimate_cost(self, model: str, usage: Usage) -> float:
        """Estimated USD cost; local providers return 0."""
        ...


@runtime_checkable
class EmbedProvider(Protocol):
    name: str

    def embed(self, texts: list[str], kind: EmbedKind = "doc", cpu: bool = False) -> np.ndarray:
        """L2-normalised vectors, one row per text."""
        ...
