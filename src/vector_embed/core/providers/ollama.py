"""Ollama provider: embeddings and chat with explicit load/unload control.

Handles the Ollama pitfalls from the design: ``num_ctx`` is always passed (Ollama truncates
silently), embeddings are batched, per-model prefixes are applied, and ``keep_alive`` plus the
GPU/CPU switch let the app keep VRAM at zero outside active work.
"""

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeAlias

import httpx
import numpy as np
import ollama

from vector_embed.core.providers.base import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    ChatChunk,
    ChatOptions,
    EmbedKind,
    JsonResult,
    Message,
    ModelInfo,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    PullProgress,
    Usage,
)
from vector_embed.core.settings import EmbeddingSettings

logger = logging.getLogger(__name__)

ClientLike: TypeAlias = Any  # ollama.Client or a test double
Response: TypeAlias = Any

PROVIDER_NAME = "ollama"
_MAX_BATCH_CHARS = 60_000  # keeps one request's total prompt size sane
_RETRIES = 4
_TRANSIENT = (ConnectionError, TimeoutError, httpx.TransportError)


class Interrupted(Exception):  # noqa: N818  # control-flow signal, not an error
    """Raised between embedding batches when the caller's ``stop_check`` says to stop."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ollama.ResponseError):
        return exc.status_code >= 500
    return isinstance(exc, _TRANSIENT)


class OllamaProvider:
    """Chat and embedding provider backed by a local Ollama server."""

    name = PROVIDER_NAME

    def __init__(
        self,
        embedding: EmbeddingSettings | None = None,
        client: ClientLike | None = None,
        host: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._embedding = embedding or EmbeddingSettings()
        self._client: ClientLike = client or ollama.Client(host=host)
        self._sleep = sleep
        self._dim: int | None = None
        self.tokens_seen = 0  # prompt tokens reported by Ollama while embedding

    @property
    def client(self) -> ClientLike:
        """The underlying Ollama client (shared with the vision captioner)."""
        return self._client

    # ------------------------------------------------------------------ plumbing
    def _call(self, fn: Callable[[], Response]) -> Response:
        """Run a client call, retrying transient failures and translating the rest."""
        last: BaseException | None = None
        for attempt in range(_RETRIES):
            try:
                return fn()
            except ollama.ResponseError as exc:
                if exc.status_code == 404:
                    raise ModelNotFoundError(str(exc)) from exc
                if not _is_transient(exc):
                    raise ProviderError(str(exc)) from exc
                last = exc
            except _TRANSIENT as exc:
                last = exc
            if attempt < _RETRIES - 1:
                self._sleep(1.5 * (attempt + 1))
        raise ProviderUnavailableError(f"ollama unavailable: {last}") from last

    # ------------------------------------------------------------------ embeddings
    @property
    def embed_model(self) -> str:
        return self._embedding.model

    @property
    def dim(self) -> int:
        """Output dimension (probes the model once)."""
        if self._dim is None:
            self._dim = int(self._embed_batch(["dimension probe"], cpu=False).shape[1])
        return self._dim

    def _embed_batch(
        self, texts: list[str], cpu: bool, keep_alive: str | int | None = None
    ) -> np.ndarray:
        options: dict[str, Any] = {"num_ctx": self._embedding.num_ctx}
        if cpu:
            options["num_gpu"] = 0
        alive = self._embedding.keep_alive if keep_alive is None else keep_alive
        response = self._call(
            lambda: self._client.embed(
                model=self._embedding.model, input=texts, options=options, keep_alive=alive
            )
        )
        self.tokens_seen += int(getattr(response, "prompt_eval_count", 0) or 0)
        vectors: np.ndarray = np.asarray(response["embeddings"], dtype=np.float32)
        if vectors.shape[0] != len(texts):
            raise ProviderError(f"expected {len(texts)} vectors, got {vectors.shape[0]}")
        return self._finish(vectors)

    def _finish(self, vectors: np.ndarray) -> np.ndarray:
        """Matryoshka-truncate to the configured dim, then L2-normalise."""
        if vectors.shape[1] > self._embedding.dim:
            vectors = vectors[:, : self._embedding.dim]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised: np.ndarray = (vectors / norms).astype(np.float32)
        return normalised

    def embed(
        self,
        texts: list[str],
        kind: EmbedKind = "doc",
        cpu: bool = False,
        stop_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Embed ``texts`` with the model's query/document prefix, in size-bounded batches."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixes = self._embedding.prefixes_for()
        prefix = prefixes.query if kind == "query" else prefixes.document
        prepared = [prefix + t for t in texts]
        out: list[np.ndarray] = []
        batch: list[str] = []
        size = 0
        for text in prepared:
            if batch and (
                len(batch) >= self._embedding.batch_size or size + len(text) > _MAX_BATCH_CHARS
            ):
                out.append(self._flush(batch, cpu, stop_check))
                batch, size = [], 0
            batch.append(text)
            size += len(text)
        out.append(self._flush(batch, cpu, stop_check))
        stacked: np.ndarray = np.vstack(out)
        return stacked

    def _flush(
        self, batch: list[str], cpu: bool, stop_check: Callable[[], bool] | None
    ) -> np.ndarray:
        if stop_check is not None and stop_check():
            raise Interrupted
        return self._embed_batch(batch, cpu)

    def embed_query(self, query: str, cpu: bool = False) -> np.ndarray:
        vector: np.ndarray = self.embed([query], kind="query", cpu=cpu)[0]
        return vector

    def warm_embedder(self, cpu: bool = False) -> None:
        """Load the embedding model so the first real query is fast; failures are non-fatal."""
        try:
            self._embed_batch(["warmup"], cpu)
        except ProviderError:
            logger.debug("ollama: embedder warm-up failed", exc_info=True)

    def unload_embedder(self) -> None:
        """Release the embedder's VRAM now (``keep_alive=0``)."""
        try:
            self._embed_batch(["x"], cpu=False, keep_alive=0)
        except ProviderError:
            logger.debug("ollama: embedder unload failed", exc_info=True)

    # ------------------------------------------------------------------ chat
    @staticmethod
    def _chat_options(options: ChatOptions) -> dict[str, Any]:
        out: dict[str, Any] = {"num_ctx": options.num_ctx, "temperature": options.temperature}
        if options.cpu:
            out["num_gpu"] = 0
        return out

    @staticmethod
    def _wire(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _usage(response: Response) -> Usage:
        return Usage(
            int(getattr(response, "prompt_eval_count", 0) or 0),
            int(getattr(response, "eval_count", 0) or 0),
        )

    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]:
        opts = options or ChatOptions()
        stream = self._call(
            lambda: self._client.chat(
                model=model,
                messages=self._wire(messages),
                stream=True,
                options=self._chat_options(opts),
                keep_alive=opts.keep_alive,
            )
        )
        try:
            for part in stream:
                text = part["message"]["content"] or ""
                if part.get("done"):
                    yield ChatChunk(text, self._usage(part))
                elif text:
                    yield ChatChunk(text)
        except _TRANSIENT as exc:
            raise ProviderUnavailableError(f"ollama stream interrupted: {exc}") from exc

    def chat_json(
        self,
        messages: list[Message],
        model: str,
        schema: dict[str, Any],
        options: ChatOptions | None = None,
    ) -> JsonResult:
        """One non-streaming call constrained to ``schema``; the reply must parse as JSON."""
        opts = options or ChatOptions()
        response = self._call(
            lambda: self._client.chat(
                model=model,
                messages=self._wire(messages),
                format=schema,
                options=self._chat_options(opts),
                keep_alive=opts.keep_alive,
            )
        )
        content = response["message"]["content"]
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"model returned invalid JSON: {exc}") from exc
        return JsonResult(data, self._usage(response))

    def prewarm(self, model: str, options: ChatOptions | None = None) -> None:
        """Start loading ``model`` (empty request) so the first answer has no load delay."""
        opts = options or ChatOptions()
        try:
            self._call(
                lambda: self._client.generate(
                    model=model,
                    prompt="",
                    keep_alive=opts.keep_alive,
                    options={"num_ctx": opts.num_ctx, **({"num_gpu": 0} if opts.cpu else {})},
                )
            )
        except ProviderError:
            logger.debug("ollama: prewarm of %s failed", model, exc_info=True)

    def unload(self, model: str) -> None:
        """Release ``model``'s memory immediately."""
        try:
            self._call(lambda: self._client.generate(model=model, prompt="", keep_alive=0))
        except ProviderError:
            logger.debug("ollama: unload of %s failed", model, exc_info=True)

    def loaded_models(self) -> list[str]:
        """Names of models currently resident (``ollama ps``)."""
        response = self._call(self._client.ps)
        return [m.model for m in response.models]

    def unload_all(self) -> None:
        for name in self.loaded_models():
            self.unload(name)

    # ------------------------------------------------------------------ discovery
    def list_models(self) -> list[ModelInfo]:
        listing = self._call(self._client.list)
        return [self._describe(entry) for entry in listing.models]

    def _describe(self, entry: Response) -> ModelInfo:
        name = str(entry.model)
        details = getattr(entry, "details", None)
        info = ModelInfo(
            name=name,
            provider=PROVIDER_NAME,
            size_bytes=getattr(entry, "size", None),
            parameter_size=getattr(details, "parameter_size", None),
            quantization=getattr(details, "quantization_level", None),
        )
        try:
            shown = self._call(lambda: self._client.show(name))
        except ProviderError:
            return info
        caps = frozenset(getattr(shown, "capabilities", None) or ())
        return ModelInfo(
            name=info.name,
            provider=info.provider,
            size_bytes=info.size_bytes,
            parameter_size=info.parameter_size,
            quantization=info.quantization,
            context_length=_context_length(getattr(shown, "modelinfo", None)),
            capabilities=caps,
        )

    def capabilities(self, model: str) -> frozenset[str]:
        shown = self._call(lambda: self._client.show(model))
        return frozenset(getattr(shown, "capabilities", None) or (CAP_COMPLETION,))

    def estimate_cost(self, model: str, usage: Usage) -> float:
        return 0.0

    def pull(self, model: str) -> Iterator[PullProgress]:
        """Download ``model``, yielding progress for a progress bar."""
        stream = self._call(lambda: self._client.pull(model, stream=True))
        for part in stream:
            yield PullProgress(
                status=str(part.status or ""),
                completed=int(part.completed or 0),
                total=int(part.total or 0),
            )

    def has_model(self, model: str) -> bool:
        wanted = model if ":" in model else model + ":latest"
        return any(m.name in {model, wanted} for m in self.list_models())

    def embedding_models(self) -> list[str]:
        return [m.name for m in self.list_models() if CAP_EMBEDDING in m.capabilities]


def _context_length(modelinfo: Response) -> int | None:
    """``<arch>.context_length`` from ``ollama show`` model info."""
    if not modelinfo:
        return None
    for key, value in dict(modelinfo).items():
        if str(key).endswith(".context_length"):
            return int(value)
    return None
