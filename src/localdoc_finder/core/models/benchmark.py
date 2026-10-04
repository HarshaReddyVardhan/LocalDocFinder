"""Speed test for a freshly downloaded model: is it fast enough to be pleasant on this machine?

One model is measured at a time and always unloaded afterwards, so the embedder and the chat model
are never resident together and VRAM returns to zero. Timing goes through the provider protocols
(streamed chunks and wall-clock time), so no Ollama-specific fields are needed.
"""

import json
import logging
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from localdoc_finder.core.providers.base import ChatChunk, ChatOptions, EmbedKind, Message
from localdoc_finder.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

MIN_CHAT_TOKENS_PER_SECOND = 8.0  # below this an answer feels sluggish
MIN_EMBEDS_PER_SECOND = 5.0  # below this indexing a large folder takes hours
CHAT_SAMPLE_TOKENS = 64
CHAT_PROMPT = "Explain in about fifty words why the sky looks blue during the day."
CHAT_NUM_CTX = 2048
EMBED_SAMPLE_COUNT = 16
EMBED_SAMPLE_TEXT = (
    "Quarterly planning notes: the migration to the new storage layer finished ahead of "
    "schedule, but the reporting service still reads from the legacy tables and needs a "
    "follow-up before the next release window. "
)
_RESULTS_KEY = "speed_test_results"


class BenchmarkError(RuntimeError):
    """The model could not be measured (no output, or the provider failed)."""


class BenchKind(StrEnum):
    CHAT = "chat"
    EMBED = "embed"


class Verdict(StrEnum):
    OK = "ok"
    SLOW = "slow"


@dataclass(frozen=True)
class BenchResult:
    model: str
    kind: BenchKind
    rate: float  # tokens/s for chat, texts/s for embed
    startup_s: float  # seconds to the first token (chat) or to load the model (embed)

    @property
    def unit(self) -> str:
        return "tok/s" if self.kind is BenchKind.CHAT else "texts/s"


class ChatBenchProvider(Protocol):
    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]: ...

    def unload(self, model: str) -> None: ...


class EmbedBenchProvider(Protocol):
    @property
    def embed_model(self) -> str: ...

    def embed(self, texts: list[str], kind: EmbedKind = "doc", cpu: bool = False) -> np.ndarray: ...

    def unload_embedder(self) -> None: ...


def bench_chat(
    provider: ChatBenchProvider,
    model: str,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchResult:
    """Cold-start the model, stream ~64 tokens and time them."""
    provider.unload(model)  # start cold so the first-token time includes loading
    options = ChatOptions(num_ctx=CHAT_NUM_CTX, temperature=0.0, keep_alive=0)
    started = clock()
    first_at: float | None = None
    last_at = started
    tokens = 0
    stream = provider.stream_chat([Message("user", CHAT_PROMPT)], model, options)
    try:
        for chunk in stream:
            if not chunk.text:
                continue
            last_at = clock()
            if first_at is None:
                first_at = last_at
            tokens += 1
            if tokens >= CHAT_SAMPLE_TOKENS:
                break
    finally:
        if isinstance(stream, Generator):
            stream.close()  # stop the server generating once we have enough
        provider.unload(model)
    if first_at is None or tokens < 2 or last_at <= first_at:
        raise BenchmarkError(f"{model} produced too little output to measure")
    # the first token ends the wait, so the rate counts the tokens that followed it
    return BenchResult(
        model, BenchKind.CHAT, (tokens - 1) / (last_at - first_at), first_at - started
    )


def bench_embed(
    provider: EmbedBenchProvider, clock: Callable[[], float] = time.perf_counter
) -> BenchResult:
    """Time one cold single-text call (load), then a warm fixed batch (throughput)."""
    model = provider.embed_model
    texts = [f"{EMBED_SAMPLE_TEXT}Item {i}." for i in range(EMBED_SAMPLE_COUNT)]
    try:
        started = clock()
        provider.embed(texts[:1])
        loaded = clock()
        provider.embed(texts)
        finished = clock()
    finally:
        provider.unload_embedder()
    elapsed = finished - loaded
    if elapsed <= 0:
        raise BenchmarkError(f"{model} embedded too fast to measure")
    return BenchResult(model, BenchKind.EMBED, len(texts) / elapsed, loaded - started)


def judge(result: BenchResult) -> Verdict:
    floor = MIN_CHAT_TOKENS_PER_SECOND if result.kind is BenchKind.CHAT else MIN_EMBEDS_PER_SECOND
    return Verdict.SLOW if result.rate < floor else Verdict.OK


def record_result(state: StateDb, result: BenchResult) -> None:
    """Keep the latest result per model for the Settings window."""
    results = {(r.model, r.kind): r for r in load_results(state)}
    results[(result.model, result.kind)] = result
    payload = [{**asdict(r), "kind": str(r.kind)} for r in results.values()]
    state.set_meta(_RESULTS_KEY, json.dumps(payload))


def load_results(state: StateDb) -> list[BenchResult]:
    raw = state.get_meta(_RESULTS_KEY)
    if not raw:
        return []
    try:
        return [
            BenchResult(
                str(d["model"]), BenchKind(d["kind"]), float(d["rate"]), float(d["startup_s"])
            )
            for d in json.loads(raw)
        ]
    except (ValueError, KeyError, TypeError):
        logger.warning("benchmark: ignoring unreadable stored results")
        return []
