from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from localdoc_finder.core.models import benchmark as bench
from localdoc_finder.core.models.benchmark import (
    BenchKind,
    BenchmarkError,
    BenchResult,
    Verdict,
    bench_chat,
    bench_embed,
    judge,
    load_results,
    record_result,
)
from localdoc_finder.core.providers.base import (
    ChatChunk,
    ChatOptions,
    EmbedKind,
    Message,
    ProviderError,
)
from localdoc_finder.core.store.sqlite import StateDb


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeChat:
    """Streams ``tokens`` chunks: ``load_s`` before the first, ``per_token_s`` between them."""

    def __init__(
        self, clock: FakeClock, tokens: int, load_s: float = 2.0, per_token_s: float = 0.05
    ) -> None:
        self.clock = clock
        self.tokens = tokens
        self.load_s = load_s
        self.per_token_s = per_token_s
        self.unloaded: list[str] = []
        self.yielded = 0
        self.fail = False
        self.options: ChatOptions | None = None

    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]:
        self.options = options
        if self.fail:
            raise ProviderError("boom")
        return self._stream()

    def _stream(self) -> Iterator[ChatChunk]:
        for i in range(self.tokens):
            self.clock.now += self.load_s if i == 0 else self.per_token_s
            self.yielded += 1
            yield ChatChunk("tok ")

    def unload(self, model: str) -> None:
        self.unloaded.append(model)


class FakeEmbedder:
    def __init__(self, clock: FakeClock, load_s: float = 1.0, batch_s: float = 0.5) -> None:
        self.clock = clock
        self.load_s = load_s
        self.batch_s = batch_s
        self.calls = 0
        self.unloaded = 0
        self.fail = False

    @property
    def embed_model(self) -> str:
        return "fake-embed"

    def embed(self, texts: list[str], kind: EmbedKind = "doc", cpu: bool = False) -> np.ndarray:
        if self.fail:
            raise ProviderError("boom")
        self.calls += 1
        self.clock.now += self.load_s if self.calls == 1 else self.batch_s
        return np.zeros((len(texts), 4), dtype=np.float32)

    def unload_embedder(self) -> None:
        self.unloaded += 1


def test_chat_measures_rate_and_first_token_time() -> None:
    clock = FakeClock()
    provider = FakeChat(clock, tokens=200)
    result = bench_chat(provider, "m", clock=clock)
    assert result.kind is BenchKind.CHAT
    assert result.startup_s == pytest.approx(2.0)
    assert result.rate == pytest.approx(20.0)  # 1 / 0.05 s per token
    assert provider.yielded == bench.CHAT_SAMPLE_TOKENS  # stopped early, not 200
    assert provider.unloaded == ["m", "m"]  # cold start, then cleanup
    assert provider.options is not None
    assert provider.options.keep_alive == 0


def test_chat_unloads_when_the_stream_fails() -> None:
    clock = FakeClock()
    provider = FakeChat(clock, tokens=10)
    provider.fail = True
    with pytest.raises(ProviderError):
        bench_chat(provider, "m", clock=clock)
    assert provider.unloaded == ["m"]  # failed before the stream existed; only the cold unload


def test_chat_unloads_after_midstream_error() -> None:
    clock = FakeClock()

    class Broken(FakeChat):
        def _stream(self) -> Iterator[ChatChunk]:
            yield ChatChunk("a")
            raise ProviderError("dropped")

    provider = Broken(clock, tokens=5)
    with pytest.raises(ProviderError):
        bench_chat(provider, "m", clock=clock)
    assert provider.unloaded == ["m", "m"]


def test_chat_with_no_output_raises() -> None:
    clock = FakeClock()
    with pytest.raises(BenchmarkError, match="too little output"):
        bench_chat(FakeChat(clock, tokens=1), "m", clock=clock)


def test_embed_measures_load_and_throughput() -> None:
    clock = FakeClock()
    provider = FakeEmbedder(clock)
    result = bench_embed(provider, clock=clock)
    assert result.model == "fake-embed"
    assert result.startup_s == pytest.approx(1.0)
    assert result.rate == pytest.approx(bench.EMBED_SAMPLE_COUNT / 0.5)
    assert provider.unloaded == 1


def test_embed_unloads_on_error() -> None:
    clock = FakeClock()
    provider = FakeEmbedder(clock)
    provider.fail = True
    with pytest.raises(ProviderError):
        bench_embed(provider, clock=clock)
    assert provider.unloaded == 1


def test_embed_with_zero_elapsed_time_raises() -> None:
    clock = FakeClock()
    with pytest.raises(BenchmarkError, match="too fast"):
        bench_embed(FakeEmbedder(clock, batch_s=0.0), clock=clock)


@pytest.mark.parametrize(
    ("kind", "rate", "verdict"),
    [
        (BenchKind.CHAT, bench.MIN_CHAT_TOKENS_PER_SECOND - 0.1, Verdict.SLOW),
        (BenchKind.CHAT, bench.MIN_CHAT_TOKENS_PER_SECOND, Verdict.OK),
        (BenchKind.EMBED, bench.MIN_EMBEDS_PER_SECOND - 0.1, Verdict.SLOW),
        (BenchKind.EMBED, bench.MIN_EMBEDS_PER_SECOND, Verdict.OK),
    ],
)
def test_judge_thresholds(kind: BenchKind, rate: float, verdict: Verdict) -> None:
    assert judge(BenchResult("m", kind, rate, 1.0)) is verdict


def test_unit_labels() -> None:
    assert BenchResult("m", BenchKind.CHAT, 1, 1).unit == "tok/s"
    assert BenchResult("m", BenchKind.EMBED, 1, 1).unit == "texts/s"


def test_results_round_trip_and_keep_latest_per_model(tmp_path: Path) -> None:
    state = StateDb(tmp_path / "state.db")
    assert load_results(state) == []
    first = BenchResult("a", BenchKind.CHAT, 10.0, 1.0)
    record_result(state, first)
    record_result(state, BenchResult("a", BenchKind.EMBED, 30.0, 0.5))
    newer = BenchResult("a", BenchKind.CHAT, 12.0, 0.8)
    record_result(state, newer)
    stored = load_results(state)
    assert sorted(stored, key=lambda r: r.kind) == [
        newer,
        BenchResult("a", BenchKind.EMBED, 30.0, 0.5),
    ]


def test_unreadable_stored_results_are_ignored(tmp_path: Path) -> None:
    state = StateDb(tmp_path / "state.db")
    state.set_meta("speed_test_results", "not json")
    assert load_results(state) == []
    state.set_meta("speed_test_results", '[{"model": "a"}]')
    assert load_results(state) == []
