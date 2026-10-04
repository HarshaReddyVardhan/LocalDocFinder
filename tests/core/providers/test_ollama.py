import json
from typing import ClassVar

import numpy as np
import ollama
import pytest
from tests.core.providers.fakes import FakeOllamaClient

from localdoc_finder.core.providers import ollama as om
from localdoc_finder.core.providers.base import (
    CAP_EMBEDDING,
    ChatChunk,
    ChatOptions,
    ChatProvider,
    EmbedProvider,
    InvalidJsonError,
    Message,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    Usage,
)
from localdoc_finder.core.settings import EmbeddingProfile, EmbeddingSettings

QUERY_PREFIX = "Q: "
DOC_PREFIX = "D: "


def make(client: FakeOllamaClient, **overrides: object) -> om.OllamaProvider:
    fields: dict[str, object] = {
        "model": "m",
        "dim": 4,
        "batch_size": 2,
        "profiles": {"m": EmbeddingProfile(query=QUERY_PREFIX, document=DOC_PREFIX)},
        **overrides,
    }
    settings = EmbeddingSettings(**fields)  # type: ignore[arg-type]
    return om.OllamaProvider(settings, client=client, sleep=lambda _s: None)


def test_implements_both_protocols() -> None:
    provider = make(FakeOllamaClient())
    assert isinstance(provider, ChatProvider)
    assert isinstance(provider, EmbedProvider)


class TestEmbeddings:
    def test_vectors_are_normalised(self) -> None:
        vectors = make(FakeOllamaClient()).embed(["a"])
        assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
        assert vectors.dtype == np.float32

    def test_prefix_depends_on_kind(self) -> None:
        client = FakeOllamaClient()
        provider = make(client)
        provider.embed(["x"], kind="query")
        provider.embed(["y"], kind="doc")
        sent = [c[1]["input"] for c in client.calls if c[0] == "embed"]
        assert sent == [[QUERY_PREFIX + "x"], [DOC_PREFIX + "y"]]

    def test_num_ctx_is_always_passed_and_cpu_switches_gpu_off(self) -> None:
        client = FakeOllamaClient()
        provider = make(client)
        provider.embed(["x"])
        provider.embed(["x"], cpu=True)
        gpu, cpu = [c[1]["options"] for c in client.calls]
        assert gpu == {"num_ctx": 8192}
        assert cpu == {"num_ctx": 8192, "num_gpu": 0}

    def test_batches_by_count(self) -> None:
        client = FakeOllamaClient()
        make(client).embed(["a", "b", "c", "d", "e"])
        assert [len(c[1]["input"]) for c in client.calls] == [2, 2, 1]

    def test_batches_by_characters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(om, "_MAX_BATCH_CHARS", 10)
        client = FakeOllamaClient()
        make(client, batch_size=100).embed(["aaaaaa", "bbbbbb"])
        assert len(client.calls) == 2

    def test_empty_input(self) -> None:
        client = FakeOllamaClient()
        assert make(client).embed([]).shape == (0, 4)

    def test_truncates_wider_models_to_configured_dim(self) -> None:
        client = FakeOllamaClient(embed_fn=lambda t: [[1.0, 1.0, 1.0, 1.0, 9.0] for _ in t])
        assert make(client).embed(["a"]).shape == (1, 4)

    def test_zero_vector_does_not_divide_by_zero(self) -> None:
        client = FakeOllamaClient(embed_fn=lambda t: [[0.0, 0.0, 0.0, 0.0] for _ in t])
        assert not np.isnan(make(client).embed(["a"])).any()

    def test_dim_probes_once(self) -> None:
        client = FakeOllamaClient()
        provider = make(client)
        assert provider.dim == 4
        assert provider.dim == 4
        assert len(client.calls) == 1

    def test_wrong_vector_count_is_an_error(self) -> None:
        client = FakeOllamaClient(embed_fn=lambda _t: [[1.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ProviderError, match="expected 2 vectors"):
            make(client).embed(["a", "b"])

    def test_stop_check_interrupts_between_batches(self) -> None:
        client = FakeOllamaClient()
        stops = iter([False, True])
        with pytest.raises(om.Interrupted):
            make(client).embed(["a", "b", "c"], stop_check=lambda: next(stops))
        assert len(client.calls) == 1

    def test_embed_query_returns_one_vector(self) -> None:
        assert make(FakeOllamaClient()).embed_query("q").shape == (4,)

    def test_tokens_are_counted(self) -> None:
        provider = make(FakeOllamaClient())
        provider.embed(["a", "b"])
        assert provider.tokens_seen == 2

    def test_warm_and_unload_use_keep_alive(self) -> None:
        client = FakeOllamaClient()
        provider = make(client)
        provider.warm_embedder(cpu=True)
        client.loaded = [provider.embed_model]
        client.loaded_on_cpu = {provider.embed_model}
        provider.unload_embedder()
        warm, unload = (c[1] for c in client.calls if c[0] == "embed")
        assert warm["options"]["num_gpu"] == 0
        assert unload["keep_alive"] == 0
        assert unload["options"]["num_gpu"] == 0  # matches how it was loaded: no second runner

    def test_resident_vram_sums_loaded_models_and_survives_an_outage(self) -> None:
        client = FakeOllamaClient()
        client.loaded = ["a", "b"]
        assert make(client).resident_vram_mb() == 1  # two fake 1,000,000-byte entries
        client.failures = [ConnectionError("down")] * 10
        assert make(client).resident_vram_mb() == 0

    def test_unload_does_not_load_a_model_that_is_not_resident(self) -> None:
        client = FakeOllamaClient()
        make(client).unload_embedder()
        assert [name for name, _ in client.calls] == ["ps"]

    def test_unload_of_a_gpu_resident_embedder_asks_for_gpu(self) -> None:
        client = FakeOllamaClient()
        provider = make(client)
        client.loaded = [provider.embed_model + ":latest"]
        provider.unload_embedder()
        embed = next(kw for name, kw in client.calls if name == "embed")
        assert "num_gpu" not in embed["options"]
        assert embed["keep_alive"] == 0

    def test_warm_and_unload_swallow_provider_errors(self) -> None:
        client = FakeOllamaClient()
        client.failures = [ollama.ResponseError("nope", 400)] * 2
        client.loaded = ["whatever"]
        provider = make(client)
        provider.warm_embedder()
        provider.unload_embedder()


class TestRetries:
    def test_transient_failures_are_retried(self) -> None:
        client = FakeOllamaClient()
        client.failures = [ConnectionError("x"), ollama.ResponseError("busy", 503)]
        assert make(client).embed(["a"]).shape == (1, 4)
        assert len(client.calls) == 3

    def test_a_refused_connection_gives_up_quickly(self) -> None:
        client = FakeOllamaClient()
        client.failures = [ConnectionError("down")] * 10
        slept: list[float] = []
        provider = om.OllamaProvider(EmbeddingSettings(), client=client, sleep=slept.append)
        with pytest.raises(ProviderUnavailableError):
            provider.embed(["a"])
        assert len(client.calls) == 2  # nobody is listening: not four attempts and ten seconds
        assert sum(slept) <= 1.0

    def test_timeouts_and_server_errors_still_get_the_full_set_of_retries(self) -> None:
        client = FakeOllamaClient()
        client.failures = [TimeoutError("slow")] * 10
        with pytest.raises(ProviderUnavailableError):
            make(client).embed(["a"])
        assert len(client.calls) == 4

    def test_the_real_client_has_a_short_connect_and_a_long_read_timeout(self) -> None:
        import httpx

        provider = om.OllamaProvider(EmbeddingSettings(), host="http://127.0.0.1:1")
        timeout = provider.client._client.timeout
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == om._CONNECT_TIMEOUT_SECONDS
        assert timeout.read == om._READ_TIMEOUT_SECONDS

    def test_missing_model_is_not_retried(self) -> None:
        client = FakeOllamaClient()
        client.failures = [ollama.ResponseError("model not found", 404)]
        with pytest.raises(ModelNotFoundError):
            make(client).embed(["a"])
        assert len(client.calls) == 1

    def test_client_errors_are_not_retried(self) -> None:
        client = FakeOllamaClient()
        client.failures = [ollama.ResponseError("bad", 400)]
        with pytest.raises(ProviderError):
            make(client).embed(["a"])
        assert len(client.calls) == 1


class TestChat:
    messages: ClassVar[list[Message]] = [Message("system", "be brief"), Message("user", "hi")]

    def test_stream_yields_text_then_usage(self) -> None:
        client = FakeOllamaClient()
        chunks = list(make(client).stream_chat(self.messages, "chat-model"))
        assert "".join(c.text for c in chunks) == "Hello"
        assert chunks[-1] == ChatChunk("", Usage(11, 5))
        sent = client.calls[0][1]
        assert sent["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
        assert sent["options"]["num_ctx"] == 8192
        assert sent["keep_alive"] == "10m"
        assert "num_gpu" not in sent["options"]

    def test_cpu_option(self) -> None:
        client = FakeOllamaClient()
        list(make(client).stream_chat(self.messages, "m", ChatOptions(cpu=True, keep_alive=0)))
        sent = client.calls[0][1]
        assert sent["options"]["num_gpu"] == 0
        assert sent["keep_alive"] == 0

    def test_interrupted_stream_is_unavailable(self) -> None:
        client = FakeOllamaClient()

        def broken() -> object:
            yield ollama.ChatResponse(message=ollama.Message(role="assistant", content="a"))
            raise ConnectionError("cut")

        client.chat = lambda **_kw: broken()  # type: ignore[method-assign]
        with pytest.raises(ProviderUnavailableError):
            list(make(client).stream_chat(self.messages, "m"))

    def test_abandoning_the_stream_closes_the_http_reader(self) -> None:
        client = FakeOllamaClient()
        closed: list[int] = []

        def reader() -> object:
            try:
                for piece in ("a", "b", "c"):
                    yield ollama.ChatResponse(
                        message=ollama.Message(role="assistant", content=piece)
                    )
            finally:
                closed.append(1)

        client.chat = lambda **_kw: reader()  # type: ignore[method-assign]
        stream = make(client).stream_chat(self.messages, "m")
        assert next(stream).text == "a"
        stream.close()
        assert closed == [1]

    def test_chat_json_parses_and_reports_usage(self) -> None:
        client = FakeOllamaClient()
        client.chat_json_reply = json.dumps({"a": 1})
        result = make(client).chat_json(self.messages, "m", {"type": "object"})
        assert result.data == {"a": 1}
        assert result.usage == Usage(7, 3)
        assert client.calls[0][1]["format"] == {"type": "object"}

    def test_chat_json_rejects_invalid_json(self) -> None:
        client = FakeOllamaClient()
        client.chat_json_reply = "not json"
        with pytest.raises(InvalidJsonError, match="invalid JSON"):
            make(client).chat_json(self.messages, "m", {})

    @pytest.mark.parametrize(
        ("status", "error"), [(400, ProviderError), (500, ProviderUnavailableError)]
    )
    def test_an_error_inside_the_stream_is_a_provider_error(
        self, status: int, error: type[ProviderError]
    ) -> None:
        client = FakeOllamaClient()

        def failing() -> object:
            yield ollama.ChatResponse(message=ollama.Message(role="assistant", content="a"))
            raise ollama.ResponseError("runner terminated", status)

        client.chat = lambda **_kw: failing()  # type: ignore[method-assign]
        with pytest.raises(error, match="runner terminated"):
            list(make(client).stream_chat(self.messages, "m"))

    def test_a_prompt_that_fills_the_context_window_is_flagged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeOllamaClient()
        client.chat_json_reply = json.dumps({"a": 1})
        provider = make(client)
        with caplog.at_level("DEBUG", logger=om.__name__):
            provider.chat_json(self.messages, "m", {}, ChatOptions(num_ctx=7))  # 7 prompt tokens
            provider.chat_json(self.messages, "m", {}, ChatOptions(num_ctx=8192))
        cut, fine = caplog.records
        assert cut.levelname == "WARNING" and "cut" in cut.getMessage()
        assert fine.levelname == "DEBUG"
        assert (cut.prompt_tokens, cut.num_ctx) == (7, 7)  # type: ignore[attr-defined]

    def test_prewarm_and_unload(self) -> None:
        client = FakeOllamaClient()
        provider = make(client)
        provider.prewarm("m", ChatOptions(cpu=True))
        provider.unload("m")
        warm, unload = (c[1] for c in client.calls)
        assert warm["prompt"] == "" and warm["options"]["num_gpu"] == 0
        assert unload["keep_alive"] == 0

    def test_prewarm_and_unload_swallow_errors(self) -> None:
        client = FakeOllamaClient()
        client.failures = [ollama.ResponseError("x", 400)] * 2
        provider = make(client)
        provider.prewarm("m")
        provider.unload("m")

    def test_unload_all(self) -> None:
        client = FakeOllamaClient()
        client.loaded = ["a", "b"]
        provider = make(client)
        assert provider.loaded_models() == ["a", "b"]
        provider.unload_all()
        unloads = [c[1]["model"] for c in client.calls if c[0] == "generate"]
        assert unloads == ["a", "b"]

    def test_cost_is_zero_locally(self) -> None:
        assert make(FakeOllamaClient()).estimate_cost("m", Usage(1000, 1000)) == 0.0


class TestDiscovery:
    models: ClassVar[dict[str, dict[str, object]]] = {
        "chat:9b": {
            "size": 5,
            "params": "9B",
            "quant": "Q4_K_M",
            "caps": ["completion", "tools"],
            "info": {"qwen.context_length": 32768, "other": 1},
        },
        "embed:0.6b": {"caps": ["embedding"], "info": {}},
        "broken": {"show_fails": True},
    }

    def test_list_models_describes_capabilities(self) -> None:
        provider = make(FakeOllamaClient(models=self.models))
        listed = {m.name: m for m in provider.list_models()}
        chat = listed["chat:9b"]
        assert chat.parameter_size == "9B"
        assert chat.quantization == "Q4_K_M"
        assert chat.context_length == 32768
        assert chat.capabilities == {"completion", "tools"}
        assert listed["embed:0.6b"].is_embedding
        assert listed["embed:0.6b"].context_length is None
        assert listed["broken"].capabilities == frozenset()

    def test_embedding_models_and_has_model(self) -> None:
        provider = make(FakeOllamaClient(models=self.models))
        assert provider.embedding_models() == ["embed:0.6b"]
        assert provider.has_model("chat:9b")
        assert not provider.has_model("nope")
        assert CAP_EMBEDDING in provider.capabilities("embed:0.6b")

    def test_latest_tag_is_implied(self) -> None:
        provider = make(FakeOllamaClient(models={"bge-m3:latest": {}}))
        assert provider.has_model("bge-m3")

    def test_pull_reports_progress(self) -> None:
        provider = make(FakeOllamaClient())
        progress = list(provider.pull("m"))
        assert progress[0].fraction == 0.5
        assert progress[1].fraction == 0.0
