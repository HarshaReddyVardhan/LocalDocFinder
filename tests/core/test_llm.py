from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import pytest
from tests.core.providers.fakes import FakeOllamaClient

from localdoc_finder.core import llm
from localdoc_finder.core.llm import (
    ChatBlockedError,
    ChatTarget,
    ConsentRequiredError,
    LlmGateway,
    NoChatModelError,
)
from localdoc_finder.core.models.catalog import load_catalog
from localdoc_finder.core.models.hardware import Hardware
from localdoc_finder.core.models.registry import ModelRegistry
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.privacy.policy import PrivacyFilter
from localdoc_finder.core.providers.base import (
    ChatChunk,
    ChatOptions,
    InvalidJsonError,
    JsonResult,
    Message,
    ModelInfo,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    Usage,
)
from localdoc_finder.core.providers.ollama import OllamaProvider
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.settings import (
    ChatSettings,
    EmbeddingSettings,
    PowerSettings,
    PrivacySettings,
    ScopeSettings,
)
from localdoc_finder.core.store.sqlite import CHAT_LOCK, StateDb

GPU = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
MODELS = {
    "qwen3-embedding:0.6b": {"caps": ["embedding"], "size": 600 * 1024**2},
    "qwen3.5:9b": {"caps": ["completion"], "size": 5 * 1024**3},
}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class World:
    def __init__(self, tmp_path: Path, models: dict[str, dict[str, object]] | None = None) -> None:
        self.client = FakeOllamaClient(models=models if models is not None else MODELS)
        self.provider = OllamaProvider(
            EmbeddingSettings(), client=self.client, sleep=lambda _s: None
        )
        self.client.loaded = [
            self.provider.embed_model
        ]  # the embedder is resident, as after a search
        self.state = StateDb(tmp_path / "data")
        self.ac = True
        self.fullscreen = False
        self.clock = Clock()
        self.power = PowerGate(PowerSettings(), probe=lambda: self.ac)
        registry = ModelRegistry(
            load_catalog(), [self.provider], self.state, hardware_probe=lambda: GPU
        )
        self.gateway = LlmGateway(
            ChatSettings(),
            registry,
            self.provider,
            self.state,
            self.power,
            clock=self.clock,
            fullscreen=lambda: self.fullscreen,
        )

    def calls(self, kind: str) -> list[dict[str, object]]:
        return [kw for k, kw in self.client.calls if k == kind]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


MESSAGES = [Message("user", "hi")]


class TestTarget:
    def test_resolves_the_chat_model_locally(self, world: World) -> None:
        target = world.gateway.target()
        assert (target.model, target.local) == ("qwen3.5:9b", True)

    def test_unknown_role_falls_back_to_chat(self, world: World) -> None:
        assert world.gateway.target("code_chat").model == "qwen3.5:9b"

    def test_no_model_gives_a_pull_hint(self, tmp_path: Path) -> None:
        empty = World(tmp_path, models={})
        with pytest.raises(NoChatModelError, match=r"ollama pull qwen3.5:9b"):
            empty.gateway.target()

    def test_battery_blocks_local_chat(self, world: World) -> None:
        world.ac = False
        with pytest.raises(ChatBlockedError, match="plug in"):
            world.gateway.target()

    def test_router_can_override_the_local_choice(self, world: World) -> None:
        cloud = ChatTarget("chat", "gpt-x", world.provider, local=False)
        world.ac = False
        world.gateway.router = lambda role, power: cloud
        assert world.gateway.target() is cloud
        world.gateway.router = lambda role, power: None
        with pytest.raises(ChatBlockedError):
            world.gateway.target()

    def test_options_depend_on_session(self, world: World) -> None:
        session, one_shot = (
            world.gateway.options(session=True),
            world.gateway.options(session=False),
        )
        assert (session.num_ctx, session.keep_alive) == (8192, "10m")
        assert one_shot.keep_alive == 0


class TestOneShot:
    def test_streaming_unloads_the_embedder_first_and_uses_keep_alive_zero(
        self, world: World
    ) -> None:
        text = "".join(c.text for c in world.gateway.stream(MESSAGES))
        assert text == "Hello"
        order = [k for k, _ in world.client.calls if k in {"embed", "chat"}]
        assert order == ["embed", "chat"]  # embedder unloaded (keep_alive=0) before the LLM loads
        assert world.calls("embed")[0]["keep_alive"] == 0
        assert world.calls("chat")[0]["keep_alive"] == 0
        assert world.gateway._loaded == set()

    def test_json_call(self, world: World) -> None:
        world.client.chat_json_reply = '{"ok": 1}'
        result = world.gateway.chat_json(MESSAGES, {"type": "object"})
        assert result.data == {"ok": 1}
        assert world.calls("chat")[0]["keep_alive"] == 0


class TestLease:
    def test_a_one_shot_call_holds_the_chat_lock_while_it_runs(self, world: World) -> None:
        stream = world.gateway.stream(MESSAGES)
        next(stream)
        assert world.state.lock_held(CHAT_LOCK)  # the indexer and the search stay off the GPU
        list(stream)
        assert not world.state.lock_held(CHAT_LOCK)

    def test_json_calls_hold_the_lock_too(self, world: World) -> None:
        seen: list[bool] = []
        world.client.chat_json_fn = lambda _kw: (
            seen.append(world.state.lock_held(CHAT_LOCK)) or "{}"
        )
        world.gateway.chat_json(MESSAGES, {"type": "object"})
        assert seen == [True]
        assert not world.state.lock_held(CHAT_LOCK)

    def test_another_process_chatting_blocks_a_one_shot_call(self, world: World) -> None:
        world.state.acquire_lock(CHAT_LOCK, "other-process", 60)
        with pytest.raises(ChatBlockedError):
            list(world.gateway.stream(MESSAGES))

    def test_overlapping_one_shot_calls_share_the_lock_until_the_last_ends(
        self, world: World
    ) -> None:
        first = world.gateway.stream(MESSAGES)
        second = world.gateway.stream(MESSAGES)
        next(first)
        next(second)
        list(first)
        assert world.state.lock_held(CHAT_LOCK)  # the second call is still generating
        list(second)
        assert not world.state.lock_held(CHAT_LOCK)

    def test_a_session_ending_during_a_one_shot_call_leaves_it_the_lock(self, world: World) -> None:
        stream = world.gateway.stream(MESSAGES)
        next(stream)
        world.gateway.end_chat()
        assert world.state.lock_held(CHAT_LOCK)
        list(stream)
        assert not world.state.lock_held(CHAT_LOCK)

    def test_parallel_json_calls_keep_the_bookkeeping_consistent(self, world: World) -> None:
        world.client.chat_json_fn = lambda _kw: "{}"
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _i: world.gateway.chat_json(MESSAGES, {}), range(64)))
        assert world.gateway._one_shot_calls == 0
        assert not world.gateway._loaded
        assert not world.state.lock_held(CHAT_LOCK)

    def test_the_lock_is_released_when_the_caller_abandons_the_stream(self, world: World) -> None:
        stream = world.gateway.stream(MESSAGES)
        next(stream)
        stream.close()
        assert not world.state.lock_held(CHAT_LOCK)

    def test_other_processes_searches_move_to_the_cpu(self, world: World) -> None:
        assert not world.gateway.query_on_cpu
        world.state.acquire_lock(CHAT_LOCK, "other-process", 60)
        assert world.gateway.query_on_cpu

    def test_the_lease_is_renewed_at_most_every_thirty_seconds(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gw = world.gateway
        gw.begin_chat()
        renewals: list[int] = []
        original = world.state.acquire_lock
        monkeypatch.setattr(
            world.state,
            "acquire_lock",
            lambda *a, **k: renewals.append(1) or original(*a, **k),
        )
        for _ in range(50):  # one touch per streamed token
            gw.touch()
        assert renewals == []
        world.clock.now += 31
        gw.touch()
        gw.touch()
        assert renewals == [1]


class TestLocalFilter:
    ID_MESSAGES: ClassVar[list[Message]] = [Message("user", "My SSN 123-45-6789 please summarise")]

    def test_local_models_see_ids_when_the_option_is_off(self, world: World) -> None:
        list(world.gateway.stream(self.ID_MESSAGES))
        assert "123-45-6789" in str(world.calls("chat")[0]["messages"])

    def test_the_option_masks_ids_for_local_models_too(self, world: World) -> None:
        privacy = PrivacyFilter(PrivacySettings(), ScopePolicy(ScopeSettings()))
        world.gateway.local_filter = privacy.mask_ids
        list(world.gateway.stream(self.ID_MESSAGES))
        sent = str(world.calls("chat")[0]["messages"])
        assert "123-45-6789" not in sent
        assert "[SSN REMOVED]" in sent

    def test_json_calls_are_masked_too(self, world: World) -> None:
        privacy = PrivacyFilter(PrivacySettings(), ScopePolicy(ScopeSettings()))
        world.gateway.local_filter = privacy.mask_ids
        world.client.chat_json_reply = "{}"
        world.gateway.chat_json(self.ID_MESSAGES, {"type": "object"})
        assert "123-45-6789" not in str(world.calls("chat")[0]["messages"])


class TestSessionPinning:
    def test_the_model_does_not_change_mid_session(self, world: World) -> None:
        gw = world.gateway
        gw.begin_chat()
        first = gw.target().model
        calls: list[str] = []
        original = gw._registry.resolve
        gw._registry.resolve = lambda role, hw=None: (  # type: ignore[method-assign,assignment]
            calls.append(role),
            original(role, hw),
        )[1]
        assert gw.target().model == first
        assert calls == []  # answered from the pin, not re-resolved
        gw.end_chat()
        gw.target()
        assert calls == ["chat"]


class TestSession:
    def test_begin_takes_the_lock_and_moves_queries_to_the_cpu(self, world: World) -> None:
        gw = world.gateway
        gw.begin_chat()
        assert world.state.lock_held(CHAT_LOCK)
        assert gw.query_on_cpu
        assert world.calls("embed")[0]["keep_alive"] == 0

    def test_second_session_is_refused(self, world: World) -> None:
        world.gateway.begin_chat()
        other = LlmGateway(
            ChatSettings(), world.gateway._registry, world.provider, world.state, world.power,
            owner="other",
        )  # fmt: skip
        with pytest.raises(ChatBlockedError, match="another chat"):
            other.begin_chat()

    def test_session_keeps_the_model_warm_until_end(self, world: World) -> None:
        gw = world.gateway
        gw.begin_chat()
        list(gw.stream(MESSAGES, session=True))
        assert world.calls("chat")[0]["keep_alive"] == "10m"
        assert gw._loaded == {"qwen3.5:9b"}
        gw.end_chat()
        assert [c["keep_alive"] for c in world.calls("generate")] == [0]
        assert not world.state.lock_held(CHAT_LOCK)
        assert not gw.query_on_cpu

    def test_prewarm_loads_without_blocking_the_caller(self, world: World) -> None:
        world.gateway.prewarm()
        generate = world.calls("generate")[0]
        assert generate["keep_alive"] == "10m"
        assert generate["prompt"] == ""
        assert world.gateway._loaded == {"qwen3.5:9b"}

    def test_touch_refreshes_the_lease(self, world: World) -> None:
        gw = world.gateway
        gw.begin_chat()
        world.clock.now += 500
        gw.touch()
        assert world.state.lock_held(CHAT_LOCK)

    def test_idle_timeout_unloads(self, world: World) -> None:
        gw = world.gateway
        gw.begin_chat()
        list(gw.stream(MESSAGES, session=True))
        assert gw.check() is None
        world.clock.now += 601
        assert gw.check() == llm.REASON_IDLE
        assert gw._loaded == set()
        assert gw.check() is None  # nothing left to unload

    def test_unplugging_unloads(self, world: World) -> None:
        gw = world.gateway
        gw.begin_chat()
        list(gw.stream(MESSAGES, session=True))
        world.ac = False
        assert gw.check() == llm.REASON_UNPLUGGED
        assert world.calls("generate")[-1]["keep_alive"] == 0

    def test_fullscreen_game_unloads(self, world: World) -> None:
        gw = world.gateway
        gw.begin_chat()
        list(gw.stream(MESSAGES, session=True))
        world.fullscreen = True
        assert gw.check() == llm.REASON_FULLSCREEN
        assert not world.state.lock_held(CHAT_LOCK)

    def test_check_without_a_session_is_a_noop(self, world: World) -> None:
        assert world.gateway.check() is None
        assert world.calls("generate") == []

    def test_embedder_unload_failures_do_not_block_chat(self, world: World) -> None:
        import ollama

        world.client.failures = [ollama.ResponseError("nope", 400)]
        world.gateway.begin_chat()
        assert world.gateway.session_active


class FlakyCloud:
    """A cloud provider that fails as scripted: before its first chunk, or part-way through."""

    name = "flaky"
    label = "Flaky Cloud"

    def __init__(self, error: Exception | None = None, *, after: int = 0) -> None:
        self.error = error
        self.after = after  # chunks delivered before the error
        self.calls = 0

    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]:
        self.calls += 1
        for index in range(self.after):
            yield ChatChunk(f"c{index}")
        if self.error is not None:
            raise self.error
        yield ChatChunk("cloud answer")
        yield ChatChunk("", Usage(1, 1))

    def chat_json(
        self,
        messages: list[Message],
        model: str,
        schema: dict[str, Any],
        options: ChatOptions | None = None,
    ) -> JsonResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return JsonResult({"from": "cloud"})

    def list_models(self) -> list[ModelInfo]:
        return []

    def capabilities(self, model: str) -> frozenset[str]:
        return frozenset()

    def estimate_cost(self, model: str, usage: Usage) -> float:
        return 0.0


def routed(cloud: FlakyCloud, *, fallback: bool = True) -> ChatTarget:
    return ChatTarget("chat", "vendor/m", cloud, local=False, fallback=fallback)


class TestCloudFallback:
    def test_a_routed_call_that_fails_at_once_is_answered_locally_with_a_notice(
        self, world: World
    ) -> None:
        cloud = FlakyCloud(ProviderUnavailableError("Flaky Cloud unavailable: rate limit"))
        chunks = list(world.gateway.stream(MESSAGES, target=routed(cloud)))
        assert "".join(c.text for c in chunks) == "Hello"
        assert chunks[0].notice == (
            "Flaky Cloud unavailable: rate limit; answered locally with qwen3.5:9b"
        )
        assert all(c.notice is None for c in chunks[1:])
        assert world.calls("chat")  # the local model really answered

    @pytest.mark.parametrize(
        "error",
        [
            ProviderError("Flaky Cloud: the API key was rejected"),
            ModelNotFoundError("no such model"),
            ChatBlockedError("the monthly cloud budget ($5.00) would be exceeded"),
        ],
    )
    def test_rejected_keys_unknown_models_and_the_budget_fall_back_too(
        self, world: World, error: Exception
    ) -> None:
        chunks = list(world.gateway.stream(MESSAGES, target=routed(FlakyCloud(error))))
        assert "".join(c.text for c in chunks) == "Hello"
        assert str(error) in (chunks[0].notice or "")

    def test_nothing_falls_back_once_text_has_streamed(self, world: World) -> None:
        cloud = FlakyCloud(ProviderUnavailableError("dropped"), after=2)
        received: list[str] = []
        with pytest.raises(ProviderUnavailableError):
            for chunk in world.gateway.stream(MESSAGES, target=routed(cloud)):
                received.append(chunk.text)
        assert received == ["c0", "c1"]
        assert world.calls("chat") == []  # the local model was never asked

    def test_answer_better_shows_the_error_instead(self, world: World) -> None:
        cloud = FlakyCloud(ProviderUnavailableError("down"))
        with pytest.raises(ProviderUnavailableError):
            list(world.gateway.stream(MESSAGES, target=routed(cloud, fallback=False)))
        assert world.calls("chat") == []

    def test_a_missing_consent_is_never_answered_locally(self, world: World) -> None:
        cloud = FlakyCloud(ConsentRequiredError("cloud requests need your consent"))
        with pytest.raises(ConsentRequiredError):
            list(world.gateway.stream(MESSAGES, target=routed(cloud)))
        assert world.calls("chat") == []

    def test_when_the_local_model_cannot_answer_either_the_cloud_error_is_shown(
        self, world: World
    ) -> None:
        world.ac = False  # on battery: no local chat
        cloud = FlakyCloud(ProviderUnavailableError("Flaky Cloud unavailable: down"))
        with pytest.raises(ProviderUnavailableError, match="Flaky Cloud unavailable"):
            list(world.gateway.stream(MESSAGES, target=routed(cloud)))

    def test_a_local_target_is_never_retried(self, world: World) -> None:
        flaky = FlakyCloud(ProviderError("boom"))
        target = ChatTarget("chat", "qwen3.5:9b", flaky, local=True, fallback=True)
        with pytest.raises(ProviderError):
            list(world.gateway.stream(MESSAGES, target=target))
        assert flaky.calls == 1 and world.calls("chat") == []

    def test_a_long_provider_error_is_shortened_in_the_notice(self, world: World) -> None:
        cloud = FlakyCloud(ProviderUnavailableError("x" * 500))
        stream = world.gateway.stream(MESSAGES, target=routed(cloud))
        notice = next(stream).notice
        stream.close()
        assert notice is not None
        assert len(notice) < 200
        assert "…" in notice

    def test_json_calls_fall_back_and_carry_the_notice(self, world: World) -> None:
        world.client.chat_json_reply = '{"ok": 1}'
        cloud = FlakyCloud(ProviderUnavailableError("rate limit"))
        result = world.gateway.chat_json(MESSAGES, {"type": "object"}, target=routed(cloud))
        assert result.data == {"ok": 1}
        assert result.notice == "rate limit; answered locally with qwen3.5:9b"

    def test_json_from_the_cloud_has_no_notice(self, world: World) -> None:
        result = world.gateway.chat_json(MESSAGES, {"type": "object"}, target=routed(FlakyCloud()))
        assert (result.data, result.notice) == ({"from": "cloud"}, None)

    def test_unusable_json_is_left_to_the_callers_retry_not_answered_locally(
        self, world: World
    ) -> None:
        cloud = FlakyCloud(InvalidJsonError("not json"))
        with pytest.raises(InvalidJsonError):
            world.gateway.chat_json(MESSAGES, {"type": "object"}, target=routed(cloud))
        assert world.calls("chat") == []

    def test_escalated_and_consent_cases_for_json(self, world: World) -> None:
        with pytest.raises(ProviderUnavailableError):
            world.gateway.chat_json(
                MESSAGES,
                {},
                target=routed(FlakyCloud(ProviderUnavailableError("down")), fallback=False),
            )
        with pytest.raises(ConsentRequiredError):
            world.gateway.chat_json(
                MESSAGES, {}, target=routed(FlakyCloud(ConsentRequiredError("consent")))
            )
