from pathlib import Path
from typing import ClassVar

import pytest
from tests.core.providers.fakes import FakeOllamaClient

from vector_embed.core import llm
from vector_embed.core.llm import ChatBlockedError, ChatTarget, LlmGateway, NoChatModelError
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.power import PowerGate
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.providers.base import Message
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import (
    ChatSettings,
    EmbeddingSettings,
    PowerSettings,
    PrivacySettings,
    ScopeSettings,
)
from vector_embed.core.store.sqlite import CHAT_LOCK, StateDb

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
