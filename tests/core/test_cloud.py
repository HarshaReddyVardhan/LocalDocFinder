from pathlib import Path
from typing import Any

import pytest
from tests.core.conftest import Chat, Env, Power
from tests.core.fakes import FakeCloudInner

from vector_embed.core.cloud import (
    CloudChatProvider,
    CloudConsent,
    CloudDestination,
    CloudRouter,
    _Restorer,
)
from vector_embed.core.llm import ChatBlockedError
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.providers.base import (
    Message,
    Usage,
)
from vector_embed.core.settings import (
    CloudProviderSettings,
    CloudSettings,
    PrivacySettings,
)

KEY_TEXT = "Passport No: K1234567 and SSN 123-45-6789"


class Clock:
    now = 1_780_000_000.0

    def __call__(self) -> float:
        return self.now


RESUME = "Jane Doe\njane@example.com\n" + KEY_TEXT + "\nBuilt systems in Python."


def make(env: Env, *, budget: float | None = None, redact: bool = False, consent: bool = True):  # type: ignore[no-untyped-def]
    inner = FakeCloudInner()
    privacy = PrivacyFilter(PrivacySettings(redact_personal=redact), env.scope)
    settings = CloudSettings(monthly_budget_usd=budget)
    gate = CloudConsent()
    if consent:
        gate.grant()
    clock = Clock()
    provider = CloudChatProvider(inner, privacy, env.state, settings, gate, clock=clock)
    return provider, inner, gate, clock


MESSAGES = [Message("system", "judge"), Message("user", f"Resume (r.pdf):\n{RESUME}")]


class TestCloudChatProvider:
    def test_requests_need_consent(self, env: Env) -> None:
        provider, inner, gate, _ = make(env, consent=False)
        with pytest.raises(ChatBlockedError, match="consent"):
            list(provider.stream_chat(MESSAGES, "m"))
        assert inner.sent == []
        gate.grant()
        assert "".join(c.text for c in provider.stream_chat(MESSAGES, "m"))
        gate.revoke()
        with pytest.raises(ChatBlockedError):
            provider.chat_json(MESSAGES, "m", {})

    def test_ids_never_reach_the_cloud(self, env: Env) -> None:
        provider, inner, _, _ = make(env)
        list(provider.stream_chat(MESSAGES, "m"))
        sent = "\n".join(m.content for m in inner.sent[0])
        assert "K1234567" not in sent
        assert "123-45-6789" not in sent
        assert "[PASSPORT REMOVED]" in sent
        assert "jane@example.com" in sent  # personal details stay unless the box is ticked
        assert provider.last_outbound is not None
        assert len(provider.last_outbound.findings) == 2

    def test_personal_details_are_redacted_and_restored_in_the_streamed_answer(
        self, env: Env
    ) -> None:
        provider, inner, _, _ = make(env, redact=True)
        text = "".join(c.text for c in provider.stream_chat(MESSAGES, "m"))
        sent = "\n".join(m.content for m in inner.sent[0])
        assert "jane@example.com" not in sent
        assert "[EMAIL_1]" in sent
        assert text == "Contact Jane Doe at jane@example.com."  # placeholder split across chunks

    def test_json_answers_are_restored_recursively(self, env: Env) -> None:
        provider, inner, _, _ = make(env, redact=True)
        inner.json_data = {"summary": "[NAME_1] fits", "rows": [{"quote": "[EMAIL_1]"}], "n": 3}
        result = provider.chat_json(MESSAGES, "m", {})
        assert result.data == {
            "summary": "Jane Doe fits",
            "rows": [{"quote": "jane@example.com"}],
            "n": 3,
        }

    def test_usage_and_cost_are_recorded(self, env: Env) -> None:
        provider, _, _, _ = make(env)
        list(provider.stream_chat(MESSAGES, "m"))
        provider.chat_json(MESSAGES, "m", {})
        (total,) = env.state.usage_totals()
        assert (total.provider, total.model) == ("openrouter", "m")
        assert (total.prompt_tokens, total.completion_tokens) == (2000, 1000)
        assert total.cost_usd == pytest.approx(3.0)

    def test_the_monthly_budget_blocks_further_calls(self, env: Env) -> None:
        provider, inner, _, _ = make(env, budget=2.0)
        list(provider.stream_chat(MESSAGES, "m"))  # costs $1.50
        env.state.record_usage("openrouter", "m", 1, 1, 1.0)  # pushes spend over the budget
        with pytest.raises(ChatBlockedError, match="budget"):
            list(provider.stream_chat(MESSAGES, "m"))
        assert len(inner.sent) == 1

    def test_passthrough_methods(self, env: Env) -> None:
        provider, _, _, _ = make(env)
        assert provider.name == "openrouter"
        assert provider.label == "OpenRouter"
        assert [m.name for m in provider.list_models()] == ["m"]
        assert provider.capabilities("m") == {"completion"}
        assert provider.estimate_cost("m", Usage(500, 500)) == 1.0

    def test_prepare_shows_what_would_be_sent(self, env: Env) -> None:
        provider, _, _, _ = make(env)
        outbound = provider.prepare(MESSAGES)
        assert "[SSN REMOVED]" in outbound.messages[1].content


class TestRestorer:
    def test_text_without_placeholders_passes_through(self, env: Env) -> None:
        provider, _, _, _ = make(env)
        restorer = _Restorer(provider.prepare([Message("user", "hello")]))
        assert restorer.feed("a [b") == "a [b"
        assert restorer.flush() == ""

    def test_an_unterminated_bracket_is_flushed_at_the_end(self, env: Env) -> None:
        provider, _, _, _ = make(env, redact=True)
        outbound = provider.prepare([Message("user", "Jane Doe\njane@example.com")])
        restorer = _Restorer(outbound)
        assert restorer.feed("see [NAME") == "see "
        assert restorer.feed("_1] ok [x") == "Jane Doe ok "
        assert restorer.flush() == "[x"


class TestRouter:
    def setup_cloud(
        self, env: Env, chat: Chat, **overrides: Any
    ) -> tuple[CloudRouter, CloudChatProvider, Chat]:
        inner = FakeCloudInner()
        settings = CloudSettings(
            providers={
                "openrouter": CloudProviderSettings(
                    base_url="https://openrouter.ai/api/v1",
                    label="OpenRouter",
                    models={"chat": "vendor/chat-model", "match_scorer": "vendor/judge"},
                )
            },
            active="openrouter",
            **overrides,
        )
        consent = CloudConsent()
        consent.grant()
        provider = CloudChatProvider(
            inner, PrivacyFilter(PrivacySettings(), env.scope), env.state, settings, consent
        )
        router = CloudRouter(settings, provider, chat.gateway._registry)
        chat.gateway.router = router
        return router, provider, chat

    def test_local_is_the_default(self, env: Env, chat: Chat, power_state: Power) -> None:
        router, _, _ = self.setup_cloud(env, chat)
        assert chat.gateway.target().local
        power_state.on_ac = False  # even on battery, "local" never goes to the cloud
        assert router("chat", chat.gateway._power) is None

    def test_cloud_policy_always_uses_the_cloud(self, env: Env, chat: Chat) -> None:
        _, provider, _ = self.setup_cloud(
            env, chat, routing={"chat": "cloud", "match_scorer": "cloud", "summarizer": "cloud"}
        )
        target = chat.gateway.target()
        assert (target.local, target.model, target.provider) == (
            False,
            "vendor/chat-model",
            provider,
        )
        assert chat.gateway.will_use_cloud()
        assert chat.gateway.target("match_scorer").model == "vendor/judge"
        assert chat.gateway.target("summarizer").model == "vendor/chat-model"  # falls back to chat

    def test_auto_stays_local_unless_a_reason_appears(
        self, env: Env, chat: Chat, power_state: Power
    ) -> None:
        self.setup_cloud(env, chat, routing={"chat": "auto"})
        assert chat.gateway.target().local
        power_state.on_ac = False
        assert not chat.gateway.target().local  # battery: the cloud uses no local GPU
        power_state.on_ac = True
        for name in list(chat.client.models):
            if name != "qwen3-embedding:0.6b":
                del chat.client.models[name]
        chat.gateway._registry.refresh()
        assert not chat.gateway.target().local  # no local chat model fits or exists

    def test_answer_better_escalates_one_request_until_reset(self, env: Env, chat: Chat) -> None:
        router, _, _ = self.setup_cloud(env, chat)
        router.escalate = True
        assert not chat.gateway.target().local
        assert not chat.gateway.target().local  # still escalated until reset
        router.reset()
        assert chat.gateway.target().local

    def test_local_only_requests_bypass_the_router(self, env: Env, chat: Chat) -> None:
        self.setup_cloud(env, chat, routing={"chat": "cloud"})
        assert not chat.gateway.target().local
        assert chat.gateway.target(local_only=True).local

    def test_cloud_without_configuration_is_a_clear_error(self, env: Env, chat: Chat) -> None:
        settings = CloudSettings(routing={"chat": "cloud"})
        router = CloudRouter(settings, None, chat.gateway._registry)
        chat.gateway.router = router
        with pytest.raises(ChatBlockedError, match="no provider/model/key"):
            chat.gateway.target()
        assert not chat.gateway.will_use_cloud()
        auto = CloudRouter(CloudSettings(routing={"chat": "auto"}), None, chat.gateway._registry)
        chat.gateway.router = auto
        assert chat.gateway.target().local  # auto quietly stays local when no cloud is set up

    def test_destination_describes_where_a_request_would_go(self, env: Env, chat: Chat) -> None:
        router, _, _ = self.setup_cloud(env, chat)
        destination = router.destination("chat")
        assert destination == CloudDestination("OpenRouter", "vendor/chat-model")
        assert str(destination) == "OpenRouter / vendor/chat-model"
        assert (
            CloudRouter(CloudSettings(), None, chat.gateway._registry).destination("chat") is None
        )

    def test_an_unknown_active_provider_means_no_cloud(self, env: Env, chat: Chat) -> None:
        settings = CloudSettings(active="ghost")
        assert CloudRouter(settings, None, chat.gateway._registry).model_for("chat") is None


def test_cloud_settings_defaults_are_safe() -> None:
    settings = CloudSettings()
    assert settings.providers == {}
    assert settings.policy("chat") == "local"
    assert settings.monthly_budget_usd is None


def test_state_paths_are_not_needed(tmp_path: Path) -> None:
    assert tmp_path.exists()
