import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from vector_embed.core.providers.base import (
    ChatChunk,
    ChatOptions,
    ChatProvider,
    InvalidJsonError,
    Message,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    Usage,
)
from vector_embed.core.providers.openai_compat import OpenAICompatibleProvider
from vector_embed.core.settings import CloudProviderSettings

KEY = "sk-secret-key-1234567890abcdef"
REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


def api_error(cls: type[openai.APIStatusError], status: int, message: str = "boom") -> Exception:
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)


def event(text: str | None = None, usage: Any = None, choices: bool = True) -> SimpleNamespace:
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if choices else [], usage=usage)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stream_events: list[SimpleNamespace] = []
        self.json_reply = '{"ok": true}'
        self.errors: list[Exception] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        if kwargs.get("stream"):
            return iter(self.stream_events)
        message = SimpleNamespace(content=self.json_reply)
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=4)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.model_entries: list[SimpleNamespace] = []
        self.models = SimpleNamespace(list=self._list)
        self.list_error: Exception | None = None

    def _list(self) -> SimpleNamespace:
        if self.list_error:
            raise self.list_error
        return SimpleNamespace(data=self.model_entries)


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def provider(client: FakeClient) -> OpenAICompatibleProvider:
    settings = CloudProviderSettings(
        base_url="https://openrouter.ai/api/v1",
        label="OpenRouter",
        pricing={"cheap/model": (1.0, 2.0)},
    )
    return OpenAICompatibleProvider("openrouter", settings, KEY, client=client)


MESSAGES = [Message("system", "be brief"), Message("user", "hi")]


def test_satisfies_the_chat_provider_protocol(provider: OpenAICompatibleProvider) -> None:
    assert isinstance(provider, ChatProvider)
    assert provider.name == "openrouter"
    assert provider.label == "OpenRouter"


class TestStreaming:
    def test_yields_text_then_usage(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        usage = SimpleNamespace(prompt_tokens=20, completion_tokens=6)
        client.completions.stream_events = [
            event("Hel"),
            event(None),
            event("lo"),
            event(None, usage=usage, choices=False),
        ]
        chunks = list(provider.stream_chat(MESSAGES, "m", ChatOptions(temperature=0.5)))
        assert "".join(c.text for c in chunks) == "Hello"
        assert chunks[-1] == ChatChunk("", Usage(20, 6))
        sent = client.completions.calls[0]
        assert sent["model"] == "m"
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["temperature"] == 0.5
        assert sent["messages"][1] == {"role": "user", "content": "hi"}

    def test_stream_without_usage_reports_zero(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.completions.stream_events = [event("x")]
        assert list(provider.stream_chat(MESSAGES, "m"))[-1] == ChatChunk("", Usage())

    @pytest.mark.parametrize(
        ("error", "expected", "text"),
        [
            (
                api_error(openai.AuthenticationError, 401, f"bad key {KEY}"),
                ProviderError,
                "rejected",
            ),
            (
                api_error(openai.NotFoundError, 404, "no such model"),
                ModelNotFoundError,
                "no such model",
            ),
            (
                api_error(openai.RateLimitError, 429, "slow down"),
                ProviderUnavailableError,
                "unavailable",
            ),
            (openai.APIConnectionError(request=REQUEST), ProviderUnavailableError, "unavailable"),
            (api_error(openai.InternalServerError, 500, f"oops {KEY}"), ProviderError, "[API KEY]"),
        ],
    )
    def test_errors_are_translated_and_never_leak_the_key(
        self,
        provider: OpenAICompatibleProvider,
        client: FakeClient,
        error: Exception,
        expected: type[Exception],
        text: str,
    ) -> None:
        client.completions.errors = [error]
        with pytest.raises(expected) as raised:
            list(provider.stream_chat(MESSAGES, "m"))
        assert text in str(raised.value)
        assert KEY not in str(raised.value)


class TestJson:
    def test_uses_json_schema_mode(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.completions.json_reply = json.dumps({"a": 1})
        result = provider.chat_json(MESSAGES, "m", {"type": "object"})
        assert result.data == {"a": 1}
        assert result.usage == Usage(11, 4)
        sent = client.completions.calls[0]
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["response_format"]["json_schema"]["schema"] == {"type": "object"}

    def test_falls_back_to_json_object_mode(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.completions.errors = [
            api_error(openai.BadRequestError, 400, "response_format json_schema is not supported")
        ]
        assert provider.chat_json(MESSAGES, "m", {"type": "object"}).data == {"ok": True}
        retry = client.completions.calls[1]
        assert retry["response_format"] == {"type": "json_object"}
        assert "Reply with JSON matching" in retry["messages"][-1]["content"]

    def test_both_modes_rejected_is_an_error(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.completions.errors = [
            api_error(openai.BadRequestError, 400, "unknown response_format")
        ] * 2
        with pytest.raises(ProviderError):
            provider.chat_json(MESSAGES, "m", {})

    def test_other_bad_requests_do_not_fall_back(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.completions.errors = [
            api_error(openai.BadRequestError, 400, "prompt is too long: 210000 tokens")
        ]
        with pytest.raises(ProviderError, match="too long"):
            provider.chat_json(MESSAGES, "m", {})
        assert len(client.completions.calls) == 1  # no second, doomed call in json_object mode

    def test_invalid_json_and_transport_errors(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.completions.json_reply = "not json"
        with pytest.raises(InvalidJsonError, match="invalid JSON"):
            provider.chat_json(MESSAGES, "m", {})
        client.completions.errors = [openai.APIConnectionError(request=REQUEST)]
        with pytest.raises(ProviderUnavailableError):
            provider.chat_json(MESSAGES, "m", {})


class TestDiscoveryAndCost:
    def test_models_and_openrouter_pricing(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.model_entries = [
            SimpleNamespace(
                id="vendor/big",
                context_length=128000,
                pricing={"prompt": "0.000003", "completion": "0.000015"},
            ),
            SimpleNamespace(id="plain", pricing=None),
            SimpleNamespace(id="odd", pricing={"prompt": "free"}),
        ]
        models = provider.list_models()
        assert [m.name for m in models] == ["vendor/big", "plain", "odd"]
        assert models[0].context_length == 128000
        assert models[1].context_length is None
        assert provider.has_price("vendor/big")
        assert not provider.has_price("odd")
        cost = provider.estimate_cost("vendor/big", Usage(1_000_000, 100_000))
        assert cost == pytest.approx(3.0 + 1.5)

    def test_configured_prices_and_unknown_models(self, provider: OpenAICompatibleProvider) -> None:
        assert provider.estimate_cost("cheap/model", Usage(500_000, 500_000)) == pytest.approx(1.5)
        assert provider.estimate_cost("mystery", Usage(10, 10)) == 0.0

    def test_discovery_errors_are_translated(
        self, provider: OpenAICompatibleProvider, client: FakeClient
    ) -> None:
        client.list_error = api_error(openai.AuthenticationError, 401)
        with pytest.raises(ProviderError, match="rejected"):
            provider.list_models()

    def test_capabilities(self, provider: OpenAICompatibleProvider) -> None:
        assert provider.capabilities("anything") == {"completion"}


class TestReplyLengthCap:
    def test_no_cap_means_no_parameter(
        self, client: FakeClient, provider: OpenAICompatibleProvider
    ) -> None:
        provider.chat_json([Message("user", "hi")], "m", {}, ChatOptions())
        sent = client.completions.calls[0]
        assert "max_tokens" not in sent
        assert "max_completion_tokens" not in sent

    def test_the_cap_is_sent_as_max_tokens(
        self, client: FakeClient, provider: OpenAICompatibleProvider
    ) -> None:
        client.completions.stream_events = [
            event("hi"),
            event(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1), choices=False),
        ]
        list(provider.stream_chat([Message("user", "hi")], "m", ChatOptions(max_tokens=300)))
        assert client.completions.calls[0]["max_tokens"] == 300

    def test_models_that_insist_on_the_newer_name_get_it(
        self, client: FakeClient, provider: OpenAICompatibleProvider
    ) -> None:
        client.completions.errors = [
            api_error(
                openai.BadRequestError,
                400,
                "Unsupported parameter: 'max_tokens' is not supported; use 'max_completion_tokens'",
            )
        ]
        result = provider.chat_json([Message("user", "hi")], "m", {}, ChatOptions(max_tokens=300))
        assert result.data == {"ok": True}
        retry = client.completions.calls[1]
        assert retry["max_completion_tokens"] == 300
        assert "max_tokens" not in retry

    def test_an_unrelated_bad_request_is_not_mistaken_for_the_name_problem(
        self, client: FakeClient, provider: OpenAICompatibleProvider
    ) -> None:
        client.completions.errors = [
            api_error(openai.BadRequestError, 400, "model does not support images"),
            api_error(openai.BadRequestError, 400, "model does not support images"),
        ]
        with pytest.raises(ProviderError):
            provider.chat_json([Message("user", "hi")], "m", {}, ChatOptions(max_tokens=300))
        assert all("max_completion_tokens" not in call for call in client.completions.calls)
