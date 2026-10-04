"""One adapter for every OpenAI-compatible API: OpenAI, OpenRouter, LM Studio, vLLM, Groq, ...

Only ``base_url`` and an API key differ (OpenRouter: ``https://openrouter.ai/api/v1``). Keys
come from the credential store and are scrubbed from every error message.
"""

import json
import logging
from collections.abc import Iterator
from typing import Any, TypeAlias

import openai
from openai import OpenAI

from localdoc_finder.core.providers.base import (
    CAP_COMPLETION,
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
from localdoc_finder.core.secrets import scrub
from localdoc_finder.core.settings import CloudProviderSettings

logger = logging.getLogger(__name__)

ClientLike: TypeAlias = Any  # openai.OpenAI or a test double
_PER_MILLION = 1_000_000


_RESPONSE_FORMAT_WORDS = ("response_format", "json_schema", "json_object", "structured output")


def _rejects_response_format(exc: openai.BadRequestError) -> bool:
    """Whether a 400 is the endpoint refusing the requested JSON mode, the one case where the
    next, plainer JSON mode is worth trying. Any other 400 (bad model, prompt too long) is not."""
    message = str(exc).lower()
    return any(word in message for word in _RESPONSE_FORMAT_WORDS)


class OpenAICompatibleProvider:
    """``ChatProvider`` for any endpoint that speaks the OpenAI chat-completions protocol."""

    def __init__(
        self,
        name: str,
        settings: CloudProviderSettings,
        api_key: str,
        client: ClientLike | None = None,
    ) -> None:
        self.name = name
        self.label = settings.label or name
        self._settings = settings
        self._key = api_key
        self._client: ClientLike = client or OpenAI(base_url=settings.base_url, api_key=api_key)
        # (input, output) USD per million tokens; seeded from settings, extended by discovery
        self._pricing: dict[str, tuple[float, float]] = dict(settings.pricing)

    # ------------------------------------------------------------------ helpers
    def _translate(self, exc: Exception) -> ProviderError:
        text = scrub(str(exc), self._key)
        if isinstance(exc, openai.AuthenticationError):
            return ProviderError(f"{self.label}: the API key was rejected")
        if isinstance(exc, openai.NotFoundError):
            return ModelNotFoundError(f"{self.label}: {text}")
        if isinstance(
            exc, openai.RateLimitError | openai.APIConnectionError | openai.APITimeoutError
        ):
            return ProviderUnavailableError(f"{self.label} unavailable: {text}")
        return ProviderError(f"{self.label}: {text}")

    @staticmethod
    def _wire(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _usage(raw: Any) -> Usage:  # noqa: ANN401
        if raw is None:
            return Usage()
        return Usage(int(raw.prompt_tokens or 0), int(raw.completion_tokens or 0))

    def _create(self, opts: ChatOptions, **request: Any) -> Any:  # noqa: ANN401
        """``chat.completions.create`` with the reply-length cap in the spelling the API wants.

        Most endpoints take ``max_tokens``; OpenAI's newer models insist on
        ``max_completion_tokens`` and answer 400 to the old name, so that is tried next.
        """
        if opts.max_tokens is None:
            return self._client.chat.completions.create(**request)
        try:
            return self._client.chat.completions.create(max_tokens=opts.max_tokens, **request)
        except openai.BadRequestError as exc:
            if "max_tokens" not in str(exc) and "max_completion_tokens" not in str(exc):
                raise
            return self._client.chat.completions.create(
                max_completion_tokens=opts.max_tokens, **request
            )

    # ------------------------------------------------------------------ chat
    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]:
        opts = options or ChatOptions()
        try:
            stream = self._create(
                opts,
                model=model,
                messages=self._wire(messages),
                temperature=opts.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            usage = Usage()
            for event in stream:
                if event.usage is not None:
                    usage = self._usage(event.usage)
                if event.choices:
                    text = event.choices[0].delta.content
                    if text:
                        yield ChatChunk(text)
            yield ChatChunk("", usage)
        except (openai.OpenAIError, OSError) as exc:
            raise self._translate(exc) from exc

    def chat_json(
        self,
        messages: list[Message],
        model: str,
        schema: dict[str, Any],
        options: ChatOptions | None = None,
    ) -> JsonResult:
        opts = options or ChatOptions()
        formats: list[dict[str, Any]] = [
            {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}},
            {"type": "json_object"},
        ]
        wire = self._wire(messages)
        last: Exception | None = None
        for index, response_format in enumerate(formats):
            payload = wire
            if index == 1:  # json_object mode needs the schema spelled out in the prompt
                payload = [
                    *wire,
                    {"role": "user", "content": "Reply with JSON matching: " + json.dumps(schema)},
                ]
            try:
                response = self._create(
                    opts,
                    model=model,
                    messages=payload,
                    temperature=opts.temperature,
                    response_format=response_format,
                )
            except openai.BadRequestError as exc:
                if not _rejects_response_format(exc):  # a real error, not a missing JSON mode
                    raise self._translate(exc) from exc
                last = exc
                continue
            except (openai.OpenAIError, OSError) as exc:
                raise self._translate(exc) from exc
            try:
                data = json.loads(response.choices[0].message.content)
            except (TypeError, json.JSONDecodeError) as exc:
                raise InvalidJsonError(f"{self.label}: model returned invalid JSON: {exc}") from exc
            return JsonResult(data, self._usage(response.usage))
        raise self._translate(last or RuntimeError("no JSON mode accepted"))

    # ------------------------------------------------------------------ discovery and cost
    def list_models(self) -> list[ModelInfo]:
        try:
            listing = self._client.models.list()
        except (openai.OpenAIError, OSError) as exc:
            raise self._translate(exc) from exc
        models: list[ModelInfo] = []
        for entry in listing.data:
            self._remember_pricing(entry)
            models.append(
                ModelInfo(
                    name=entry.id,
                    provider=self.name,
                    context_length=getattr(entry, "context_length", None),
                    capabilities=frozenset({CAP_COMPLETION}),
                )
            )
        return models

    def _remember_pricing(self, entry: Any) -> None:  # noqa: ANN401
        """OpenRouter's catalogue lists USD per token; convert to per million."""
        pricing = getattr(entry, "pricing", None)
        if not isinstance(pricing, dict):
            return
        try:
            self._pricing.setdefault(
                entry.id,
                (
                    float(pricing["prompt"]) * _PER_MILLION,
                    float(pricing["completion"]) * _PER_MILLION,
                ),
            )
        except (KeyError, TypeError, ValueError):
            logger.debug("openai_compat: unusable pricing for %s", entry.id)

    def capabilities(self, model: str) -> frozenset[str]:
        return frozenset({CAP_COMPLETION})

    def estimate_cost(self, model: str, usage: Usage) -> float:
        """USD for ``usage`` at the model's price; 0 when the price is unknown."""
        price = self._pricing.get(model)
        if price is None:
            return 0.0
        return (usage.prompt_tokens * price[0] + usage.completion_tokens * price[1]) / _PER_MILLION

    def has_price(self, model: str) -> bool:
        return model in self._pricing
