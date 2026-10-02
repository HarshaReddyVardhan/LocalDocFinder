"""Cloud access: consent, budget, privacy filtering, usage tracking and routing.

``CloudChatProvider`` wraps a real provider so that *every* cloud request:
  1. needs the user's per-session consent,
  2. is refused once the monthly budget is spent,
  3. goes through the privacy filter (IDs masked, optional personal-detail redaction),
  4. is recorded (tokens and estimated cost) in the state database.
``CloudRouter`` decides per role whether a request goes local or to the cloud.
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from vector_embed.core.health import month_start
from vector_embed.core.llm import ChatBlockedError, ChatTarget
from vector_embed.core.models.catalog import ROLE_CHAT
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.power import PowerGate
from vector_embed.core.privacy.policy import Outbound, PrivacyFilter
from vector_embed.core.providers.base import (
    ChatChunk,
    ChatOptions,
    ChatProvider,
    JsonResult,
    Message,
    ModelInfo,
    Usage,
)
from vector_embed.core.settings import CloudSettings
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

_PLACEHOLDER_MAX = 24  # longest placeholder, e.g. "[ADDRESS_12]", plus slack


class CloudConsent:
    """Per-session opt-in. Nothing is sent to the cloud until the user grants it."""

    def __init__(self) -> None:
        self.granted = False

    def grant(self) -> None:
        self.granted = True

    def revoke(self) -> None:
        self.granted = False


def _restore_strings(data: Any, restore: Callable[[str], str]) -> Any:  # noqa: ANN401
    if isinstance(data, str):
        return restore(data)
    if isinstance(data, list):
        return [_restore_strings(item, restore) for item in data]
    if isinstance(data, dict):
        return {k: _restore_strings(v, restore) for k, v in data.items()}
    return data


class _Restorer:
    """Swaps placeholders back into streamed text, holding back a possibly split ``[...]``."""

    def __init__(self, outbound: Outbound) -> None:
        self._outbound = outbound
        self._pending = ""

    def feed(self, text: str) -> str:
        if not self._outbound.placeholders:
            return text
        buffer = self._pending + text
        cut = buffer.rfind("[")
        if cut != -1 and "]" not in buffer[cut:] and len(buffer) - cut < _PLACEHOLDER_MAX:
            self._pending, buffer = buffer[cut:], buffer[:cut]
        else:
            self._pending = ""
        return self._outbound.restore(buffer)

    def flush(self) -> str:
        rest, self._pending = self._pending, ""
        return self._outbound.restore(rest)


class CloudChatProvider:
    """A cloud ``ChatProvider`` that enforces consent, budget and privacy on every call."""

    def __init__(
        self,
        inner: ChatProvider,
        privacy: PrivacyFilter,
        state: StateDb,
        settings: CloudSettings,
        consent: CloudConsent,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._inner = inner
        self._privacy = privacy
        self._state = state
        self._settings = settings
        self._consent = consent
        self._clock = clock
        self.name = inner.name
        self.label: str = getattr(inner, "label", inner.name)
        self.last_outbound: Outbound | None = None

    # ------------------------------------------------------------------ guards
    def _guard(self) -> None:
        if not self._consent.granted:
            raise ChatBlockedError("cloud requests need your consent for this session")
        budget = self._settings.monthly_budget_usd
        if budget is not None and self._state.spend_since(month_start(self._clock())) >= budget:
            raise ChatBlockedError(f"the monthly cloud budget (${budget:.2f}) is used up")

    def prepare(self, messages: list[Message]) -> Outbound:
        """What would be sent (after masking). Used for the badge and 'view what will be sent'."""
        return self._privacy.prepare(messages)

    def _record(self, model: str, usage: Usage) -> None:
        cost = self._inner.estimate_cost(model, usage)
        self._state.record_usage(
            self.name, model, usage.prompt_tokens, usage.completion_tokens, cost
        )

    # ------------------------------------------------------------------ ChatProvider
    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]:
        self._guard()
        outbound = self.prepare(messages)
        self.last_outbound = outbound
        restorer = _Restorer(outbound)
        usage = Usage()
        for chunk in self._inner.stream_chat(outbound.messages, model, options):
            if chunk.usage is not None:
                usage = chunk.usage
            text = restorer.feed(chunk.text) if chunk.text else ""
            if text:
                yield ChatChunk(text)
        tail = restorer.flush()
        self._record(model, usage)
        yield ChatChunk(tail, usage)

    def chat_json(
        self,
        messages: list[Message],
        model: str,
        schema: dict[str, Any],
        options: ChatOptions | None = None,
    ) -> JsonResult:
        self._guard()
        outbound = self.prepare(messages)
        self.last_outbound = outbound
        result = self._inner.chat_json(outbound.messages, model, schema, options)
        self._record(model, result.usage)
        return JsonResult(_restore_strings(result.data, outbound.restore), result.usage)

    def list_models(self) -> list[ModelInfo]:
        return self._inner.list_models()

    def capabilities(self, model: str) -> frozenset[str]:
        return self._inner.capabilities(model)

    def estimate_cost(self, model: str, usage: Usage) -> float:
        return self._inner.estimate_cost(model, usage)


@dataclass(frozen=True)
class CloudDestination:
    """Where a cloud request would go, for the footer and the privacy badge."""

    label: str
    model: str

    def __str__(self) -> str:
        return f"{self.label} / {self.model}"


class CloudRouter:
    """Routing policy per role: ``local``, ``cloud`` or ``auto`` (local first).

    ``auto`` uses the cloud when the user asked for "Answer better", when no local model fits
    the machine, or when on battery (the cloud uses no local GPU or battery).
    """

    def __init__(
        self,
        settings: CloudSettings,
        provider: CloudChatProvider | None,
        registry: ModelRegistry,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._registry = registry
        self.escalate = False  # set by "Answer better" for one request; clear with ``reset``

    def reset(self) -> None:
        """End a one-request escalation (call after the answer is complete)."""
        self.escalate = False

    def model_for(self, role: str) -> str | None:
        active = self._settings.active
        if self._provider is None or active is None or active not in self._settings.providers:
            return None
        models = self._settings.providers[active].models
        return models.get(role) or models.get(ROLE_CHAT)

    def destination(self, role: str) -> CloudDestination | None:
        model = self.model_for(role)
        if model is None or self._provider is None:
            return None
        return CloudDestination(self._provider.label, model)

    def __call__(self, role: str, power: PowerGate) -> ChatTarget | None:
        wanted = self._settings.policy(role)
        escalate = self.escalate
        model = self.model_for(role)
        if wanted == "local" and not escalate:
            return None
        use_cloud = wanted == "cloud" or escalate
        if wanted == "auto" and not use_cloud:
            self._registry.refresh_if_stale()
            use_cloud = not power.local_chat_allowed() or self._registry.resolve(role).model is None
        if not use_cloud:
            return None
        if model is None or self._provider is None:
            if wanted == "cloud" or escalate:
                raise ChatBlockedError("cloud is selected but no provider/model/key is configured")
            return None
        return ChatTarget(role, model, self._provider, local=False)
