"""Cloud access: consent, budget, privacy filtering, usage tracking and routing.

``CloudChatProvider`` wraps a real provider so that *every* cloud request:
  1. needs the user's per-session consent,
  2. is refused once the monthly budget is spent,
  3. goes through the privacy filter (IDs masked, optional personal-detail redaction),
  4. is recorded (tokens and estimated cost) in the state database.
``CloudRouter`` decides per role whether a request goes local or to the cloud.
"""

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from localdoc_finder.core.health import month_start
from localdoc_finder.core.llm import ChatBlockedError, ChatTarget
from localdoc_finder.core.models.catalog import ROLE_CHAT
from localdoc_finder.core.models.registry import ModelRegistry
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.privacy.policy import Outbound, PrivacyFilter
from localdoc_finder.core.providers.base import (
    ChatChunk,
    ChatOptions,
    ChatProvider,
    JsonResult,
    Message,
    ModelInfo,
    ProviderError,
    Usage,
)
from localdoc_finder.core.settings import CloudSettings
from localdoc_finder.core.store.sqlite import StateDb
from localdoc_finder.core.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_PLACEHOLDER_MAX = 24  # longest placeholder, e.g. "[ADDRESS_12]", plus slack
_PRICE_REFRESH_SECONDS = 3600  # how often an unpriced model prompts a look at the price list
_CHARS_PER_TOKEN = 4  # rough size of a streamed reply when the provider reports no usage


class CloudConsent:
    """Opt-in. Nothing is sent to the cloud until the user grants it.

    Two grants exist. The *request* grant covers one request and is withdrawn when it ends. The
    *session* grant is the user's explicit "don't ask again until LocalDoc Finder restarts"; it
    lives as long as this object (the app keeps one for its whole run) or until
    ``revoke_session``.
    """

    def __init__(self) -> None:
        self._request = False
        self._session = False

    @property
    def granted(self) -> bool:
        return self._request or self._session

    @property
    def session_granted(self) -> bool:
        return self._session

    def grant(self) -> None:
        self._request = True

    def grant_session(self) -> None:
        self._session = True

    def revoke(self) -> None:
        """End the per-request consent; a session consent the user chose stays."""
        self._request = False

    def revoke_session(self) -> None:
        self._request = self._session = False


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
        self._budget_lock = threading.Lock()
        self._reserved = 0.0  # worst-case cost of the calls in flight
        self._prices_asked: float = -_PRICE_REFRESH_SECONDS  # so the first need asks at once

    # ------------------------------------------------------------------ guards
    def _guard(self) -> None:
        if not self._consent.granted:
            raise ChatBlockedError(
                "cloud requests need your consent for this session (command line: pass --cloud-ok)"
            )

    def _has_price(self, model: str) -> bool:
        """Whether ``model``'s cost is known (a provider with no price list counts as priced)."""
        has_price = getattr(self._inner, "has_price", None)
        return True if has_price is None else bool(has_price(model))

    def _learn_prices(self, model: str) -> None:
        """Ask the provider for its price list when ``model`` has no price: at first use, then at
        most hourly. Without prices every call costs "$0" and no budget could ever be reached."""
        if self._has_price(model):
            return
        now = self._clock()
        if now - self._prices_asked < _PRICE_REFRESH_SECONDS:
            return
        self._prices_asked = now
        try:
            self._inner.list_models()
        except ProviderError:
            logger.info("cloud: could not fetch %s's price list", self.name)

    def _reserve(self, model: str, messages: list[Message], options: ChatOptions) -> float:
        """Consent, price and budget checks, then set aside the worst-case cost of this call.

        Reserved money counts against the budget immediately, so scoring several documents at the
        same time cannot each pass the check and together overspend.
        """
        self._guard()
        self._learn_prices(model)
        budget = self._settings.monthly_budget_usd
        if budget is None:
            if not self._has_price(model):
                logger.warning("cloud: no price known for %s; its cost is not tracked", model)
            return 0.0
        if not self._has_price(model):
            raise ChatBlockedError(
                f"the price of {model} is unknown, so the monthly budget cannot be enforced; set "
                f"it under cloud.providers.{self.name}.pricing or turn the budget off"
            )
        prompt = sum(estimate_tokens(m.content) for m in messages)
        worst = self._inner.estimate_cost(model, Usage(prompt, options.max_tokens or 0))
        with self._budget_lock:
            spent = self._state.spend_since(month_start(self._clock()))
            if spent + self._reserved + worst > budget:
                raise ChatBlockedError(
                    f"the monthly cloud budget (${budget:.2f}) would be exceeded "
                    f"(${spent:.2f} spent, this request could cost up to ${worst:.2f})"
                )
            self._reserved += worst
        return worst

    def _release(self, amount: float) -> None:
        with self._budget_lock:
            self._reserved = max(0.0, self._reserved - amount)

    def _capped(self, options: ChatOptions | None) -> ChatOptions:
        return replace(options or ChatOptions(), max_tokens=self._settings.max_output_tokens)

    def prepare(self, messages: list[Message]) -> Outbound:
        """What would be sent (after masking). Used for the badge and 'view what will be sent'."""
        return self._privacy.prepare(messages)

    def _settle(self, model: str, usage: Usage, reserved: float) -> None:
        """Record what was used and free the reservation. Called from ``finally``: a failed or
        abandoned call may still have been billed, and the reservation must never leak."""
        try:
            self._record(model, usage)
        finally:
            self._release(reserved)

    def _record(self, model: str, usage: Usage) -> None:
        cost = self._inner.estimate_cost(model, usage)
        self._state.record_usage(
            self.name, model, usage.prompt_tokens, usage.completion_tokens, cost
        )

    # ------------------------------------------------------------------ ChatProvider
    def stream_chat(
        self, messages: list[Message], model: str, options: ChatOptions | None = None
    ) -> Iterator[ChatChunk]:
        outbound = self.prepare(messages)  # per call: concurrent requests never share it
        capped = self._capped(options)
        reserved = self._reserve(model, outbound.messages, capped)
        restorer = _Restorer(outbound)
        reported = Usage()
        streamed_chars = 0
        try:
            for chunk in self._inner.stream_chat(outbound.messages, model, capped):
                if chunk.usage is not None:
                    reported = chunk.usage
                streamed_chars += len(chunk.text)
                text = restorer.feed(chunk.text) if chunk.text else ""
                if text:
                    yield ChatChunk(text)
            tail = restorer.flush()
            yield ChatChunk(tail, reported)
        finally:
            used = reported
            if used.prompt_tokens + used.completion_tokens == 0:  # usage never reported
                prompt = sum(estimate_tokens(m.content) for m in outbound.messages)
                used = Usage(prompt, max(1, streamed_chars // _CHARS_PER_TOKEN))
            self._settle(model, used, reserved)

    def chat_json(
        self,
        messages: list[Message],
        model: str,
        schema: dict[str, Any],
        options: ChatOptions | None = None,
    ) -> JsonResult:
        outbound = self.prepare(messages)  # per call: concurrent requests never share it
        capped = self._capped(options)
        reserved = self._reserve(model, outbound.messages, capped)
        usage = Usage()
        try:
            result = self._inner.chat_json(outbound.messages, model, schema, capped)
            usage = result.usage
        finally:
            self._settle(model, usage, reserved)
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
        self._escalations: Counter[int] = Counter()  # thread id -> open "Answer better" scopes
        self._escalations_lock = threading.Lock()

    @contextmanager
    def escalated(self) -> Iterator[None]:
        """Route this thread's requests to the cloud ("Answer better") until the block ends.

        Per thread, not global: a Match run or an MCP call on another thread is never sent to the
        cloud because the user escalated one answer. The thread is captured on entry, so the
        scope still ends correctly when a generator holding it is closed from another thread.
        """
        thread = threading.get_ident()
        with self._escalations_lock:
            self._escalations[thread] += 1
        try:
            yield
        finally:
            with self._escalations_lock:
                self._escalations[thread] -= 1
                if self._escalations[thread] <= 0:
                    del self._escalations[thread]

    @property
    def escalate(self) -> bool:
        """Whether the calling thread is inside an ``escalated()`` block."""
        with self._escalations_lock:
            return threading.get_ident() in self._escalations

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
