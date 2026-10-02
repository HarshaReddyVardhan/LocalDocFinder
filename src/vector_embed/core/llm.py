"""LLM gateway: picks the model for a role and owns its GPU lifecycle.

Rules from the design:
* the embedder and the chat model are never on the GPU together (the embedder is unloaded
  first; queries embed on the CPU while a chat session is active);
* the chat model loads on demand and stays for ``keep_alive`` between follow-ups;
* it is unloaded when the session closes, after ``idle_unload_seconds``, when the laptop is
  unplugged, or when a fullscreen app starts;
* a ``chat`` lock in the state DB keeps the indexing worker from starting meanwhile.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from vector_embed.core.models.catalog import ROLE_CHAT
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.power import PowerGate
from vector_embed.core.providers.base import (
    ChatChunk,
    ChatOptions,
    ChatProvider,
    JsonResult,
    Message,
    ProviderError,
    Usage,
)
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.settings import ChatSettings
from vector_embed.core.store.sqlite import CHAT_LOCK, StateDb

logger = logging.getLogger(__name__)

REASON_UNPLUGGED = "unplugged"
REASON_FULLSCREEN = "fullscreen app"
REASON_IDLE = "idle"
_LOCK_GRACE_SECONDS = 60
_ONE_SHOT_LEASE_SECONDS = 180  # a one-shot call renews this while it streams
_LEASE_REFRESH_SECONDS = 30  # renew the lock at most this often, not on every token


class ChatBlockedError(RuntimeError):
    """Chat cannot run right now (for example on battery with no cloud provider)."""


class NoChatModelError(RuntimeError):
    """No installed model can serve the role; the message says what to pull."""


@dataclass(frozen=True)
class ChatTarget:
    role: str
    model: str
    provider: ChatProvider
    local: bool


class TargetRouter(Protocol):
    """Hook for cloud routing (step 9): may replace the local target for a role."""

    def __call__(self, role: str, power: PowerGate) -> ChatTarget | None: ...


class LlmGateway:
    def __init__(
        self,
        settings: ChatSettings,
        registry: ModelRegistry,
        local: OllamaProvider,
        state: StateDb,
        power: PowerGate,
        *,
        clock: Callable[[], float] = time.monotonic,
        fullscreen: Callable[[], bool] = lambda: False,
        owner: str = "app",
        refresh_seconds: int = 3600,
    ) -> None:
        self._cfg = settings
        self._registry = registry
        self._local = local
        self._state = state
        self._power = power
        self._clock = clock
        self._fullscreen = fullscreen
        self._owner = owner
        self._refresh_seconds = refresh_seconds
        self._loaded: set[str] = set()
        self._last_activity = clock()
        self._session_models: dict[str, str] = {}  # role -> model, fixed for one session
        self._lease_ttl = 0.0  # > 0 while this gateway holds the chat lock
        self._lease_renewed = 0.0
        self.session_active = False
        self.router: TargetRouter | None = None

    # ------------------------------------------------------------------ model choice
    def target(self, role: str = ROLE_CHAT, *, local_only: bool = False) -> ChatTarget:
        """The model that will serve ``role``, or a clear error.

        ``local_only`` bypasses cloud routing (used for files that must never leave the machine).
        """
        if (
            not local_only
            and self.router is not None
            and (routed := self.router(role, self._power)) is not None
        ):
            return routed
        if not self._power.local_chat_allowed():
            raise ChatBlockedError("on battery: plug in to chat (or configure a cloud provider)")
        if self.session_active and (pinned := self._session_models.get(role)) is not None:
            return ChatTarget(role, pinned, self._local, local=True)  # no mid-chat model switch
        self._registry.refresh_if_stale(self._refresh_seconds)
        resolution = self._registry.resolve(role)
        if resolution.model is None and role != ROLE_CHAT:
            resolution = self._registry.resolve(ROLE_CHAT)
        if resolution.model is not None and self.session_active:
            self._session_models[role] = resolution.model
        if resolution.model is None:
            preferred = self._registry.preferences(role)
            hint = preferred[0] if preferred else "qwen3.5:9b"
            raise NoChatModelError(f"no usable chat model; run: ollama pull {hint}")
        return ChatTarget(role, resolution.model, self._local, local=True)

    def will_use_cloud(self, role: str = ROLE_CHAT) -> bool:
        """Whether a request for ``role`` would go to the cloud right now (never raises)."""
        try:
            return not self.target(role).local
        except (ChatBlockedError, NoChatModelError):
            return False

    def cloud_destination(self, role: str = ROLE_CHAT) -> str | None:
        """``Provider / model`` when the request would go to the cloud, else ``None``."""
        if not self.will_use_cloud(role):
            return None
        describe = getattr(self.router, "destination", None)
        destination = describe(role) if callable(describe) else None
        return str(destination) if destination is not None else None

    def estimate_cost(self, role: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimated USD for a request of this size; 0 for local models and unknown prices."""
        try:
            target = self.target(role)
        except (ChatBlockedError, NoChatModelError):
            return 0.0
        if target.local:
            return 0.0
        return target.provider.estimate_cost(target.model, Usage(prompt_tokens, completion_tokens))

    def options(self, *, session: bool) -> ChatOptions:
        """Request options: sessions keep the model warm, one-shot calls unload right after."""
        return ChatOptions(
            num_ctx=self._cfg.num_ctx,
            temperature=self._cfg.temperature,
            keep_alive=self._cfg.keep_alive if session else 0,
        )

    @property
    def query_on_cpu(self) -> bool:
        """While any session owns the GPU, embed queries on the CPU."""
        return self.session_active or self._state.lock_held(CHAT_LOCK)

    @contextmanager
    def _local_lease(self, target: "ChatTarget") -> Iterator[None]:
        """Hold the chat lock for one local call, so the indexer and the search stay off the GPU.

        A session already holds it; a cloud call does not need it.
        """
        if not target.local or self.session_active:
            yield
            return
        if not self._state.acquire_lock(CHAT_LOCK, self._owner, _ONE_SHOT_LEASE_SECONDS):
            raise ChatBlockedError("another chat session is active")
        self._lease_ttl, self._lease_renewed = _ONE_SHOT_LEASE_SECONDS, self._clock()
        try:
            yield
        finally:
            self._lease_ttl = 0.0
            self._state.release_lock(CHAT_LOCK, self._owner)

    # ------------------------------------------------------------------ session lifecycle
    def begin_chat(self) -> None:
        """Start a session: take the chat lock and clear the embedder off the GPU."""
        ttl = self._cfg.idle_unload_seconds + _LOCK_GRACE_SECONDS
        if not self._state.acquire_lock(CHAT_LOCK, self._owner, ttl):
            raise ChatBlockedError("another chat session is active")
        self.session_active = True
        self._lease_ttl, self._lease_renewed = ttl, self._clock()
        self._last_activity = self._clock()
        self._free_embedder()

    def touch(self) -> None:
        """Record activity: refreshes the idle timer and, at most every 30 s, the lock lease."""
        now = self._clock()
        self._last_activity = now
        if self._lease_ttl and now - self._lease_renewed >= _LEASE_REFRESH_SECONDS:
            self._state.acquire_lock(CHAT_LOCK, self._owner, self._lease_ttl)
            self._lease_renewed = now

    def end_chat(self, reason: str = "closed") -> None:
        """Unload every chat model this gateway loaded and release the lock."""
        logger.info("chat session ended", extra={"reason": reason})
        for model in sorted(self._loaded):
            self._local.unload(model)
        self._loaded.clear()
        self._state.release_lock(CHAT_LOCK, self._owner)
        self._lease_ttl = 0.0
        self._session_models.clear()
        self.session_active = False

    def check(self) -> str | None:
        """Poll the unload conditions; ends the session and returns the reason if one applies."""
        if not self.session_active and not self._loaded:
            return None
        reason: str | None = None
        if not self._power.local_chat_allowed() and self._loaded:
            reason = REASON_UNPLUGGED
        elif self._fullscreen():
            reason = REASON_FULLSCREEN
        elif self._clock() - self._last_activity > self._cfg.idle_unload_seconds:
            reason = REASON_IDLE
        if reason is not None:
            self.end_chat(reason)
        return reason

    # ------------------------------------------------------------------ calls
    def _free_embedder(self) -> None:
        try:
            self._local.unload_embedder()
        except ProviderError:
            logger.debug("llm: embedder unload failed", exc_info=True)

    def prewarm(self, role: str = ROLE_CHAT) -> None:
        """Start loading the model while the user is still typing."""
        target = self.target(role)
        if target.local:
            self._free_embedder()
            self._local.prewarm(target.model, self.options(session=True))
            self._loaded.add(target.model)

    def stream(
        self,
        messages: list[Message],
        role: str = ROLE_CHAT,
        *,
        session: bool = False,
        local_only: bool = False,
    ) -> Iterator[ChatChunk]:
        target = self.target(role, local_only=local_only)
        if target.local:
            self._free_embedder()
            self._loaded.add(target.model)
        options = self.options(session=session)
        try:
            with self._local_lease(target):
                for chunk in target.provider.stream_chat(messages, target.model, options):
                    self.touch()
                    yield chunk
        finally:
            if not session and target.local:
                self._loaded.discard(target.model)  # keep_alive=0 already unloaded it

    def chat_json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        role: str = ROLE_CHAT,
        *,
        session: bool = False,
        local_only: bool = False,
    ) -> JsonResult:
        target = self.target(role, local_only=local_only)
        if target.local:
            self._free_embedder()
            self._loaded.add(target.model)
        with self._local_lease(target):
            result = target.provider.chat_json(
                messages, target.model, schema, self.options(session=session)
            )
        self.touch()
        if not session and target.local:
            self._loaded.discard(target.model)
        return result
