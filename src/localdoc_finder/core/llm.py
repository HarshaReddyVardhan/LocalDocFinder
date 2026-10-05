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
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from typing import Any, Protocol

from localdoc_finder.core.models.catalog import ROLE_CHAT
from localdoc_finder.core.models.registry import ModelRegistry
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.providers.base import (
    ChatChunk,
    ChatOptions,
    ChatProvider,
    InvalidJsonError,
    JsonResult,
    Message,
    ProviderError,
    Usage,
)
from localdoc_finder.core.providers.ollama import OllamaProvider
from localdoc_finder.core.settings import ChatSettings
from localdoc_finder.core.store.sqlite import CHAT_LOCK, StateDb

logger = logging.getLogger(__name__)

REASON_UNPLUGGED = "unplugged"
REASON_FULLSCREEN = "fullscreen app"
REASON_IDLE = "idle"
_LOCK_GRACE_SECONDS = 60
_ONE_SHOT_LEASE_SECONDS = 180  # a one-shot call renews this while it streams
_LEASE_REFRESH_SECONDS = 30  # renew the lock at most this often, not on every token
_NOTICE_REASON_MAX = 120  # longest provider error shown in a fallback notice


class ChatBlockedError(RuntimeError):
    """Chat cannot run right now (for example on battery with no cloud provider)."""


class ConsentRequiredError(ChatBlockedError):
    """A cloud request was made without the user's consent. Never answered locally instead: the
    caller must ask the user, not quietly change where the question goes."""


class NoChatModelError(RuntimeError):
    """No installed model can serve the role; the message says what to pull."""


@dataclass(frozen=True)
class ChatTarget:
    role: str
    model: str
    provider: ChatProvider
    local: bool
    # A routed cloud call (not the user's "Answer better") may be answered locally if it fails.
    fallback: bool = False


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
        self._one_shot_calls = 0  # local one-shot calls sharing the chat lock right now
        # Guards the bookkeeping above: Match scores in parallel, and the UI, MCP and the idle
        # check call in from different threads. Never held while a model generates.
        self._lock = threading.RLock()
        self.session_active = False
        self.router: TargetRouter | None = None
        # Optional rewrite of what a *local* model is sent (e.g. mask IDs); cloud has its own.
        self.local_filter: Callable[[list[Message]], list[Message]] | None = None

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
        with self._lock:
            if self.session_active and (pinned := self._session_models.get(role)) is not None:
                return ChatTarget(role, pinned, self._local, local=True)  # no mid-chat switch
        self._registry.refresh_if_stale(self._refresh_seconds)  # may reach Ollama: not locked
        resolution = self._registry.resolve(role)
        if resolution.model is None and role != ROLE_CHAT:
            resolution = self._registry.resolve(ROLE_CHAT)
        if resolution.model is not None and self.session_active:
            with self._lock:  # the first resolution wins, so parallel callers agree on a model
                model = self._session_models.setdefault(role, resolution.model)
            return ChatTarget(role, model, self._local, local=True)
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
        with self._lock:
            held = target.local and not self.session_active
            if held:
                # Calls in flight share one lease: the first takes the lock and the last frees
                # it, so one call ending never releases the lock under another still running.
                if self._one_shot_calls == 0:
                    if not self._state.acquire_lock(
                        CHAT_LOCK, self._owner, _ONE_SHOT_LEASE_SECONDS
                    ):
                        raise ChatBlockedError("another chat session is active")
                    self._lease_ttl, self._lease_renewed = _ONE_SHOT_LEASE_SECONDS, self._clock()
                self._one_shot_calls += 1
        try:
            yield
        finally:
            if held:
                self._end_one_shot()

    def _end_one_shot(self) -> None:
        with self._lock:
            self._one_shot_calls -= 1
            if self._one_shot_calls == 0 and not self.session_active:
                self._lease_ttl = 0.0
                self._state.release_lock(CHAT_LOCK, self._owner)

    # ------------------------------------------------------------------ session lifecycle
    def begin_chat(self) -> None:
        """Start a session: take the chat lock and clear the embedder off the GPU."""
        ttl = self._cfg.idle_unload_seconds + _LOCK_GRACE_SECONDS
        with self._lock:
            if not self._state.acquire_lock(CHAT_LOCK, self._owner, ttl):
                raise ChatBlockedError("another chat session is active")
            self.session_active = True
            self._lease_ttl, self._lease_renewed = ttl, self._clock()
            self._last_activity = self._clock()
        self._free_embedder()

    def touch(self) -> None:
        """Record activity: refreshes the idle timer and, at most every 30 s, the lock lease."""
        with self._lock:
            now = self._clock()
            self._last_activity = now
            if self._lease_ttl and now - self._lease_renewed >= _LEASE_REFRESH_SECONDS:
                self._state.acquire_lock(CHAT_LOCK, self._owner, self._lease_ttl)
                self._lease_renewed = now

    def end_chat(self, reason: str = "closed") -> None:
        """Unload every chat model this gateway loaded and release the lock."""
        logger.info("chat session ended", extra={"reason": reason})
        with self._lock:
            loaded = sorted(self._loaded)
            self._loaded.clear()
            self._session_models.clear()
            self.session_active = False
            if self._one_shot_calls == 0:  # otherwise the last one-shot call releases it
                self._state.release_lock(CHAT_LOCK, self._owner)
                self._lease_ttl = 0.0
        for model in loaded:
            self._local.unload(model)

    def check(self) -> str | None:
        """Poll the unload conditions; ends the session and returns the reason if one applies."""
        with self._lock:
            loaded = bool(self._loaded)
            idle_for = self._clock() - self._last_activity
        if not self.session_active and not loaded:
            return None
        reason: str | None = None
        if not self._power.local_chat_allowed() and loaded:
            reason = REASON_UNPLUGGED
        elif self._fullscreen():
            reason = REASON_FULLSCREEN
        elif idle_for > self._cfg.idle_unload_seconds:
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

    def _remember_loaded(self, model: str) -> None:
        with self._lock:
            self._loaded.add(model)

    def _forget_loaded(self, model: str) -> None:
        with self._lock:
            self._loaded.discard(model)

    def _for_local(self, messages: list[Message]) -> list[Message]:
        return self.local_filter(messages) if self.local_filter is not None else messages

    def prewarm(self, role: str = ROLE_CHAT) -> None:
        """Start loading the model while the user is still typing."""
        target = self.target(role)
        if target.local:
            self._free_embedder()
            self._local.prewarm(target.model, self.options(session=True))
            self._remember_loaded(target.model)

    def stream(
        self,
        messages: list[Message],
        role: str = ROLE_CHAT,
        *,
        session: bool = False,
        local_only: bool = False,
        target: ChatTarget | None = None,
    ) -> Iterator[ChatChunk]:
        """Stream a reply. Pass ``target`` (from ``target()``) to send to exactly the model that a
        privacy decision was made for, rather than resolving the route a second time.

        A routed cloud call that fails before its first chunk is answered by the local model; the
        first chunk then carries a ``notice`` saying so.
        """
        target = target or self.target(role, local_only=local_only)
        streamed = False
        try:
            for chunk in self._stream_target(messages, target, session):
                streamed = True
                yield chunk
            return
        except (ProviderError, ChatBlockedError) as exc:
            if streamed or not self._may_fall_back(target, exc):
                raise
            local, notice = self._local_fallback(role, target, exc)
        first = True
        # closing(): cancelling the answer must end the local stream (and free the model) now
        with closing(self._stream_target(messages, local, session)) as answer:
            for chunk in answer:
                yield replace(chunk, notice=notice) if first else chunk
                first = False

    @staticmethod
    def _may_fall_back(target: ChatTarget, exc: Exception) -> bool:
        """Only a routed cloud call, and only for a failure of the cloud itself: not a missing
        consent (the user must be asked), and not unusable output (callers retry that)."""
        if target.local or not target.fallback:
            return False
        return not isinstance(exc, ConsentRequiredError | InvalidJsonError)

    def _local_fallback(
        self, role: str, target: ChatTarget, exc: Exception
    ) -> tuple[ChatTarget, str]:
        """The local target to retry on, and the notice for the user. When no local model can
        answer either (on battery, nothing installed) the cloud's own error is the one to show.
        The local route is strictly more private, so no new privacy decision is needed."""
        try:
            local = self.target(role, local_only=True)
        except (ChatBlockedError, NoChatModelError) as local_error:
            raise exc from local_error
        reason = " ".join(str(exc).split())
        if len(reason) > _NOTICE_REASON_MAX:
            reason = reason[: _NOTICE_REASON_MAX - 1] + "…"
        logger.info(
            "cloud call failed; answering locally",
            extra={"provider": target.provider.name, "error": type(exc).__name__},
        )
        return local, f"{reason}; answered locally with {local.model}"

    def _stream_target(
        self, messages: list[Message], target: ChatTarget, session: bool
    ) -> Generator[ChatChunk]:
        if target.local:
            self._free_embedder()
            self._remember_loaded(target.model)
            messages = self._for_local(messages)
        options = self.options(session=session)
        try:
            with self._local_lease(target):
                for chunk in target.provider.stream_chat(messages, target.model, options):
                    self.touch()
                    yield chunk
        finally:
            if not session and target.local:
                self._forget_loaded(target.model)  # keep_alive=0 already unloaded it

    def chat_json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        role: str = ROLE_CHAT,
        *,
        session: bool = False,
        local_only: bool = False,
        target: ChatTarget | None = None,
    ) -> JsonResult:
        """One JSON answer; a failed routed cloud call is answered locally (see ``stream``)."""
        target = target or self.target(role, local_only=local_only)
        try:
            return self._json_target(messages, schema, target, session)
        except (ProviderError, ChatBlockedError) as exc:
            if not self._may_fall_back(target, exc):
                raise
            local, notice = self._local_fallback(role, target, exc)
        return replace(self._json_target(messages, schema, local, session), notice=notice)

    def _json_target(
        self, messages: list[Message], schema: dict[str, Any], target: ChatTarget, session: bool
    ) -> JsonResult:
        if target.local:
            self._free_embedder()
            self._remember_loaded(target.model)
            messages = self._for_local(messages)
        with self._local_lease(target):
            result = target.provider.chat_json(
                messages, target.model, schema, self.options(session=session)
            )
        self.touch()
        if not session and target.local:
            self._forget_loaded(target.model)
        return result
