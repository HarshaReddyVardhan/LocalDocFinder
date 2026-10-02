"""Qt-free service behind the Ask and Chat modes: streams events, owns the model lifecycle."""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from vector_embed.core.documents import DocumentError
from vector_embed.core.llm import ChatBlockedError, LlmGateway, NoChatModelError
from vector_embed.core.models.catalog import ROLE_CHAT
from vector_embed.core.providers.base import Message, ProviderError
from vector_embed.core.rag import Source, build_messages
from vector_embed.core.runtime import CloudContext
from vector_embed.core.skills.ask import AskSkill, gateway_of, privacy_of
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.chat import ChatInput, ChatSkill

logger = logging.getLogger(__name__)

_KNOWN_ERRORS = (ChatBlockedError, NoChatModelError, ProviderError, DocumentError, RuntimeError)


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass(frozen=True)
class Finished:
    sources: list[Source] = field(default_factory=list)
    note: str = ""  # footer text (citations, truncation notices)
    session_id: int | None = None


@dataclass(frozen=True)
class Failed:
    message: str


Event = Delta | Finished | Failed


@dataclass(frozen=True)
class CloudPreview:
    """What "Answer better" would send, shown to the user before they agree."""

    destination: str  # "OpenRouter / model"
    badge: str  # "☁ Sending 6 excerpts (≈3.1k tokens) to OpenRouter / model"
    shield: str  # "🛡 2 sensitive items will be masked: ..." or ""
    text: str  # the exact text that will be sent, after masking


@dataclass
class ChatState:
    """What the window knows about the conversation in progress."""

    session_id: int | None = None
    pinned: list[str] = field(default_factory=list)
    scratch: str = ""

    def reset(self) -> None:
        self.session_id, self.pinned, self.scratch = None, [], ""

    def describe(self) -> str:
        parts = [f"{len(self.pinned)} file(s) pinned"] if self.pinned else []
        if self.scratch:
            parts.append(f"pasted text ({len(self.scratch)} chars)")
        return ", ".join(parts)


class AssistantService:
    """Builds the skills lazily and turns their output into UI events."""

    def __init__(self, context_factory: Callable[[], SkillContext]) -> None:
        self._factory = context_factory
        self._ctx: SkillContext | None = None

    @property
    def ctx(self) -> SkillContext:
        if self._ctx is None:
            self._ctx = self._factory()
        return self._ctx

    @property
    def gateway(self) -> LlmGateway:
        return gateway_of(self.ctx)

    @property
    def session_active(self) -> bool:
        return self.gateway.session_active

    # ------------------------------------------------------------------ lifecycle
    def begin_chat(self) -> None:
        """Enter chat mode: take the chat lock, free the GPU, start loading the model."""
        gateway = self.gateway
        if not gateway.session_active:
            gateway.begin_chat()
        gateway.prewarm()

    def end_chat(self, reason: str = "closed") -> None:
        if self._ctx is not None and self.gateway.session_active:
            self.gateway.end_chat(reason)
        privacy = privacy_of(self._ctx) if self._ctx is not None else None
        if privacy is not None:
            privacy.forget_names()  # the next chat must not inherit this one's masked names

    def maintain(self) -> str | None:
        """Poll idle/unplug/fullscreen conditions; returns why the model was unloaded."""
        return self.gateway.check() if self._ctx is not None else None

    # ------------------------------------------------------------------ cloud ("Answer better")
    def _cloud(self) -> CloudContext | None:
        cloud = self.ctx.extras.get("cloud")
        if not isinstance(cloud, CloudContext) or cloud.provider is None:
            return None
        return cloud if cloud.router.destination(ROLE_CHAT) is not None else None

    def cloud_available(self) -> bool:
        """True when a cloud provider with a stored key and a chat model is configured."""
        return self._cloud() is not None

    def _preview(self, cloud: CloudContext, messages: list[Message], excerpts: int) -> CloudPreview:
        assert cloud.provider is not None
        destination = str(cloud.router.destination(ROLE_CHAT))
        outbound = cloud.provider.prepare(messages)
        privacy = cloud.privacy
        return CloudPreview(
            destination,
            privacy.badge(outbound, destination, excerpts),
            privacy.shield_note(outbound),
            privacy.preview(outbound),
        )

    def cloud_preview_ask(self, question: str) -> CloudPreview | None:
        """The request an escalated Ask would send; ``None`` if no cloud is configured."""
        cloud = self._cloud()
        if cloud is None:
            return None
        cloud.router.escalate = True  # so private files are filtered as for a cloud request
        try:
            run = AskSkill(self.ctx).prepare(question, session=self.session_active)
        finally:
            cloud.router.reset()
        return self._preview(
            cloud, build_messages(question, run.result.sources), len(run.result.sources)
        )

    def cloud_preview_chat(self, message: str, state: ChatState) -> CloudPreview | None:
        """The request an escalated chat turn would send (opens the session if needed)."""
        cloud = self._cloud()
        if cloud is None:
            return None
        skill = ChatSkill(self.ctx)
        if state.session_id is None:
            state.session_id = skill.open_session(message, state.pinned, state.scratch or None)
        cloud.router.escalate = True
        try:
            messages, _ = skill.build_prompt(state.session_id, message)
        finally:
            cloud.router.reset()
        return self._preview(cloud, messages, len(state.pinned) + (1 if state.scratch else 0) or 1)

    def escalated(self, events: Iterator[Event]) -> Iterator[Event]:
        """Run ``events`` with the cloud for this one request, after the user's consent."""
        cloud = self._cloud()
        if cloud is None:
            yield Failed("no cloud provider is configured")
            return
        cloud.consent.grant()
        cloud.router.escalate = True
        try:
            yield from events
        finally:
            cloud.router.reset()

    # ------------------------------------------------------------------ streaming
    def ask(self, question: str) -> Iterator[Event]:
        try:
            run = AskSkill(self.ctx).prepare(question, session=self.session_active)
            for text in run.deltas():
                yield Delta(text)
            result = run.result
            ordered = result.cited + [s for s in result.sources if s not in result.cited]
            yield Finished(ordered, run.footer())
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))

    def chat(self, message: str, state: ChatState) -> Iterator[Event]:
        try:
            skill = ChatSkill(self.ctx)
            params = ChatInput(
                message=message,
                session=state.session_id,
                pin=[] if state.session_id else state.pinned,
                scratch=None if state.session_id else state.scratch or None,
            )
            turn, deltas = skill.turn(params)
            state.session_id = turn.session_id
            for text in deltas:
                yield Delta(text)
            note = (
                f"(cut to fit the context: {', '.join(turn.truncated)})" if turn.truncated else ""
            )
            yield Finished(note=note, session_id=turn.session_id)
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))
