"""Qt-free service behind the Ask and Chat modes: streams events, owns the model lifecycle."""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from pydantic import ValidationError

from vector_embed.core.documents import DocumentError
from vector_embed.core.llm import ChatBlockedError, LlmGateway, NoChatModelError
from vector_embed.core.models.catalog import ROLE_CHAT
from vector_embed.core.providers.base import Message, ProviderError
from vector_embed.core.rag import Source
from vector_embed.core.registry import RegistryError
from vector_embed.core.runtime import CloudContext
from vector_embed.core.skills.ask import AskRun, AskSkill, gateway_of, privacy_of
from vector_embed.core.skills.base import SkillContext, create_skill
from vector_embed.core.skills.chat import ChatInput, ChatSkill, PreparedTurn, SessionSummary

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
class _Previewed:
    """The request the user is looking at; sending it must use this object, not a rebuild."""

    kind: str  # "ask" or "chat"
    key: str  # the question or message it was prepared for
    payload: object


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
        self._previewed: _Previewed | None = None

    def reset(self) -> None:
        """Drop the cached context (a setting changed); the next request rebuilds it."""
        self._ctx = None
        self._previewed = None

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
        self.revoke_consent()
        privacy = privacy_of(self._ctx) if self._ctx is not None else None
        if privacy is not None:
            privacy.forget_names()  # the next chat must not inherit this one's masked names

    def revoke_consent(self) -> None:
        """Withdraw cloud consent (the window was hidden, the chat ended, a new match started)."""
        cloud = self.ctx.extras.get("cloud") if self._ctx is not None else None
        if isinstance(cloud, CloudContext):
            cloud.consent.revoke()

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
        """The request an escalated Ask would send; ``None`` if no cloud is configured.

        The prepared answer is kept, and ``ask_escalated`` sends that same object, so what the
        user inspected is exactly what leaves.
        """
        cloud = self._cloud()
        if cloud is None:
            return None
        run = self._prepare_cloud_ask(cloud, question)
        self._previewed = _Previewed("ask", question, run)
        return self._preview(cloud, run.messages, len(run.result.sources))

    def cloud_preview_chat(self, message: str, state: ChatState) -> CloudPreview | None:
        """The request an escalated chat turn would send (opens the session if needed)."""
        cloud = self._cloud()
        if cloud is None:
            return None
        prepared = self._prepare_cloud_turn(cloud, message, state)
        self._previewed = _Previewed("chat", message, prepared)
        excerpts = len(state.pinned) + (1 if state.scratch else 0) or 1
        return self._preview(cloud, prepared.messages, excerpts)

    def _prepare_cloud_ask(self, cloud: CloudContext, question: str) -> AskRun:
        # escalated so private files are filtered as for a cloud request; the prepared run keeps
        # its cloud route, nothing else does
        with cloud.router.escalated():
            return AskSkill(self.ctx).prepare(question, session=self.session_active)

    def _prepare_cloud_turn(
        self, cloud: CloudContext, message: str, state: ChatState
    ) -> PreparedTurn:
        skill = ChatSkill(self.ctx)
        params = ChatInput(
            message=message,
            session=state.session_id,
            pin=[] if state.session_id else state.pinned,
            scratch=None if state.session_id else state.scratch or None,
        )
        with cloud.router.escalated():
            prepared = skill.prepare_turn(params)
        state.session_id = prepared.session_id
        return prepared

    def _take_previewed(self, kind: str, key: str) -> object | None:
        previewed, self._previewed = self._previewed, None
        if previewed is not None and (previewed.kind, previewed.key) == (kind, key):
            return previewed.payload
        return None

    def ask_escalated(self, question: str) -> Iterator[Event]:
        """Answer in the cloud with exactly the request the user previewed (or prepare it now)."""
        cloud = self._cloud()
        if cloud is None:
            yield Failed("no cloud provider is configured")
            return
        previewed = self._take_previewed("ask", question)
        try:
            run = (
                previewed
                if isinstance(previewed, AskRun)
                else self._prepare_cloud_ask(cloud, question)
            )
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))
            return
        yield from self.escalated(self._ask_events(run))

    def chat_escalated(self, message: str, state: ChatState) -> Iterator[Event]:
        cloud = self._cloud()
        if cloud is None:
            yield Failed("no cloud provider is configured")
            return
        previewed = self._take_previewed("chat", message)
        try:
            prepared = (
                previewed
                if isinstance(previewed, PreparedTurn)
                else self._prepare_cloud_turn(cloud, message, state)
            )
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))
            return
        yield from self.escalated(self._turn_events(prepared, state))

    def escalated(self, events: Iterator[Event]) -> Iterator[Event]:
        """Run ``events`` with the cloud for this one request, after the user's consent."""
        cloud = self._cloud()
        if cloud is None:
            yield Failed("no cloud provider is configured")
            return
        cloud.consent.grant()
        try:
            with cloud.router.escalated():
                yield from events
        finally:
            cloud.consent.revoke()  # consent covers this one request, never the next

    # ------------------------------------------------------------------ streaming
    def ask(self, question: str) -> Iterator[Event]:
        try:
            run = AskSkill(self.ctx).prepare(question, session=self.session_active)
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))
            return
        yield from self._ask_events(run)

    def _ask_events(self, run: AskRun) -> Iterator[Event]:
        try:
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
            prepared = skill.prepare_turn(params)
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))
            return
        yield from self._turn_events(prepared, state)

    def pin(self, paths: list[str], state: ChatState) -> None:
        """Add files to the conversation: to the running session, or to the next one."""
        if state.session_id is not None:
            state.pinned = ChatSkill(self.ctx).pin(state.session_id, paths)
        else:
            state.pinned = list(dict.fromkeys([*state.pinned, *paths]))

    def recent_sessions(self) -> list[SessionSummary]:
        return ChatSkill(self.ctx).recent_sessions()

    def reopen(self, session_id: int, state: ChatState) -> str:
        """Continue an earlier conversation: its pinned files come back, and the conversation so
        far is returned as text to show. The next message is sent within that session."""
        store = self.ctx.state
        context = store.session_context(session_id)
        if not context and not store.messages(session_id):
            raise RuntimeError(f"no chat session {session_id}")
        state.session_id = session_id
        state.pinned = [str(p) for p in context.get("pinned", [])]
        state.scratch = str(context.get("scratch", ""))
        return "".join(
            f"**You:** {m.content}\n\n" if m.role == "user" else f"{m.content}\n\n"
            for m in store.messages(session_id)
        )

    def run_skill(self, name: str, text: str) -> Iterator[Event]:
        """Run any panel skill from the registry with ``text`` as its main input (a skill added
        as one file gets a window mode without UI code)."""
        try:
            skill = create_skill(name, self.ctx)
            if skill.cli_positional is None:
                raise RuntimeError(f"the {name} skill takes no text input")
            params = skill.Input(**{skill.cli_positional: text})
            deltas = skill.stream(params)
            if deltas is None:
                yield Delta(skill.render(skill.run(params)))
            else:
                for delta in deltas:
                    yield Delta(delta)
            yield Finished()
        except ValidationError as exc:
            yield Failed(f"{name}: {exc.errors(include_input=False)[0]['msg']}")
        except RegistryError as exc:
            yield Failed(str(exc.args[0]))
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))

    def _turn_events(self, prepared: PreparedTurn, state: ChatState) -> Iterator[Event]:
        try:
            turn, deltas = ChatSkill(self.ctx).start_turn(prepared)
            state.session_id = turn.session_id
            for text in deltas:
                yield Delta(text)
            note = (
                f"(cut to fit the context: {', '.join(turn.truncated)})" if turn.truncated else ""
            )
            yield Finished(note=note, session_id=turn.session_id)
        except _KNOWN_ERRORS as exc:
            yield Failed(str(exc))
