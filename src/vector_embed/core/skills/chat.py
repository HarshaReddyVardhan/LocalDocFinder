"""Chat with documents: a conversation pinned to chosen files and/or pasted text.

Pinned context goes first in every prompt (providers that cache a repeated prompt start then
charge less for follow-ups). A pasted job description or similar is a *scratch document*: it is
kept in the chat session row, never written to the search index. With nothing pinned, the chat
falls back to retrieving context from the index for each message.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field

from vector_embed.core.documents import DocumentError, DocumentLoader, LoadedDocument
from vector_embed.core.llm import ChatBlockedError, ChatTarget, LlmGateway, NoChatModelError
from vector_embed.core.models.catalog import ROLE_CHAT
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.providers.base import Message
from vector_embed.core.rag import SOURCE_COLUMNS, build_sources, format_sources
from vector_embed.core.retrieval import hybrid_candidates
from vector_embed.core.skills.ask import gateway_of, privacy_of
from vector_embed.core.skills.base import (
    UI_PANEL,
    Skill,
    SkillContext,
    SkillInput,
    register_skill,
)
from vector_embed.core.store.lance import CHUNKS
from vector_embed.core.store.sqlite import ChatMessage
from vector_embed.core.tokens import estimate_tokens, fit_to_budget

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful assistant helping the user work with their own documents. Base your "
    "answers on the provided documents and the conversation; quote them when it helps. If "
    "something is not in the documents, say so instead of guessing."
)
_TRUNCATED = "\n[... document truncated to fit the model's context ...]"
_MIN_DOC_TOKENS = 200


class ChatInput(SkillInput):
    message: str = Field(description="What to ask or tell the assistant")
    session: int | None = Field(default=None, description="Continue this chat session id")
    pin: list[str] = Field(default_factory=list, description="Files to pin to a new session")
    scratch: str | None = Field(default=None, description="Pasted text kept only in the session")
    keep_loaded: bool = Field(default=False, description="Keep the model loaded for follow-ups")


@dataclass
class ChatTurn:
    """One reply in progress; iterate ``deltas()``, then read ``reply``."""

    session_id: int
    reply: str = ""
    truncated: list[str] = field(default_factory=list)


def _pinned_block(
    docs: list[LoadedDocument], scratch: str, budget_tokens: int
) -> tuple[str, list[str]]:
    """Pinned documents and scratch text sharing ``budget_tokens``; returns text and cut titles."""
    items = [(d.path, d.text) for d in docs]
    if scratch.strip():
        items.append(("pasted text", scratch))
    if not items:
        return "", []
    weights = [max(estimate_tokens(text), 1) for _, text in items]
    total = sum(weights)
    blocks: list[str] = []
    cut: list[str] = []
    for (name, text), weight in zip(items, weights, strict=True):
        share = max(_MIN_DOC_TOKENS, budget_tokens * weight // total)
        body, was_cut = fit_to_budget(text, share)
        if was_cut:
            cut.append(name)
            body += _TRUNCATED
        blocks.append(f"=== {name} ===\n{body}")
    return "\n\n".join(blocks), cut


def trim_history(history: list[ChatMessage], budget_tokens: int) -> list[Message]:
    """Newest messages that fit ``budget_tokens``, oldest dropped first."""
    kept: list[Message] = []
    used = 0
    for item in reversed(history):
        cost = estimate_tokens(item.content)
        if used + cost > budget_tokens and kept:
            break
        used += cost
        role = "assistant" if item.role == "assistant" else "user"
        kept.append(Message(role, item.content))  # type: ignore[arg-type]
    return list(reversed(kept))


@register_skill("chat")
class ChatSkill(Skill):
    name = "chat"
    title = "Chat"
    description = "Chat about pinned documents or pasted text; follow-ups keep the context."
    Input = ChatInput
    roles = (ROLE_CHAT,)
    ui_hint = UI_PANEL
    cli_positional = "message"

    def __init__(self, ctx: SkillContext) -> None:
        super().__init__(ctx)
        self._gateway: LlmGateway = gateway_of(ctx)
        loader = ctx.extras.get("documents")
        if not isinstance(loader, DocumentLoader):
            raise RuntimeError("no document loader configured for this context")
        self._loader = loader

    # ------------------------------------------------------------------ sessions
    def open_session(
        self, title: str, pin: list[str] | None = None, scratch: str | None = None
    ) -> int:
        """Create a chat session; loads the pinned files up front so errors surface now."""
        pinned = list(pin or [])
        for path in pinned:
            self._loader.load(path)  # raises DocumentError for secrets / unreadable files
        context = {"pinned": pinned, "scratch": scratch or ""}
        return self.ctx.state.create_session(title[:80] or "chat", context)

    def _pinned(self, session_id: int) -> tuple[list[LoadedDocument], str]:
        context = self.ctx.state.session_context(session_id)
        docs: list[LoadedDocument] = []
        for path in context.get("pinned", []):
            try:
                docs.append(self._loader.load(path))
            except DocumentError:
                logger.warning("chat: pinned file unavailable: %s", path)
        return docs, str(context.get("scratch", ""))

    # ------------------------------------------------------------------ prompt
    def build_prompt(
        self, session_id: int, message: str, target: ChatTarget | None = None
    ) -> tuple[list[Message], list[str]]:
        """Messages for the model (pinned context first) and the titles that had to be cut.

        ``target`` is the route the reply will be sent to; the privacy rules are applied for that
        exact route. Without it the current route is looked up.
        """
        cfg = self.ctx.settings.chat
        docs, scratch = self._pinned(session_id)
        cloud = self._gateway.will_use_cloud(ROLE_CHAT) if target is None else not target.local
        privacy = privacy_of(self.ctx)
        if cloud and privacy is not None:
            private = [d.path for d in docs if privacy.is_never_send(d.path, d.doc_type or None)]
            if private:
                names = ", ".join(Path(p).name for p in private)
                raise ChatBlockedError(
                    f"{names} is private and cannot be sent to a cloud model; "
                    "switch to the local model or unpin it"
                )
        block, cut = _pinned_block(docs, scratch, cfg.context_token_budget)
        if not block:
            block = self._retrieved_context(message, privacy if cloud else None)
        system = SYSTEM_PROMPT + (f"\n\nDocuments:\n\n{block}" if block else "")
        history = trim_history(self.ctx.state.messages(session_id), cfg.history_token_budget)
        return [Message("system", system), *history, Message("user", message)], cut

    def _retrieved_context(self, message: str, cloud_filter: PrivacyFilter | None = None) -> str:
        ctx = self.ctx
        cfg = ctx.settings.chat
        candidates = hybrid_candidates(
            ctx.store,
            ctx.embedder,
            ctx.power,
            ctx.settings.search,
            table=CHUNKS,
            text=message,
            columns=SOURCE_COLUMNS,
            limit=cfg.retrieve_chunks,
            force_cpu=True,
        )
        if cloud_filter is not None:  # a cloud request never contains private files
            candidates = [c for c in candidates if not cloud_filter.is_never_send(c.row["path"])]
        return format_sources(build_sources(candidates, cfg.context_token_budget))

    # ------------------------------------------------------------------ turns
    def turn(self, params: ChatInput) -> tuple[ChatTurn, Iterator[str]]:
        session_id = params.session or self.open_session(params.message, params.pin, params.scratch)
        target = self._target()
        messages, cut = self.build_prompt(session_id, params.message, target)
        turn = ChatTurn(session_id, truncated=cut)
        keep = params.keep_loaded or self._gateway.session_active
        return turn, self._generate(turn, messages, params.message, keep, target)

    def _target(self) -> ChatTarget | None:
        """The route decided once per turn; ``None`` lets the stream raise the usual error."""
        try:
            return self._gateway.target(ROLE_CHAT)
        except (ChatBlockedError, NoChatModelError):
            return None

    def _generate(
        self,
        turn: ChatTurn,
        messages: list[Message],
        user_text: str,
        keep: bool,
        target: ChatTarget | None = None,
    ) -> Iterator[str]:
        parts: list[str] = []
        try:
            for chunk in self._gateway.stream(messages, ROLE_CHAT, session=keep, target=target):
                if chunk.text:
                    parts.append(chunk.text)
                    yield chunk.text
        finally:
            turn.reply = "".join(parts).strip()
            if turn.reply:  # a cut-off reply is still kept so the conversation stays coherent
                state = self.ctx.state
                state.add_message(turn.session_id, "user", user_text)
                state.add_message(turn.session_id, "assistant", turn.reply)

    def stream(self, params: SkillInput) -> Iterator[str]:
        assert isinstance(params, ChatInput)
        turn, deltas = self.turn(params)
        yield from deltas
        notes = []
        if turn.truncated:
            notes.append("(cut to fit the context: " + ", ".join(turn.truncated) + ")")
        notes.append(f"[session {turn.session_id}]")
        yield "\n\n" + " ".join(notes)

    def run(self, params: SkillInput) -> ChatTurn:
        assert isinstance(params, ChatInput)
        turn, deltas = self.turn(params)
        for _ in deltas:
            pass
        return turn

    def render(self, output: object) -> str:
        assert isinstance(output, ChatTurn)
        return output.reply
