"""Ask: answer a question from the whole index, streamed, with checked ``[n]`` citations.

Flow: hybrid retrieval (about 20 chunks) -> merge neighbours and group by file -> pack about
5k tokens -> the LLM answers only from those sources. A question with no hits is answered
"Not found" without calling the model at all.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic import Field

from localdoc_finder.core import hooks
from localdoc_finder.core.llm import ChatBlockedError, ChatTarget, LlmGateway, NoChatModelError
from localdoc_finder.core.models.catalog import ROLE_CHAT, ROLE_CODE_CHAT
from localdoc_finder.core.privacy.policy import PrivacyFilter
from localdoc_finder.core.rag import (
    NOT_FOUND,
    SOURCE_COLUMNS,
    Source,
    build_messages,
    build_sources,
    cited_sources,
    format_citations,
    is_code_heavy,
)
from localdoc_finder.core.retrieval import Candidate, hybrid_candidates
from localdoc_finder.core.skills.base import (
    UI_PANEL,
    Skill,
    SkillContext,
    SkillInput,
    register_skill,
)
from localdoc_finder.core.skills.search import parse_query
from localdoc_finder.core.store.lance import CHUNKS


def privacy_of(ctx: SkillContext) -> PrivacyFilter | None:
    privacy = ctx.extras.get("privacy")
    return privacy if isinstance(privacy, PrivacyFilter) else None


def gateway_of(ctx: SkillContext) -> LlmGateway:
    gateway = ctx.extras.get("llm")
    if not isinstance(gateway, LlmGateway):
        raise RuntimeError("no LLM gateway configured for this context")
    return gateway


def _resolved_target(gateway: LlmGateway, role: str) -> ChatTarget | None:
    """The model that will answer, or ``None`` so the stream raises the usual clear error."""
    try:
        return gateway.target(role)
    except (ChatBlockedError, NoChatModelError):
        return None


class AskInput(SkillInput):
    question: str = Field(description="Question about your files; search filters also work")
    limit: int | None = Field(default=None, ge=1, description="Chunks to retrieve")
    session: bool = Field(default=False, description="Keep the model loaded for follow-ups")


@dataclass
class AskResult:
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    cited: list[Source] = field(default_factory=list)
    invalid_citations: set[int] = field(default_factory=set)
    not_found: bool = False
    role: str = ROLE_CHAT
    withheld: int = 0  # private files kept out of a cloud request
    notice: str = ""  # the cloud failed and the local model answered


class AskRun:
    """One answer in progress; iterate ``deltas()`` and read ``result`` afterwards."""

    def __init__(
        self,
        gateway: LlmGateway,
        question: str,
        sources: list[Source],
        role: str,
        session: bool,
        *,
        withheld: int = 0,
        target: ChatTarget | None = None,
    ) -> None:
        self._target = target
        self._gateway = gateway
        self._question = question
        self.messages = build_messages(question, sources)  # built once: preview == what is sent
        self._session = session
        self.result = AskResult(sources=sources, role=role, withheld=withheld)

    def deltas(self) -> Iterator[str]:
        result = self.result
        if not result.sources:
            result.answer, result.not_found = NOT_FOUND, True
            yield NOT_FOUND
            hooks.emit(hooks.Answered("ask", self._question, NOT_FOUND))
            return
        parts: list[str] = []
        for chunk in self._gateway.stream(
            self.messages, result.role, session=self._session, target=self._target
        ):
            if chunk.notice:
                result.notice = chunk.notice
            if chunk.text:
                parts.append(chunk.text)
                yield chunk.text
        result.answer = "".join(parts).strip()
        result.not_found = result.answer.startswith(NOT_FOUND)
        result.cited, result.invalid_citations = cited_sources(result.answer, result.sources)
        cited = tuple(dict.fromkeys(source.path for source in result.cited))
        hooks.emit(hooks.Answered("ask", self._question, result.answer, cited))

    def footer(self) -> str:
        """Text appended after the answer: the sources it actually cited."""
        result = self.result
        withheld = ""
        if result.withheld:
            noun = "file was" if result.withheld == 1 else "files were"
            withheld = f"\n({result.withheld} private {noun} not sent to the cloud)"
        if result.notice:
            withheld = f"\n({result.notice})" + withheld
        if result.not_found:
            return withheld
        if result.cited:
            note = ""
            if result.invalid_citations:
                numbers = ", ".join(f"[{n}]" for n in sorted(result.invalid_citations))
                note = f"\n(ignored citations to sources that do not exist: {numbers})"
            return "\n\nSources:\n" + format_citations(result.cited) + note + withheld
        return (
            "\n\n(The answer cited no sources; treat it with caution.)\nSearched:\n"
            + format_citations(result.sources[:5])
            + withheld
        )


@register_skill("ask")
class AskSkill(Skill):
    name = "ask"
    title = "Ask"
    description = "Ask a question and get an answer from your indexed files, with citations."
    Input = AskInput
    roles = (ROLE_CHAT, ROLE_CODE_CHAT)
    ui_hint = UI_PANEL
    cli_positional = "question"

    def prepare(
        self,
        question: str,
        limit: int | None = None,
        session: bool = False,
        *,
        outbound: bool = False,
    ) -> AskRun:
        """Retrieve and pack sources; nothing is generated until ``deltas()`` is read.

        ``outbound`` means the answer will be handed to a cloud-hosted caller (MCP), so files
        under the never-send rules are kept out even though the model itself is local.
        """
        ctx = self.ctx
        chat_cfg = ctx.settings.chat
        gateway = gateway_of(ctx)
        parsed = parse_query(question)
        candidates = hybrid_candidates(
            ctx.store,
            ctx.embedder,
            ctx.power,
            ctx.settings.search,
            table=CHUNKS,
            text=parsed.text or question,
            columns=SOURCE_COLUMNS,
            where=parsed.where,
            limit=limit or chat_cfg.retrieve_chunks,
            force_cpu=True,  # the GPU belongs to the LLM
            min_similarity=ctx.similarity_floor,
        )
        sources = build_sources(candidates, chat_cfg.context_token_budget)
        role = self._role(sources)
        # The privacy rule follows the role that will really answer: code-heavy context goes to
        # ``code_chat``, which can be routed to the cloud while ``chat`` stays local.
        withheld = 0
        if outbound or gateway.will_use_cloud(role):
            candidates, withheld = self._without_private(candidates)
            sources = build_sources(candidates, chat_cfg.context_token_budget)
            role = self._role(sources)
        target = _resolved_target(gateway, role)  # decided once; the send must not re-route
        return AskRun(
            gateway,
            parsed.text or question,
            sources,
            role,
            session,
            withheld=withheld,
            target=target,
        )

    def _role(self, sources: list[Source]) -> str:
        code = self.ctx.settings.chat.code_routing and is_code_heavy(sources)
        return ROLE_CODE_CHAT if code else ROLE_CHAT

    def _without_private(self, candidates: list[Candidate]) -> tuple[list[Candidate], int]:
        """Cloud requests never include files under the never-send rules."""
        privacy = privacy_of(self.ctx)
        if privacy is None:
            return candidates, 0
        kept = [c for c in candidates if not privacy.is_never_send(c.row["path"])]
        blocked = {c.row["path"] for c in candidates} - {c.row["path"] for c in kept}
        return kept, len(blocked)

    def stream(self, params: SkillInput) -> Iterator[str]:
        assert isinstance(params, AskInput)
        run = self.prepare(params.question, params.limit, params.session)
        yield from run.deltas()
        yield run.footer()

    def run(self, params: SkillInput) -> AskResult:
        assert isinstance(params, AskInput)
        run = self.prepare(params.question, params.limit, params.session)
        for _ in run.deltas():
            pass
        return run.result

    def render(self, output: object) -> str:
        assert isinstance(output, AskResult)
        return output.answer
