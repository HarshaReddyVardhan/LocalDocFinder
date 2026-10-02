"""Ask: answer a question from the whole index, streamed, with checked ``[n]`` citations.

Flow: hybrid retrieval (about 20 chunks) -> merge neighbours and group by file -> pack about
5k tokens -> the LLM answers only from those sources. A question with no hits is answered
"Not found" without calling the model at all.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic import Field

from vector_embed.core.llm import LlmGateway
from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_CODE_CHAT
from vector_embed.core.rag import (
    NOT_FOUND,
    SOURCE_COLUMNS,
    Source,
    build_messages,
    build_sources,
    cited_sources,
    format_citations,
    is_code_heavy,
)
from vector_embed.core.retrieval import hybrid_candidates
from vector_embed.core.skills.base import (
    UI_PANEL,
    Skill,
    SkillContext,
    SkillInput,
    register_skill,
)
from vector_embed.core.skills.search import parse_query
from vector_embed.core.store.lance import CHUNKS


def gateway_of(ctx: SkillContext) -> LlmGateway:
    gateway = ctx.extras.get("llm")
    if not isinstance(gateway, LlmGateway):
        raise RuntimeError("no LLM gateway configured for this context")
    return gateway


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


class AskRun:
    """One answer in progress; iterate ``deltas()`` and read ``result`` afterwards."""

    def __init__(
        self,
        gateway: LlmGateway,
        question: str,
        sources: list[Source],
        role: str,
        session: bool,
    ) -> None:
        self._gateway = gateway
        self._question = question
        self._session = session
        self.result = AskResult(sources=sources, role=role)

    def deltas(self) -> Iterator[str]:
        result = self.result
        if not result.sources:
            result.answer, result.not_found = NOT_FOUND, True
            yield NOT_FOUND
            return
        parts: list[str] = []
        messages = build_messages(self._question, result.sources)
        for chunk in self._gateway.stream(messages, result.role, session=self._session):
            if chunk.text:
                parts.append(chunk.text)
                yield chunk.text
        result.answer = "".join(parts).strip()
        result.not_found = result.answer.startswith(NOT_FOUND)
        result.cited, result.invalid_citations = cited_sources(result.answer, result.sources)

    def footer(self) -> str:
        """Text appended after the answer: the sources it actually cited."""
        result = self.result
        if result.not_found:
            return ""
        if result.cited:
            note = ""
            if result.invalid_citations:
                numbers = ", ".join(f"[{n}]" for n in sorted(result.invalid_citations))
                note = f"\n(ignored citations to sources that do not exist: {numbers})"
            return "\n\nSources:\n" + format_citations(result.cited) + note
        return (
            "\n\n(The answer cited no sources; treat it with caution.)\nSearched:\n"
            + format_citations(result.sources[:5])
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

    def prepare(self, question: str, limit: int | None = None, session: bool = False) -> AskRun:
        """Retrieve and pack sources; nothing is generated until ``deltas()`` is read."""
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
        )
        sources = build_sources(candidates, chat_cfg.context_token_budget)
        code = chat_cfg.code_routing and is_code_heavy(sources)
        role = ROLE_CODE_CHAT if code else ROLE_CHAT
        return AskRun(gateway, parsed.text or question, sources, role, session)

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
