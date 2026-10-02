"""MCP front-end: ``ve mcp`` serves search, ask and match to Claude Code, Cursor and similar tools.

The caller is usually a cloud-hosted model, so everything returned counts as outbound:
- files under the never-send rules (secrets, ``.claude`` memory, ...) are left out of every result;
- government/financial IDs in returned text are masked;
- the local models answer (``allow_cloud=False``): this server never routes a request to a cloud
  provider and never grants cloud consent;
- ``match`` takes the text inline, so a caller cannot make the server read arbitrary files.
Nothing is loaded until the first tool call, and chat models are unloaded after each answer.
"""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from vector_embed.core import runtime
from vector_embed.core.match.pipeline import DocumentScore
from vector_embed.core.privacy.mask import mask_sensitive
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.settings import Settings
from vector_embed.core.skills.ask import AskSkill, gateway_of, privacy_of
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.match import MatchInput, MatchSkill
from vector_embed.core.skills.search import SearchSkill
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

SERVER_NAME = "vector-embed"
LOCK_OWNER = "mcp"
MAX_RESULTS = 25
MAX_QUESTION_CHARS = 2_000
MAX_MATCH_CHARS = 30_000
_OVERFETCH = 2  # search extra rows so removing private files still leaves ``limit`` results

INSTRUCTIONS = (
    "Search the user's own computer: code, notes, documents, PDFs and images, indexed locally. "
    "Use `search` to find files and passages (filters: type:code ext:py proj:name after:2026-01). "
    "Use `ask` for a cited answer synthesised from many files. Use `match` to rank the user's "
    "documents (default: resumes) against a pasted job description. Results are read-only; "
    "private files are never included and ID numbers are masked."
)

Query = Annotated[str, Field(max_length=MAX_QUESTION_CHARS)]
Limit = Annotated[int, Field(ge=1, le=MAX_RESULTS)]


class SearchHit(BaseModel):
    path: str
    project: str
    kind: str
    location: str
    symbol: str
    start_line: int
    end_line: int
    score: float
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchHit]
    withheld: int = Field(description="Private files left out of the results")


class SourceRef(BaseModel):
    n: int
    path: str
    location: str


class AskResponse(BaseModel):
    answer: str
    not_found: bool
    sources: list[SourceRef] = Field(description="The sources the answer actually cites")
    withheld: int


class MatchEntry(BaseModel):
    rank: int
    path: str
    score: int = Field(description="0-100, computed from the requirement checklist")
    summary: str
    unverified: int = Field(description="Claims whose evidence quote was not found in the file")
    reduced: bool = Field(description="The document was cut to fit the model's context")
    error: str = ""


class MatchResponse(BaseModel):
    ranked: list[MatchEntry]
    withheld: int


def _clean(text: str) -> str:
    """Mask government/financial IDs; the same rule that guards every cloud request."""
    return mask_sensitive(text).text


class Service:
    """The three tools over one ``SkillContext``; no MCP types, so it is unit-testable."""

    def __init__(self, ctx: SkillContext) -> None:
        self._ctx = ctx
        privacy = privacy_of(ctx)
        if privacy is None:
            raise RuntimeError("the privacy filter is required to serve MCP clients")
        self._privacy: PrivacyFilter = privacy

    # ------------------------------------------------------------------ search
    def search(self, query: str, limit: int, current_project: str | None = None) -> SearchResponse:
        found = SearchSkill(self._ctx).search(query, limit * _OVERFETCH, current_project)
        allowed = [r for r in found if not self._privacy.is_never_send(r.path)]
        hits = [
            SearchHit(
                path=r.path,
                project=r.project,
                kind=r.kind,
                location=r.location,
                symbol=r.symbol,
                start_line=r.start_line,
                end_line=r.end_line,
                score=round(r.score, 4),
                snippet=_clean(r.snippet),
            )
            for r in allowed[:limit]
        ]
        return SearchResponse(results=hits, withheld=len(found) - len(allowed))

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, limit: int | None = None) -> AskResponse:
        gateway = gateway_of(self._ctx)
        gateway.begin_chat()  # takes the chat lock, so the indexer waits and the GPU is ours
        try:
            run = AskSkill(self._ctx).prepare(question, limit, outbound=True)
            for _ in run.deltas():
                pass
        finally:
            gateway.end_chat("mcp request finished")
        result = run.result
        return AskResponse(
            answer=_clean(result.answer),
            not_found=result.not_found,
            sources=[SourceRef(n=s.n, path=s.path, location=s.location) for s in result.cited],
            withheld=result.withheld,
        )

    # ------------------------------------------------------------------ match
    def match(self, text: str, doc_type: str = "resume", top: int = 5) -> MatchResponse:
        gateway = gateway_of(self._ctx)
        gateway.begin_chat()
        try:
            skill = MatchSkill(self._ctx)
            run = skill.prepare(MatchInput(jd=text, doc_type=doc_type, top=top))
            withheld = 0
            for candidate in run.candidates:
                if candidate.selected and self._privacy.is_never_send(candidate.path):
                    candidate.selected = False
                    withheld += 1
            if any(c.selected for c in run.candidates):
                skill.pipeline.score(run)
        finally:
            gateway.end_chat("mcp request finished")
        entries = [self._entry(rank, item) for rank, item in enumerate(run.ranked(), 1)]
        return MatchResponse(ranked=entries, withheld=withheld)

    @staticmethod
    def _entry(rank: int, item: DocumentScore) -> MatchEntry:
        breakdown = item.breakdown
        return MatchEntry(
            rank=rank,
            path=item.candidate.path,
            score=max(item.score, 0),
            summary=_clean(breakdown.summary_line) if breakdown else "",
            unverified=breakdown.unverified if breakdown else 0,
            reduced=bool(item.judgement and item.judgement.reduced),
            error=_clean(item.error),
        )


def build_server(service_factory: Callable[[], Service]) -> MCPServer:
    """The MCP server; the service (and with it the index and models) is built on first use."""
    lock = threading.Lock()  # tools run on worker threads; one request at a time owns the GPU
    cache: list[Service] = []

    @contextmanager
    def service() -> Iterator[Service]:
        with lock:
            if not cache:
                cache.append(service_factory())
            try:
                yield cache[0]
            except RuntimeError as exc:  # battery policy, missing model, chat lock held, ...
                raise ToolError(str(exc)) from exc

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @server.tool(annotations=read_only)
    def search(
        query: Query, limit: Limit = 10, current_project: str | None = None
    ) -> SearchResponse:
        """Hybrid semantic + keyword search over the user's files. Returns paths, line numbers
        and snippets. Filters inside the query: type:code|doc|pdf|img|plan|memory ext:py
        proj:name in:D:\\path after:2026-01 before:2026-06."""
        with service() as svc:
            return svc.search(query, limit, current_project)

    @server.tool(annotations=read_only)
    def ask(question: Query, limit: Limit | None = None) -> AskResponse:
        """Answer a question from the user's indexed files using a local model, with citations
        to real files and lines. Slower than `search` (loads a local chat model); answers
        "Not found" instead of guessing."""
        with service() as svc:
            return svc.ask(question, limit)

    @server.tool(annotations=read_only)
    def match(
        text: Annotated[str, Field(max_length=MAX_MATCH_CHARS)],
        doc_type: str = "resume",
        top: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> MatchResponse:
        """Rank the user's documents of one type (default: resumes) against pasted text such as a
        job description. Scores come from a fixed requirement checklist judged by a local model."""
        with service() as svc:
            return svc.match(text, doc_type, top)

    return server


@contextmanager
def open_service(settings: Settings) -> Iterator[Service]:
    with ExitStack() as stack:
        state = stack.enter_context(StateDb(settings.storage.data_dir))
        ctx = runtime.build_skill_context(settings, state, owner=LOCK_OWNER, allow_cloud=False)
        yield Service(ctx)


def serve(settings: Settings) -> None:
    """Run over stdio until the client disconnects."""
    stack = ExitStack()

    def factory() -> Service:
        return stack.enter_context(open_service(settings))

    with stack:
        build_server(factory).run("stdio")
