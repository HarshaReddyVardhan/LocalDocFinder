"""Hybrid search: vector + BM25 (LanceDB FTS) and file names, ranked by relevance, with filters.

Every result carries a 0-100 ``relevance`` and a ``weak`` flag (see ``core/ranking.py``): weak
results come after the strong ones, so nothing is hidden but the order can be trusted.

Filters (any order, mixed with the free text):
    type:img|code|doc|plan|memory|note|pdf|docx   ext:py   proj:name   in:D:\\path
    after:2026-01   before:2026-06
Example: ``payment retry proj:billing type:code after:2026-01``
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field

from localdoc_finder.core import hooks
from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.ranking import (
    FileEvidence,
    Ranked,
    name_match,
    query_terms,
    rank_files,
    term_coverage,
)
from localdoc_finder.core.retrieval import hybrid_candidates
from localdoc_finder.core.skills.base import (
    UI_LIST,
    Skill,
    SkillInput,
    register_skill,
)
from localdoc_finder.core.store.lance import CHUNKS, Row, sql_quote
from localdoc_finder.core.store.search_columns import chunk_body, search_text

logger = logging.getLogger(__name__)

_FILTER_RE = re.compile(r'\b(type|ext|proj|project|in|after|before):("[^"]*"|\S+)', re.IGNORECASE)
_TYPE_SQL = {
    "img": "kind = 'image'", "image": "kind = 'image'", "images": "kind = 'image'",
    "code": "kind IN ('code','outline')",
    "doc": "kind = 'doc'", "docs": "kind = 'doc'",
    "plan": "source = 'claude-plan'", "plans": "source = 'claude-plan'",
    "memory": "source = 'claude-memory'", "mem": "source = 'claude-memory'",
    "rules": "source = 'agent-rules'",
    "note": "kind = 'ai-note'", "ai": "kind = 'ai-note'",
}  # fmt: skip
_COLUMNS = [
    "text", "path", "project", "kind", "source", "ext", "symbol", "start_line", "end_line",
    "page", "chunk_hash", "mtime",
]  # fmt: skip
_SNIPPET_CHARS = 260
_FILTER_ONLY_FANOUT = 4
_PLURAL_LIKE_MIN_CHARS = 4
_FULL_RELEVANCE = 100  # a filter-only listing: every row matches the filters exactly


class SearchDisabledError(RuntimeError):
    """Search is turned off while on battery (``search_on_battery = false``)."""


@dataclass
class SearchResult:
    path: str
    project: str
    kind: str
    source: str
    symbol: str
    start_line: int
    end_line: int
    page: int
    snippet: str
    score: float
    ext: str = ""
    mtime: int = 0
    extra_hits: int = 0
    text: str = ""
    relevance: int = 0  # 0-100
    weak: bool = False  # shown after the strong results, as "less relevant"

    @property
    def location(self) -> str:
        if self.page:
            return f"page {self.page}"
        if self.start_line:
            return f"line {self.start_line}"
        return ""


@dataclass(frozen=True)
class ParsedQuery:
    text: str
    where: str


def _timestamp(value: str) -> int | None:
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return int(datetime.strptime(value, fmt).timestamp())  # local time
        except ValueError:
            continue
    return None


def _type_clause(value: str) -> str | None:
    clauses: list[str] = []
    for token in value.lower().split(","):
        if token in _TYPE_SQL:
            clauses.append(_TYPE_SQL[token])
        elif token:
            clauses.append(f"ext = {sql_quote('.' + token.lstrip('.'))}")
    return "(" + " OR ".join(clauses) + ")" if clauses else None


def parse_query(raw: str) -> ParsedQuery:
    """Split a raw query into free text and a SQL ``where`` clause."""
    clauses: list[str] = []

    def take(match: re.Match[str]) -> str:
        key, value = match.group(1).lower(), match.group(2).strip('"')
        clause: str | None = None
        if key == "type":
            clause = _type_clause(value)
        elif key == "ext":
            clause = f"ext = {sql_quote('.' + value.lower().lstrip('.'))}"
        elif key in ("proj", "project"):
            clause = f"lower(project) = {sql_quote(value.lower())}"
        elif key == "in":
            prefix = os.path.normpath(value).rstrip("\\/") + os.sep
            clause = f"starts_with(lower(path), {sql_quote(prefix.lower())})"
        elif key in ("after", "before"):
            stamp = _timestamp(value)
            if stamp is not None:
                clause = f"mtime {'>=' if key == 'after' else '<'} {stamp}"
        if clause:
            clauses.append(clause)
        return " "

    text = _FILTER_RE.sub(take, raw)
    return ParsedQuery(" ".join(text.split()), " AND ".join(clauses))


class SearchInput(SkillInput):
    query: str = Field(description="Free text plus optional filters, e.g. 'retry proj:billing'")
    limit: int | None = Field(default=None, ge=1, description="Maximum results")
    current_project: str | None = Field(default=None, description="Boost this project's files")


def _like_term(term: str) -> str:
    """A ``LIKE`` fragment that also finds the singular: ``payments`` -> ``payment``."""
    return term.removesuffix("s") if len(term) > _PLURAL_LIKE_MIN_CHARS else term


def _to_result(row: Row, ranked: Ranked | None = None, extra_hits: int = 0) -> SearchResult:
    text: str = row["text"]
    body = text.split("\n", 1)[1] if "\n" in text else text
    return SearchResult(
        path=row["path"],
        project=row["project"],
        kind=row["kind"],
        source=row["source"],
        symbol=row["symbol"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        page=row["page"],
        snippet=" ".join(body.split())[:_SNIPPET_CHARS],
        score=ranked.score if ranked else 1.0,
        ext=row["ext"],
        mtime=row["mtime"],
        extra_hits=extra_hits,
        text=body,
        relevance=ranked.relevance if ranked else _FULL_RELEVANCE,
        weak=ranked.weak if ranked else False,
    )


@register_skill("search")
class SearchSkill(Skill):
    name = "search"
    title = "Search"
    description = "Hybrid semantic + keyword search over code, notes, documents and images."
    Input = SearchInput
    ui_hint = UI_LIST
    cli_positional = "query"

    # ------------------------------------------------------------------ entry points
    def run(self, params: SkillInput) -> list[SearchResult]:
        assert isinstance(params, SearchInput)
        return self.search(params.query, params.limit, params.current_project)

    def render(self, output: object) -> str:
        results: list[SearchResult] = output  # type: ignore[assignment]
        if not results:
            return "no results"
        lines = []
        divided = False
        for r in results:
            if r.weak and not divided:
                lines.append("--- less relevant ---")
                divided = True
            where = f" ({r.location})" if r.location else ""
            lines.append(f"{r.relevance:>3}  [{r.kind}] {r.path}{where}  {r.symbol}")
            lines.append(f"     {r.snippet[:140]}")
        return "\n".join(lines)

    def warm(self) -> None:
        """Load the embedding model ahead of the first query (hides the model-load delay)."""
        try:
            self.ctx.embedder.embed(["warmup"], kind="query", cpu=self.ctx.query_on_cpu())
        except ProviderError:
            logger.debug("search: warm-up failed", exc_info=True)

    # ------------------------------------------------------------------ search
    def search(
        self, query: str, limit: int | None = None, current_project: str | None = None
    ) -> list[SearchResult]:
        results = self._search(query, limit, current_project)
        hooks.emit(hooks.QueryRan(query, len(results)))
        return results

    def _search(
        self, query: str, limit: int | None, current_project: str | None
    ) -> list[SearchResult]:
        if not self.ctx.power.search_allowed():
            raise SearchDisabledError("search is disabled on battery")
        cfg = self.ctx.settings.search
        limit = limit or cfg.results
        parsed = parse_query(query)
        store = self.ctx.store
        if store.chunks is None:
            return []
        store.require_current_vectors()

        if not parsed.text:  # filters only: the newest matching chunks
            rows = store.scan(CHUNKS, _COLUMNS, parsed.where, cfg.candidates * _FILTER_ONLY_FANOUT)
            rows.sort(key=lambda r: r["mtime"], reverse=True)
            newest: dict[str, Row] = {}
            for row in rows:
                newest.setdefault(row["path"], row)
            return [_to_result(row) for row in newest.values()][:limit]

        floor = self.ctx.similarity_floor
        candidates = hybrid_candidates(
            store,
            self.ctx.embedder,
            self.ctx.power,
            cfg,
            table=CHUNKS,
            text=parsed.text,
            columns=_COLUMNS,
            where=parsed.where,
            force_cpu=self.ctx.query_on_cpu(),
            min_similarity=floor,
        )
        files: dict[str, FileEvidence] = {}
        best_rows: dict[str, Row] = {}
        chunk_counts: dict[str, int] = {}

        def evidence(row: Row) -> FileEvidence:
            path = row["path"]
            if path not in files:
                in_project = bool(current_project) and (
                    row["project"].lower() == (current_project or "").lower()
                )
                files[path] = FileEvidence(path, name=name_match(path, parsed.text),
                                           in_current_project=in_project)  # fmt: skip
                best_rows[path] = row  # candidates come best first
            return files[path]

        terms = query_terms(parsed.text)
        for candidate in candidates:
            found = evidence(candidate.row)
            coverage = None
            if candidate.keyword_rank is not None:  # judged on what BM25 saw: name words + body
                body = chunk_body(candidate.row["text"])
                coverage = term_coverage(terms, search_text(found.path, body))
            found.add_chunk(candidate.score, candidate.similarity, coverage)
            chunk_counts[found.path] = chunk_counts.get(found.path, 0) + 1
        for row in self._filename_rows(parsed, set(files)):
            evidence(row)
        ranked = rank_files(list(files.values()), cfg, floor)
        return [
            _to_result(
                best_rows[r.evidence.path], r, max(0, chunk_counts.get(r.evidence.path, 1) - 1)
            )
            for r in ranked[:limit]
        ]

    def _filename_rows(self, parsed: ParsedQuery, seen: set[str]) -> list[Row]:
        """One chunk of each further file whose name holds a query word as a whole word.

        A file named in the query may have no chunk in the vector or BM25 top-N (its content can
        be tiny or unrelated), so the name itself is a third retrieval leg.
        """
        store, cfg = self.ctx.store, self.ctx.settings.search
        terms = query_terms(parsed.text)
        if not terms:
            return []
        # Old indexes have no name column until the worker upgrades them.
        column = "name" if store.has_column(CHUNKS, "name") else "lower(path)"
        likes = " OR ".join(f"{column} LIKE '%{_like_term(t)}%'" for t in terms)  # \w+ only
        where = f"({likes})" + (f" AND {parsed.where}" if parsed.where else "")
        rows = store.scan(CHUNKS, _COLUMNS, where, cfg.candidates * _FILTER_ONLY_FANOUT)
        found: dict[str, Row] = {}
        for row in rows:
            path = row["path"]
            if path not in seen and path not in found and name_match(path, parsed.text).hit:
                found[path] = row
        return list(found.values())
