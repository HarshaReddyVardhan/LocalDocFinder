"""Hybrid search: vector + BM25 (LanceDB FTS) fused with RRF, a filename boost and filters.

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
from pathlib import Path

from pydantic import Field

from vector_embed.core import hooks
from vector_embed.core.providers.base import ProviderError
from vector_embed.core.retrieval import hybrid_candidates
from vector_embed.core.skills.base import (
    UI_LIST,
    Skill,
    SkillInput,
    register_skill,
)
from vector_embed.core.store.lance import CHUNKS, Row, sql_quote

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
_MIN_TERM_CHARS = 3
_FILTER_ONLY_FANOUT = 4


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


def _to_result(row: Row, score: float) -> SearchResult:
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
        score=score,
        ext=row["ext"],
        mtime=row["mtime"],
        text=body,
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
        for r in results:
            where = f" ({r.location})" if r.location else ""
            lines.append(f"{r.score:.4f}  [{r.kind}] {r.path}{where}  {r.symbol}")
            lines.append(f"        {r.snippet[:140]}")
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

        if not parsed.text:  # filters only: the newest matching chunks
            rows = store.scan(CHUNKS, _COLUMNS, parsed.where, cfg.candidates * _FILTER_ONLY_FANOUT)
            rows.sort(key=lambda r: r["mtime"], reverse=True)
            return self._group([(r, 1.0) for r in rows], limit, current_project, parsed.text)

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
        )
        pairs = [(c.row, c.score) for c in candidates]
        return self._group(pairs, limit, current_project, parsed.text)

    def _group(
        self,
        pairs: list[tuple[Row, float]],
        limit: int,
        current_project: str | None,
        text: str,
    ) -> list[SearchResult]:
        """Best chunk per file, boosted for the current project and for filename matches."""
        cfg = self.ctx.settings.search
        words = [w for w in re.findall(r"\w+", text.lower()) if len(w) >= _MIN_TERM_CHARS]
        best: dict[str, SearchResult] = {}
        for row, base in pairs:
            score = base
            if current_project and row["project"].lower() == current_project.lower():
                score *= cfg.current_project_boost
            name = Path(row["path"]).name.lower()
            if any(w in name for w in words):
                score *= cfg.filename_boost
            path = row["path"]
            current = best.get(path)
            if current is None:
                best[path] = _to_result(row, score)
                continue
            hits = current.extra_hits + 1
            if score > current.score:
                best[path] = _to_result(row, score)
            best[path].extra_hits = hits
        return sorted(best.values(), key=lambda r: r.score, reverse=True)[:limit]
