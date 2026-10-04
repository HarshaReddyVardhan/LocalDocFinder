"""Retrieval-augmented answering helpers: pack sources, build the prompt, check citations."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localdoc_finder.core.extractors.base import KIND_OUTLINE
from localdoc_finder.core.prompt_safety import Fence, fence_for
from localdoc_finder.core.providers.base import Message
from localdoc_finder.core.retrieval import Candidate
from localdoc_finder.core.tokens import estimate_tokens, fit_to_budget

NOT_FOUND = "Not found in your indexed files."
SOURCE_COLUMNS = [
    "text", "path", "project", "kind", "symbol", "start_line", "end_line", "page", "chunk_hash",
]  # fmt: skip
CODE_KINDS = frozenset({"code", "outline"})

SYSTEM_PROMPT = (
    "You answer questions about the user's own files using ONLY the numbered sources below. "
    "Cite the sources that support each statement as [n]. If the sources do not contain the "
    f"answer, reply exactly: {NOT_FOUND} Never use outside knowledge and never invent file names, "
    "functions or line numbers."
)
_CITATION = re.compile(r"\[(\d+)\]")
_HEADER_TOKENS = 20  # per-source overhead in the prompt
_MIN_SOURCE_TOKENS = 50  # the best source is never cut below this, however small the budget


@dataclass(frozen=True)
class Source:
    n: int
    path: str
    project: str
    kind: str
    symbol: str
    start_line: int
    end_line: int
    page: int
    text: str

    @property
    def location(self) -> str:
        if self.page:
            return f"page {self.page}"
        if self.start_line:
            return f"lines {self.start_line}-{self.end_line}"
        return ""

    def reference(self) -> str:
        """``path:line`` (or ``path (page n)``), the form shown to the user."""
        if self.page:
            return f"{self.path} (page {self.page})"
        if self.start_line:
            return f"{self.path}:{self.start_line}"
        return self.path


@dataclass
class _Piece:
    """A chunk being merged with its neighbours."""

    project: str
    kind: str
    symbol: str
    start_line: int
    end_line: int
    page: int
    text: str

    def touches(self, other: "_Piece") -> bool:
        """True if ``other`` is on the same page or in adjacent/overlapping lines.

        A file outline spans the file's lines but is a summary, not those lines; merging it by
        line numbers would drop the real code it overlaps.
        """
        if KIND_OUTLINE in (self.kind, other.kind):
            return False
        if self.page or other.page:
            return bool(self.page) and self.page == other.page
        if not self.end_line or not other.start_line:
            return False
        return other.start_line <= self.end_line + 1

    def absorb(self, other: "_Piece") -> None:
        lines = other.text.splitlines()
        if not other.page:  # overlapping line windows repeat their shared lines
            lines = lines[max(0, self.end_line - other.start_line + 1) :]
        self.text += "\n" + "\n".join(lines)
        self.end_line = max(self.end_line, other.end_line)
        if other.symbol and other.symbol not in self.symbol:
            self.symbol = f"{self.symbol}, {other.symbol}".strip(", ")


def _body(text: str) -> str:
    """Stored chunk text starts with a ``project > path > symbol`` header line; drop it."""
    return text.split("\n", 1)[1] if "\n" in text else text


def _piece(row: dict[str, Any]) -> _Piece:
    return _Piece(
        project=row["project"],
        kind=row["kind"],
        symbol=row["symbol"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        page=int(row["page"]),
        text=_body(row["text"]),
    )


def _merge_neighbours(rows: list[dict[str, Any]]) -> list[_Piece]:
    pieces = sorted((_piece(r) for r in rows), key=lambda p: (p.page, p.start_line))
    merged: list[_Piece] = []
    for piece in pieces:
        if merged and merged[-1].touches(piece):
            merged[-1].absorb(piece)
        else:
            merged.append(piece)
    return merged


def build_sources(
    candidates: Sequence[Candidate], budget_tokens: int, max_docs: int = 8
) -> list[Source]:
    """Group hits by file, merge neighbouring chunks, and pack the best into the token budget.

    A piece that does not fit is skipped, not the end of packing: a smaller, later piece may
    still fit. The best piece is always kept, cut down to the budget if it alone is too big.
    """
    by_path: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:  # best-first, so file order follows relevance
        by_path.setdefault(candidate.row["path"], []).append(candidate.row)
    sources: list[Source] = []
    used = 0
    for path, rows in list(by_path.items())[:max_docs]:
        for piece in _merge_neighbours(rows):
            text = piece.text
            cost = estimate_tokens(text) + _HEADER_TOKENS
            if used + cost > budget_tokens:
                if sources:
                    continue
                text, _ = fit_to_budget(
                    text, max(_MIN_SOURCE_TOKENS, budget_tokens - _HEADER_TOKENS)
                )
                cost = estimate_tokens(text) + _HEADER_TOKENS
            used += cost
            sources.append(
                Source(
                    n=len(sources) + 1,
                    path=path,
                    project=piece.project,
                    kind=piece.kind,
                    symbol=piece.symbol,
                    start_line=piece.start_line,
                    end_line=piece.end_line,
                    page=piece.page,
                    text=text,
                )
            )
    return sources


def is_code_heavy(sources: Sequence[Source]) -> bool:
    return bool(sources) and sum(s.kind in CODE_KINDS for s in sources) * 2 > len(sources)


def format_sources(sources: Sequence[Source], fence: Fence) -> str:
    """Numbered source blocks; each body is fenced so its text is never taken as instructions."""
    blocks = []
    for source in sources:
        where = f" ({source.location})" if source.location else ""
        symbol = f" · {source.symbol}" if source.symbol and source.symbol != "<module>" else ""
        head = f"[{source.n}] {Path(source.path).name}{symbol}{where}"
        blocks.append(f"{head}\n{source.path}\n{fence.wrap(source.text)}")
    return "\n\n".join(blocks)


def build_messages(question: str, sources: Sequence[Source]) -> list[Message]:
    fence = fence_for(*(source.text for source in sources))
    return [
        Message("system", f"{SYSTEM_PROMPT}\n\n{fence.rule}"),
        Message("user", f"Sources:\n\n{format_sources(sources, fence)}\n\nQuestion: {question}"),
    ]


def cited_sources(answer: str, sources: Sequence[Source]) -> tuple[list[Source], set[int]]:
    """Sources the answer cites, in citation order, plus any cited numbers that do not exist."""
    by_number = {s.n: s for s in sources}
    order: list[int] = []
    invalid: set[int] = set()
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if number in by_number:
            if number not in order:
                order.append(number)
        else:
            invalid.add(number)
    return [by_number[n] for n in order], invalid


def format_citations(cited: Sequence[Source]) -> str:
    return "\n".join(f"[{s.n}] {s.reference()}" for s in cited)
