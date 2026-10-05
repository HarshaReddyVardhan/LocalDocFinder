"""Filter and sort an already fetched result list. Pure and in memory: no re-query, no I/O."""

import enum
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from localdoc_finder.core.skills.search import SearchResult


class SortOrder(enum.Enum):
    RELEVANCE = "Relevance"
    DATE_INDEXED = "Date indexed"
    NAME = "File name"


def result_ext(result: SearchResult) -> str:
    """The file's extension in lower case with its dot ('' for none)."""
    return (result.ext or Path(result.path).suffix).lower()


def ext_counts(results: Iterable[SearchResult]) -> list[tuple[str, int]]:
    """Extensions present in ``results`` with how many hits each has, most common first."""
    counts = Counter(result_ext(r) for r in results)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _name_key(result: SearchResult) -> tuple[str, str]:
    return Path(result.path).name.lower(), result.path.lower()


def refine(
    results: list[SearchResult], ext: str | None = None, order: SortOrder = SortOrder.RELEVANCE
) -> list[SearchResult]:
    """Keep hits with extension ``ext`` (all when ``None``), then order them.

    Relevance keeps the search's own ranking (it already puts strong hits before weak ones), and
    every sort is stable, so ties keep that ranking too.
    """
    kept = [r for r in results if ext is None or result_ext(r) == ext.lower()]
    if order is SortOrder.DATE_INDEXED:
        kept.sort(key=lambda r: r.indexed_at, reverse=True)
    elif order is SortOrder.NAME:
        kept.sort(key=_name_key)
    return kept
