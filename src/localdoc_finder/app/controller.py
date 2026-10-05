"""Qt-free logic behind the search window: project guessing, labels, launching, searching."""

import ctypes
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.skills.search import SearchResult, SearchSkill

logger = logging.getLogger(__name__)

_EDITOR_TITLES = ("Visual Studio Code", "Cursor", "Windsurf")
_MIN_TITLE_PARTS = 3
_NO_WINDOW = 0x08000000
_TAGS = {"image": "IMG", "ai-note": "NOTE", "outline": "FILE"}
_SNIPPET_LINE = 110
_SIZE_UNITS = ("B", "KB", "MB", "GB")
_SIZE_STEP = 1024


def foreground_title() -> str:
    """Title of the foreground window ('' if it cannot be read)."""
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
        buffer = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(user32.GetForegroundWindow(), buffer, 512)
        return str(buffer.value)
    except Exception:  # cosmetic feature; never break the hotkey over it
        return ""


def guess_project(title: str) -> str | None:
    """Editor titles look like ``file.py - project - Visual Studio Code``."""
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) >= _MIN_TITLE_PARTS and parts[-1] in _EDITOR_TITLES:
        return parts[-2]
    return None


@dataclass(frozen=True)
class ResultRow:
    """What the result list needs to draw one hit, Explorer-style."""

    name: str
    detail: str  # symbol, page or line, and "+N more", shown after the name
    path: str
    project: str
    snippet: str
    tag: str
    modified: str  # "2026-09-28", or "" when unknown
    size: str  # "12 KB", or "" when the file cannot be read
    is_image: bool  # a whole-image hit, drawn with its thumbnail
    relevance: int = 0  # 0-100, drawn as a thin bar
    weak: bool = False  # a less relevant match, drawn greyed
    divider_above: bool = False  # the first weak row carries the "Less relevant" heading

    @property
    def meta(self) -> str:
        """Right-hand column: modified date and size."""
        return "  ·  ".join(part for part in (self.modified, self.size) if part)


def format_size(size: int) -> str:
    value = float(size)
    for unit in _SIZE_UNITS[:-1]:
        if value < _SIZE_STEP:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= _SIZE_STEP
    return f"{value:.1f} {_SIZE_UNITS[-1]}"


def result_row(result: SearchResult, stat: Callable[[str], os.stat_result] = os.stat) -> ResultRow:
    symbol = f"  ·  {result.symbol}" if result.symbol and result.symbol != "<module>" else ""
    where = f"  ({result.location})" if result.location else ""
    more = f"  +{result.extra_hits} more" if result.extra_hits else ""
    try:
        size = format_size(stat(result.path).st_size)
    except OSError:  # moved or deleted since indexing; the row still opens a useful error
        size = ""
    modified = time.strftime("%Y-%m-%d", time.localtime(result.mtime)) if result.mtime else ""
    return ResultRow(
        name=Path(result.path).name,
        detail=f"{symbol}{where}{more}".strip(),
        path=result.path,
        project=result.project,
        snippet=result.snippet[:_SNIPPET_LINE],
        tag=_TAGS.get(result.kind, result.kind.upper()),
        modified=modified,
        size=size,
        is_image=result.kind == "image" and not result.page,
        relevance=result.relevance,
        weak=result.weak,
    )


def result_rows(
    results: list[SearchResult],
    stat: Callable[[str], os.stat_result] = os.stat,
    *,
    divide_weak: bool = True,
) -> list[ResultRow]:
    """One row per result, the first weak one headed "Less relevant" (results come strong first).

    The heading is drawn inside that row, not as a row of its own, so list rows stay one-to-one
    with results.
    """
    rows = [result_row(result, stat) for result in results]
    first_weak = next((i for i, row in enumerate(rows) if row.weak), None) if divide_weak else None
    if first_weak:  # not when every result is weak: there is nothing to set them apart from
        rows[first_weak] = replace(rows[first_weak], divider_above=True)
    return rows


class Launcher:
    """Open results in the default app, Explorer or VS Code. OS calls are injectable."""

    def __init__(
        self,
        startfile: Callable[[str], None] | None = None,
        popen: Callable[..., object] = subprocess.Popen,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._startfile = startfile or os.startfile  # type: ignore[attr-defined,unused-ignore]
        self._popen = popen
        self._which = which

    def open_file(self, result: SearchResult) -> None:
        self._startfile(result.path)

    def open_path(self, path: str) -> None:
        """Open a file in its default app (a PDF in the PDF reader, and so on)."""
        self._startfile(path)

    def reveal(self, result: SearchResult) -> None:
        self._popen(["explorer", "/select,", os.path.normpath(result.path)])

    def open_in_editor(self, result: SearchResult) -> None:
        self.open_at(result.path, result.start_line)

    def open_at(self, path: str, line: int = 0) -> None:
        """Open ``path`` in VS Code at ``line`` (default app if ``code`` is not installed)."""
        code = self._which("code")
        if code is None:
            self._startfile(path)
            return
        target = f"{path}:{line}" if line else path
        self._popen([code, "-g", target], creationflags=_NO_WINDOW)


@dataclass(frozen=True)
class SearchOutcome:
    results: list[SearchResult]
    milliseconds: float
    message: str = ""


class SearchService:
    """Lazily builds the search skill (so the window appears instantly) and runs queries."""

    def __init__(self, context_factory: Callable[[], SkillContext]) -> None:
        self._factory = context_factory
        self._skill: SearchSkill | None = None

    @property
    def skill(self) -> SearchSkill:
        if self._skill is None:
            self._skill = SearchSkill(self._factory())
        return self._skill

    def reset(self) -> None:
        """Forget the built skill so the next query uses the current settings."""
        self._skill = None

    def release(self) -> None:
        """Unload the embedding model (the app is quitting). Does nothing if never loaded."""
        if self._skill is None:
            return
        unload = getattr(self._skill.ctx.embedder, "unload_embedder", None)
        if callable(unload):
            try:
                unload()
            except ProviderError:
                logger.debug("search: embedder unload failed", exc_info=True)

    def warm(self) -> None:
        self.skill.warm()

    def on_battery(self) -> bool:
        """Whether queries run on the CPU right now; ``False`` until the skill exists.

        Called from the UI thread for a cosmetic status line, so it never builds the context.
        """
        return self._skill.ctx.query_on_cpu() if self._skill is not None else False

    def _with_indexed_at(self, results: list[SearchResult]) -> list[SearchResult]:
        """Stamp each hit with when it was indexed (one batched lookup) for "Date indexed" sort."""
        stamps = self.skill.ctx.state.manifest_indexed_at(r.path for r in results)
        return [replace(r, indexed_at=stamps.get(os.path.normcase(r.path), 0.0)) for r in results]

    def recent_queries(self) -> list[str]:
        return self.skill.ctx.state.recent_searches()

    def record_query(self, query: str) -> None:
        self.skill.ctx.state.record_search(query)

    def clear_history(self) -> None:
        self.skill.ctx.state.clear_search_history()

    def search(self, query: str, project: str | None) -> SearchOutcome:
        started = time.perf_counter()
        try:
            results = self._with_indexed_at(self.skill.search(query, current_project=project))
            message = ""
        except Exception as exc:  # search disabled, model server down, empty index ...
            results, message = [], f"{type(exc).__name__}: {exc}"
        return SearchOutcome(results, (time.perf_counter() - started) * 1000, message)
