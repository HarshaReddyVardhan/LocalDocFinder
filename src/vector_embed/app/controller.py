"""Qt-free logic behind the search window: project guessing, labels, launching, searching."""

import ctypes
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.search import SearchResult, SearchSkill

_EDITOR_TITLES = ("Visual Studio Code", "Cursor", "Windsurf")
_MIN_TITLE_PARTS = 3
_NO_WINDOW = 0x08000000
_TAGS = {"image": "IMG", "ai-note": "NOTE", "outline": "FILE"}
_SNIPPET_LINE = 110


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


def result_label(result: SearchResult) -> str:
    """Three-line list item: title, location, snippet."""
    name = Path(result.path).name
    symbol = f"  ·  {result.symbol}" if result.symbol and result.symbol != "<module>" else ""
    where = f"  ({result.location})" if result.location else ""
    more = f"  +{result.extra_hits} more" if result.extra_hits else ""
    tag = _TAGS.get(result.kind, result.kind.upper())
    project = f"{result.project}  " if result.project else ""
    head = f"[{tag}] {name}{symbol}{where}{more}"
    return f"{head}\n{project}{result.path}\n{result.snippet[:_SNIPPET_LINE]}"


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

    def warm(self) -> None:
        self.skill.warm()

    def on_battery(self) -> bool:
        return self.skill.ctx.power.search_on_cpu()

    def search(self, query: str, project: str | None) -> SearchOutcome:
        started = time.perf_counter()
        try:
            results = self.skill.search(query, current_project=project)
            message = ""
        except Exception as exc:  # search disabled, model server down, empty index ...
            results, message = [], f"{type(exc).__name__}: {exc}"
        return SearchOutcome(results, (time.perf_counter() - started) * 1000, message)
