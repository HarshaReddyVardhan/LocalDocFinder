"""The local filesystem: the configured roots, pruned by the scope rules and ``.gitignore``."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from localdoc_finder.core.projects import Projects
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.sources.base import SOURCES


@SOURCES.register("filesystem")
class FilesystemSource:
    name = "filesystem"

    def __init__(self, projects: Projects, scope: ScopePolicy) -> None:
        self._projects = projects
        self._scope = scope

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._projects.roots

    def iter_files(self, roots: Iterable[str | Path] | None = None) -> Iterator[Path]:
        return self._projects.iter_files(roots)

    def owns(self, path: str) -> bool:
        return Path(path).is_absolute()

    def still_valid(self, path: str) -> bool:
        return Path(path).exists() and self._scope.is_valid_file(
            path, is_ignored=self._projects.is_ignored
        )
