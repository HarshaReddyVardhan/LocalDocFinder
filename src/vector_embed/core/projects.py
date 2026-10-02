"""Project detection, per-project ``.gitignore`` handling and fast file listing."""

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import pathspec

from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import ScopeSettings

logger = logging.getLogger(__name__)

_GITIGNORE_TTL_SECONDS = 30.0  # how long a cached .gitignore is trusted before a re-stat
_GIT_TIMEOUT_SECONDS = 120
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

Clock = Callable[[], float]


def _key(path: Path) -> str:
    """Case-folded cache key; Windows paths are case-insensitive."""
    return os.path.normcase(str(path))


class Projects:
    """Finds project roots, honours ``.gitignore`` and lists indexable files.

    Roots that are watched (``roots``) are never themselves projects: a home folder or a
    drive root is a container, not something to tag every file with.
    """

    def __init__(
        self,
        scope: ScopePolicy,
        settings: ScopeSettings,
        roots: Iterable[str | Path] | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._scope = scope
        self._settings = settings
        self._roots = tuple(Path(r) for r in (roots if roots is not None else settings.roots))
        self._stop = {_key(r) for r in self._roots}
        self._clock = clock
        self._root_cache: dict[str, Path | None] = {}
        # dir -> (checked_at, .gitignore mtime_ns, compiled spec or None)
        self._spec_cache: dict[str, tuple[float, int | None, pathspec.GitIgnoreSpec | None]] = {}

    # ------------------------------------------------------------------ project roots
    def _has_marker(self, directory: Path) -> bool:
        if any((directory / marker).exists() for marker in self._settings.project_markers):
            return True
        try:
            return any(
                next(directory.glob(pattern), None) is not None
                for pattern in self._settings.project_marker_globs
            )
        except OSError:
            return False

    def _root_of(self, directory: Path) -> Path | None:
        key = _key(directory)
        if key in self._root_cache:
            return self._root_cache[key]
        result: Path | None
        if key in self._stop:
            result = None
        elif (directory / ".git").exists():
            result = directory
        else:
            parent = directory.parent
            parent_root = self._root_of(parent) if parent != directory else None
            if parent_root is not None and (parent_root / ".git").exists():
                result = parent_root  # nested folders belong to the enclosing repository
            elif self._has_marker(directory):
                result = directory
            else:
                result = parent_root
        self._root_cache[key] = result
        return result

    def project_root(self, path: str | Path) -> Path | None:
        """Root of the project containing ``path`` (a file), or ``None`` outside any project."""
        return self._root_of(Path(os.path.normpath(path)).parent)

    def project_name(self, path: str | Path) -> str:
        root = self.project_root(path)
        return root.name if root else ""

    @staticmethod
    def is_git_root(directory: str | Path) -> bool:
        return (Path(directory) / ".git").exists()  # a directory, or a file for worktrees

    # ------------------------------------------------------------------ .gitignore
    def _spec_for(self, directory: Path) -> pathspec.GitIgnoreSpec | None:
        now = self._clock()
        key = _key(directory)
        cached = self._spec_cache.get(key)
        if cached and now - cached[0] < _GITIGNORE_TTL_SECONDS:
            return cached[2]
        gitignore = directory / ".gitignore"
        try:
            mtime: int | None = gitignore.stat().st_mtime_ns
        except OSError:
            self._spec_cache[key] = (now, None, None)
            return None
        if cached and cached[1] == mtime:
            self._spec_cache[key] = (now, mtime, cached[2])
            return cached[2]
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
            spec: pathspec.GitIgnoreSpec | None = pathspec.GitIgnoreSpec.from_lines(lines)
        except OSError:
            spec = None
        self._spec_cache[key] = (now, mtime, spec)
        return spec

    def is_ignored(self, path: str | Path, is_dir: bool = False) -> bool:
        """True if any ``.gitignore`` between ``path`` and its project root matches it."""
        target = Path(os.path.normpath(path))
        root = self._root_of(target) if is_dir else self.project_root(target)
        if root is None:
            return False
        root_key = _key(root)
        directory = target.parent
        while _key(directory).startswith(root_key):
            spec = self._spec_for(directory)
            if spec is not None:
                relative = target.relative_to(directory).as_posix() + ("/" if is_dir else "")
                if spec.match_file(relative):
                    return True
            if _key(directory) == root_key or directory.parent == directory:
                break
            directory = directory.parent
        return False

    # ------------------------------------------------------------------ file listing
    @staticmethod
    def _git_files(root: Path) -> list[Path] | None:
        """Tracked + untracked-but-not-ignored paths per git, or ``None`` if git is unusable."""
        git = shutil.which("git")
        if not git:
            return None
        try:
            out = subprocess.run(  # noqa: S603  # fixed argv, git resolved via which()
                [git, "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
                capture_output=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                creationflags=_NO_WINDOW,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            logger.debug("projects: git ls-files failed in %s", root, exc_info=True)
            return None
        if out.returncode != 0:
            return None
        names = out.stdout.decode("utf-8", "replace").split("\0")
        return [Path(os.path.normpath(root / name.rstrip("/"))) for name in names if name]

    def iter_files(self, roots: Iterable[str | Path] | None = None) -> Iterator[Path]:
        """Yield indexable files under ``roots``: git's view inside repos, pruned scandir elsewhere."""
        start = self._roots if roots is None else tuple(Path(r) for r in roots)
        stack = [Path(os.path.normpath(r)) for r in reversed(start)]
        while stack:
            directory = stack.pop()
            if self.is_git_root(directory):
                entries = self._git_files(directory)
                if entries is not None:
                    yield from self._walk_git_entries(entries, stack)
                    continue
            yield from self._walk_directory(directory, stack)

    def _walk_git_entries(self, entries: list[Path], stack: list[Path]) -> Iterator[Path]:
        for path in entries:
            if path.is_dir():  # submodule or nested repository
                if self._scope.should_descend(path):
                    stack.append(path)
            elif self._scope.is_valid_file(path):
                yield path

    def _walk_directory(self, directory: Path, stack: list[Path]) -> Iterator[Path]:
        try:
            with os.scandir(directory) as scan:
                entries = list(scan)
        except OSError:
            return
        subdirs: list[Path] = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if self._scope.should_descend(
                        entry.path, is_ignored=lambda p: self.is_ignored(p, is_dir=True)
                    ):
                        subdirs.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and self._scope.is_valid_file(
                    entry.path, is_ignored=self.is_ignored
                ):
                    yield Path(entry.path)
            except OSError:
                continue
        stack.extend(reversed(subdirs))
