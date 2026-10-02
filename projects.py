"""Project detection, per-project .gitignore handling and fast file listing."""
import os
import shutil
import subprocess
import time
from typing import Callable, Iterator, Optional

import pathspec

import indexer_config as cfg

_GITIGNORE_TTL = 30.0  # seconds before a cached .gitignore is re-stat'ed


class Projects:
    def __init__(self, watch_roots=None):
        roots = watch_roots if watch_roots is not None else cfg.WATCH_ROOTS
        self._stop = {os.path.normcase(os.path.normpath(r)) for r in roots}
        self._root_cache: dict = {}
        self._spec_cache: dict = {}  # dir -> (checked_at, mtime, spec|None)

    # -- project roots -------------------------------------------------------
    def _has_marker(self, d: str) -> bool:
        for m in cfg.PROJECT_MARKERS:
            if os.path.exists(os.path.join(d, m)):
                return True
        try:
            names = os.listdir(d)
        except OSError:
            return False
        return any(n.lower().endswith(".sln") for n in names)

    def _root_of(self, d: str) -> Optional[str]:
        key = os.path.normcase(d)
        if key in self._root_cache:
            return self._root_cache[key]
        result: Optional[str]
        if key in self._stop:
            result = None
        elif os.path.exists(os.path.join(d, ".git")):
            result = d
        else:
            parent = os.path.dirname(d)
            parent_root = self._root_of(parent) if parent and parent != d else None
            if parent_root and os.path.exists(os.path.join(parent_root, ".git")):
                result = parent_root
            elif self._has_marker(d):
                result = d
            else:
                result = parent_root
        self._root_cache[key] = result
        return result

    def project_root(self, path: str) -> Optional[str]:
        return self._root_of(os.path.dirname(os.path.normpath(path)))

    def project_name(self, path: str) -> str:
        root = self.project_root(path)
        return os.path.basename(root) if root else ""

    def is_git_root(self, d: str) -> bool:
        return os.path.isdir(os.path.join(d, ".git")) or os.path.isfile(os.path.join(d, ".git"))

    # -- .gitignore ----------------------------------------------------------
    def _spec_for(self, d: str) -> Optional[pathspec.PathSpec]:
        now = time.monotonic()
        cached = self._spec_cache.get(d)
        if cached and now - cached[0] < _GITIGNORE_TTL:
            return cached[2]
        gi = os.path.join(d, ".gitignore")
        try:
            mtime = os.stat(gi).st_mtime_ns
        except OSError:
            self._spec_cache[d] = (now, None, None)
            return None
        if cached and cached[1] == mtime:
            self._spec_cache[d] = (now, mtime, cached[2])
            return cached[2]
        try:
            with open(gi, encoding="utf-8", errors="replace") as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
        except OSError:
            spec = None
        self._spec_cache[d] = (now, mtime, spec)
        return spec

    def is_ignored(self, path: str, is_dir: bool = False) -> bool:
        """True if path is matched by any .gitignore between it and its project root."""
        path = os.path.normpath(path)
        root = self.project_root(path) if not is_dir else self._root_of(path)
        if not root:
            return False
        root_key = os.path.normcase(root)
        d = os.path.dirname(path)
        while os.path.normcase(d).startswith(root_key):
            spec = self._spec_for(d)
            if spec is not None:
                rel = os.path.relpath(path, d).replace("\\", "/")
                if is_dir:
                    rel += "/"
                if spec.match_file(rel):
                    return True
            if os.path.normcase(d) == root_key:
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return False

    # -- file listing --------------------------------------------------------
    def _git_files(self, root: str):
        """Yield (path, is_dir) from git, or None if git is unavailable/failed."""
        git = shutil.which("git")
        if not git:
            return None
        try:
            out = subprocess.run(
                [git, "-C", root, "ls-files", "-co", "--exclude-standard", "-z"],
                capture_output=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        entries = []
        for rel in out.stdout.decode("utf-8", "replace").split("\0"):
            if not rel:
                continue
            rel = rel.rstrip("/")
            entries.append(os.path.normpath(os.path.join(root, rel)))
        return entries

    def iter_files(self, roots=None, is_valid: Callable = None) -> Iterator[str]:
        """Yield indexable file paths under roots. Uses git in repos, pruned scandir elsewhere."""
        is_valid = is_valid or cfg.is_valid_file
        stack = [os.path.normpath(r) for r in (roots if roots is not None else cfg.WATCH_ROOTS)]
        stack.reverse()
        while stack:
            d = stack.pop()
            if self.is_git_root(d):
                entries = self._git_files(d)
                if entries is not None:
                    for p in entries:
                        if os.path.isdir(p):  # submodule / nested repo
                            if cfg.should_descend(p):
                                stack.append(p)
                        elif is_valid(p):
                            yield p
                    continue
            try:
                with os.scandir(d) as it:
                    entries = list(it)
            except OSError:
                continue
            subdirs = []
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if cfg.should_descend(e.path, is_ignored=lambda p: self.is_ignored(p, True)):
                            subdirs.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        if is_valid(e.path, is_ignored=self.is_ignored):
                            yield e.path
                except OSError:
                    continue
            stack.extend(reversed(subdirs))
