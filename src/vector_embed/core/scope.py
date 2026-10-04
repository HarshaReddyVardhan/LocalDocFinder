"""Scope rules: which directories are walked and which files are indexed.

``is_valid_file`` checks, cheapest first, and the secrets denylist always runs first:

1. secrets denylist (names and path globs); 2. noise (lockfiles, bundles, maps);
3. extension / exact filename; 4. path inspection (blocked dirs, hidden dirs with the AI-note
allowlist, the app's own data folder); 5. ``.gitignore`` callback; 6. one ``stat`` for cloud
placeholders and size limits. Content-based checks (generated files) live in ``looks_generated``
because they need the file's text.
"""

import fnmatch
import logging
import os
import re
import stat
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path

from vector_embed.core.file_kinds import FileKind
from vector_embed.core.protection import SystemProtection
from vector_embed.core.settings import ScopeSettings

logger = logging.getLogger(__name__)

IgnoreCheck = Callable[[str], bool]

_ATTR_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_ATTR_OFFLINE = getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
_ATTR_RECALL_ON_OPEN = 0x40000
_ATTR_RECALL_ON_DATA_ACCESS = 0x400000
# Reading a placeholder makes OneDrive download it, so these files are skipped.
_ATTR_CLOUD_PLACEHOLDER = _ATTR_OFFLINE | _ATTR_RECALL_ON_OPEN | _ATTR_RECALL_ON_DATA_ACCESS

_BYTES_PER_MB = 1024 * 1024
_GENERATED_HEADER_LINES = 5


class AiNoteSource(StrEnum):
    """Kind of AI-tool note a path belongs to."""

    CLAUDE_PLAN = "claude-plan"
    CLAUDE_MEMORY = "claude-memory"
    AGENT_RULES = "agent-rules"


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a ``/``-separated glob (``*``, ``?``, ``**``) to an anchored regex."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        else:
            char = pattern[i]
            out.append("[^/]*" if char == "*" else "[^/]" if char == "?" else re.escape(char))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _normalised(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(Path(path).absolute()))


def _lower_parts(path: str | Path) -> tuple[str, ...]:
    return tuple(part.lower() for part in Path(path).parts)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def is_hidden_name(name: str) -> bool:
    """Dot-prefixed names, excluding the ``.`` / ``..`` pseudo entries."""
    return name.startswith(".") and name not in {".", ".."}


def is_reparse_point(path: str | Path) -> bool:
    """True for symlinks and Windows junctions; an unreadable path counts as one (skip it)."""
    try:
        info = os.lstat(path)
    except OSError:
        return True
    return bool(getattr(info, "st_file_attributes", 0) & _ATTR_REPARSE_POINT) or stat.S_ISLNK(
        info.st_mode
    )


def is_cloud_placeholder(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _ATTR_CLOUD_PLACEHOLDER)


class ScopePolicy:
    """Decides what the indexer may touch. Build once; methods are pure apart from ``stat``."""

    def __init__(
        self,
        settings: ScopeSettings,
        blocked_roots: Iterable[str | Path] = (),
        scan_roots: Iterable[str | Path] | None = None,
        protection: SystemProtection | None = None,
    ) -> None:
        self._s = settings
        self._protection = protection or SystemProtection()
        self._ai_keys = tuple(settings.ai_note_dirs)
        self._ai_globs = {
            key: tuple(glob_to_regex(glob) for glob in globs)
            for key, globs in settings.ai_note_dirs.items()
        }
        self._transcript_re = glob_to_regex(settings.ai_transcript_glob)
        self._secret_path_res = tuple(glob_to_regex(glob) for glob in settings.secret_path_globs)
        self._kind_by_ext = {ext: kind for kind, exts in settings.kind_exts.items() for ext in exts}
        self._kinds = settings.enabled_kinds()
        # Never index the app's own data folder (index, queue, logs, chat history).
        self._blocked_roots = tuple(_normalised(root) for root in blocked_roots)
        # The folders the user chose to index: a blocked name only counts *below* one of them.
        self._scan_roots = tuple(
            _lower_parts(_normalised(root))
            for root in (settings.roots if scan_roots is None else scan_roots)
        )

    # ------------------------------------------------------------------ secrets / noise
    def is_secret(self, path: str | Path) -> bool:
        """True for credentials and key material. Also used to keep files off cloud requests."""
        name = Path(path).name.lower()
        if name in self._s.secret_name_exceptions:
            return False
        if _matches_any(name, self._s.secret_name_patterns):
            return True
        posix = Path(path).as_posix().lower()
        return any(regex.match(posix) for regex in self._secret_path_res)

    def is_noise(self, name: str) -> bool:
        return _matches_any(name.lower(), self._s.noise_name_patterns)

    # ------------------------------------------------------------------ AI-tool notes
    def _ai_key(self, dir_name: str) -> str | None:
        """The ``ai_note_dirs`` key matching a lowercase directory name, if any."""
        return next((k for k in self._ai_keys if fnmatch.fnmatchcase(dir_name, k)), None)

    def ai_note_source(self, path: str | Path) -> AiNoteSource | None:
        """Classify AI-tool notes (plans, memory, agent rules); ``None`` for anything else."""
        parts = _lower_parts(path)
        for index, part in enumerate(parts[:-1]):
            if self._ai_key(part) is None:
                continue
            relative = parts[index + 1 :]
            if part == ".claude":
                if "plans" in relative[:-1]:
                    return AiNoteSource.CLAUDE_PLAN
                if "memory" in relative[:-1]:
                    return AiNoteSource.CLAUDE_MEMORY
            return AiNoteSource.AGENT_RULES
        if parts and parts[-1] in self._s.agent_rule_files:
            return AiNoteSource.AGENT_RULES
        return None

    # ------------------------------------------------------------------ file kinds
    def file_kind(self, path: str | Path) -> FileKind | None:
        """The kind a file belongs to; extension-less config files (``Dockerfile``) are code."""
        name = Path(path).name.lower()
        kind = self._kind_by_ext.get(Path(name).suffix)
        if kind is None and name in self._s.text_filenames:
            return FileKind.CODE
        return kind

    # ------------------------------------------------------------------ directories
    def _in_blocked_root(self, path: str | Path) -> bool:
        if not self._blocked_roots:
            return False
        candidate = _normalised(path)
        return any(
            candidate == root or candidate.startswith(root + os.sep) for root in self._blocked_roots
        )

    def should_descend(self, dir_path: str | Path, is_ignored: IgnoreCheck | None = None) -> bool:
        """Whether a directory walk should enter ``dir_path`` (prune early; it is the fast path)."""
        name = Path(os.path.normpath(dir_path)).name.lower()
        if self._protection.is_protected(dir_path):  # the OS and other profiles: never, anywhere
            return False
        if not name:  # drive root
            return True
        ai_key = self._ai_key(name)
        if name in self._s.blocked_dirs or (is_hidden_name(name) and ai_key is None):
            return False
        if name in self._s.ai_excluded_subdirs and self._has_ai_ancestor(dir_path):
            return False
        if self._in_blocked_root(dir_path) or is_reparse_point(dir_path):
            return False
        return not (is_ignored is not None and is_ignored(str(dir_path)))

    def _has_ai_ancestor(self, path: str | Path) -> bool:
        return any(self._ai_key(part) is not None for part in _lower_parts(path)[:-1])

    # ------------------------------------------------------------------ files
    def is_valid_file(self, file_path: str | Path, is_ignored: IgnoreCheck | None = None) -> bool:
        """Whether ``file_path`` should be indexed. Never raises; unreadable means no."""
        try:
            return self._is_valid_file(Path(file_path), is_ignored)
        except OSError:
            logger.debug("scope: cannot stat %s", file_path, exc_info=True)
            return False

    def _is_valid_file(self, path: Path, is_ignored: IgnoreCheck | None) -> bool:
        name = path.name.lower()
        ext = path.suffix.lower()
        cfg = self._s

        if self.is_secret(path) or self.is_noise(name):
            return False

        is_transcript_candidate = cfg.index_ai_transcripts and ext == ".jsonl"
        if self.file_kind(path) not in self._kinds and not is_transcript_candidate:
            return False
        is_doc = ext in cfg.doc_exts
        is_image = ext in cfg.image_exts

        if (
            self._protection.is_protected(path)
            or not self._path_allowed(path, is_transcript_candidate)
            or self._in_blocked_root(path)
        ):
            return False
        ignored = is_ignored is not None and is_ignored(str(path))
        # A link can point anywhere, including at a file we must not read.
        if ignored or path.is_symlink():
            return False
        info = path.stat()
        if info.st_size == 0 or is_cloud_placeholder(info):
            return False
        limit_mb = (
            cfg.max_doc_size_mb
            if is_doc
            else cfg.max_image_size_mb
            if is_image
            else cfg.max_text_size_mb
        )
        return info.st_size / _BYTES_PER_MB <= limit_mb

    def _below_scan_root(self, path: Path) -> tuple[str, ...]:
        """The lower-case parts of ``path`` that lie inside the user's chosen root.

        A project kept under a folder called ``build`` or ``env`` must not be excluded because a
        folder *above* the root has a blocked name; only what is inside the root counts. A path
        that is under no configured root keeps all its parts.
        """
        absolute = _lower_parts(_normalised(path))
        for root in self._scan_roots:
            if root and absolute[: len(root)] == root:
                return absolute[len(root) :]
        return _lower_parts(path)

    def _path_allowed(self, path: Path, is_transcript_candidate: bool) -> bool:
        """Blocked dirs, hidden dirs and the AI-note allowlist; string checks only."""
        parts = _lower_parts(path)
        dirs = parts[:-1]
        for part in self._below_scan_root(path)[:-1]:
            if part in self._s.blocked_dirs:
                return False
        ai_index: int | None = None
        for index, part in enumerate(dirs):
            if is_hidden_name(part):
                if self._ai_key(part) is None:
                    return False
                if ai_index is None:
                    ai_index = index

        if ai_index is None:
            return not is_transcript_candidate  # transcripts only exist inside AI-tool dirs

        key = self._ai_key(dirs[ai_index])
        assert key is not None  # ai_index is only set for matching dirs
        relative = parts[ai_index + 1 :]
        if any(part in self._s.ai_excluded_subdirs for part in relative[:-1]):
            return False
        relative_posix = "/".join(relative)
        if any(regex.match(relative_posix) for regex in self._ai_globs[key]):
            return not is_transcript_candidate
        return (
            is_transcript_candidate
            and key == ".claude"
            and bool(self._transcript_re.match(relative_posix))
        )

    # ------------------------------------------------------------------ content heuristics
    def looks_generated(self, text: str) -> bool:
        """Heuristic for machine-written files: a marker in the header or very long lines."""
        if not text:
            return False
        lines = text.splitlines()
        header = "\n".join(lines[:_GENERATED_HEADER_LINES]).lower()
        if any(marker in header for marker in self._s.generated_markers):
            return True
        return len(text) / max(len(lines), 1) > self._s.max_avg_line_length
