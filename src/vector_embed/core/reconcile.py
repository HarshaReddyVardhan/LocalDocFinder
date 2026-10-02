"""Reconciliation scan: compare disk with the manifest and queue the differences.

Catches what the file watcher missed: reboots, crashes and watcher buffer overflows.
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vector_embed.core.projects import Projects
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

_DELETE_PRIORITY = 1e18  # deletes run before any upsert
_CHECK_EVERY_FILES = 200  # the stop check runs power and idle probes: not once per file
_CHECK_EVERY_SECONDS = 0.25
_FLUSH_EVERY = 500  # changed files queued per transaction; an interrupt then loses little


@dataclass(frozen=True)
class ReconcileResult:
    queued: int = 0
    deleted: int = 0
    interrupted: bool = False


class _Throttled:
    """Calls ``check`` at most every 200 files or 250 ms; it runs power and idle probes."""

    def __init__(self, check: Callable[[], bool] | None) -> None:
        self._check = check
        self._files = 0
        self._last = time.monotonic()

    def __call__(self) -> bool:
        if self._check is None:
            return False
        self._files += 1
        now = time.monotonic()
        if self._files < _CHECK_EVERY_FILES and now - self._last < _CHECK_EVERY_SECONDS:
            return False
        self._files, self._last = 0, now
        return self._check()


def reconcile(
    state: StateDb,
    projects: Projects,
    scope: ScopePolicy,
    roots: list[str] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> ReconcileResult:
    """Queue new/changed files (priority = mtime, newest first) and files that vanished."""
    scan_roots = [str(r) for r in roots] if roots else None
    manifest = state.manifest_all()
    seen: set[str] = set()
    upserts: list[tuple[str, str, float]] = []
    queued = 0
    should_stop = _Throttled(stop_check)
    for found in projects.iter_files(scan_roots):
        if should_stop():
            queued += state.enqueue_many(upserts)  # what was found so far is kept
            logger.info("reconcile interrupted after queueing %d files", queued)
            return ReconcileResult(queued=queued, interrupted=True)
        path = str(found)
        key = os.path.normcase(path)
        seen.add(key)
        try:
            info = found.stat()
        except OSError:
            continue
        if manifest.get(key) != (info.st_mtime_ns, info.st_size):
            upserts.append((path, "upsert", info.st_mtime))
            if len(upserts) >= _FLUSH_EVERY:  # keep progress: an interrupted scan loses nothing
                queued += state.enqueue_many(upserts)
                upserts = []
    queued += state.enqueue_many(upserts)

    prefixes = tuple(
        os.path.normcase(os.path.normpath(r)).rstrip("\\/") + os.sep
        for r in (scan_roots or [str(r) for r in projects.roots])
    )
    gone = [
        (path, "delete", _DELETE_PRIORITY)
        for path in manifest
        if path not in seen  # both sides are lower-cased keys
        and path.startswith(prefixes)
        and (
            not Path(path).exists() or not scope.is_valid_file(path, is_ignored=projects.is_ignored)
        )
    ]
    deleted = state.enqueue_many(gone)
    logger.info("reconcile: %d changed/new, %d gone", queued, deleted)
    return ReconcileResult(queued, deleted)
