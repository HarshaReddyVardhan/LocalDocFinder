"""Manual start and pause of indexing, for the tray menu and the Settings window.

Nothing is lost by pausing: the queue and the list of indexed files live in the state database,
the worker commits after every batch of files, and a worker that is told to stop finishes
cleanly. Starting again continues from the queue; only the batch that was in progress when it
stopped is redone.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from localdoc_finder.core.process import clear_stop_request, request_stop, single_instance
from localdoc_finder.core.store.lance import reset_index
from localdoc_finder.core.store.sqlite import INDEXING_PAUSED_KEY, StateDb

logger = logging.getLogger(__name__)

WORKER_LOCK = "worker"  # the lock localdoc_finder.worker holds while it runs
REBUILD_STOP_WAIT_SECONDS = 20.0
_POLL_SECONDS = 0.5


class Launcher(Protocol):
    """Starts a worker; ``now`` skips the idle wait and debounce."""

    def start(self, reconcile: bool, now: bool = False) -> object: ...


@dataclass(frozen=True)
class IndexingStatus:
    indexed: int  # files in the index
    waiting: int  # files queued for indexing
    running: bool
    paused: bool

    @property
    def summary(self) -> str:
        if self.paused:
            state = "paused" + (" (finishing the current files)" if self.running else "")
        elif self.running:
            state = "indexing"
        else:
            state = "idle" if not self.waiting else "waiting for the PC to be idle and plugged in"
        return f"Indexing: {state}. {self.indexed} files indexed, {self.waiting} waiting."

    @property
    def tray_summary(self) -> str:
        """A short status line for the tray menu, which sizes itself to its widest entry."""
        if self.paused:
            state = "paused"
        elif self.running:
            state = "indexing"
        elif self.waiting:
            state = "waiting to index"
        else:
            state = "idle"
        return f"Indexing: {state} ({self.indexed} files)"

    @property
    def fraction(self) -> float:
        total = self.indexed + self.waiting
        return self.indexed / total if total else 1.0


@dataclass(frozen=True)
class StartResult:
    started: bool
    message: str


class IndexingControl:
    def __init__(
        self,
        state: StateDb,
        data_dir: Path,
        launcher: Launcher,
        on_ac_power: Callable[[], bool],
        require_ac_power: bool = True,
    ) -> None:
        self._state = state
        self._data_dir = data_dir
        self._launcher = launcher
        self._on_ac = on_ac_power
        self._require_ac = require_ac_power

    def is_paused(self) -> bool:
        return self._state.get_meta(INDEXING_PAUSED_KEY) == "1"

    def is_running(self) -> bool:
        """A worker holds its lock while it runs; try to take it to find out."""
        with single_instance(WORKER_LOCK, self._data_dir) as acquired:
            return not acquired

    def status(self) -> IndexingStatus:
        return IndexingStatus(
            indexed=self._state.manifest_count(),
            waiting=self._state.queue_size(),
            running=self.is_running(),
            paused=self.is_paused(),
        )

    def pause(self) -> str:
        """Stop after the current batch and stay stopped (the watcher will not restart it)."""
        self._state.set_meta(INDEXING_PAUSED_KEY, "1")
        return "Pausing: the files being processed are finished and saved; nothing is lost."

    def start(self) -> StartResult:
        """Resume and run now, without waiting for the PC to be idle. Never on battery."""
        running = self.is_running()
        if running and self.is_paused():
            return StartResult(False, "Indexing is still stopping; try again in a few seconds.")
        if running:
            return StartResult(False, "Indexing is already running.")
        if self._require_ac and not self._on_ac():
            return StartResult(False, "Indexing only runs while the PC is plugged in.")
        self._state.delete_meta(INDEXING_PAUSED_KEY)
        self._launcher.start(True, now=True)
        logger.info("indexing started by the user")
        return StartResult(True, "Indexing started. It continues where it left off.")

    def rebuild(self, wait_seconds: float = REBUILD_STOP_WAIT_SECONDS) -> StartResult:
        """Delete the index and index every file again from scratch.

        A running worker is asked to stop first (it saves its batch); if it does not stop in
        time nothing is deleted. The files on disk are never touched.
        """
        if self.is_running():
            request_stop(self._data_dir)
            deadline = time.monotonic() + wait_seconds
            try:
                while self.is_running() and time.monotonic() < deadline:
                    time.sleep(_POLL_SECONDS)
            finally:
                clear_stop_request(self._data_dir)
            if self.is_running():
                return StartResult(False, "Indexing did not stop in time; try again in a moment.")
        reset_index(self._data_dir, self._state)
        logger.info("index deleted by the user; rebuilding from scratch")
        self._state.delete_meta(INDEXING_PAUSED_KEY)
        if self._require_ac and not self._on_ac():
            return StartResult(
                True, "Index cleared. Indexing starts again once the PC is plugged in."
            )
        self._launcher.start(True, now=True)
        return StartResult(True, "Index cleared. Re-indexing every file from scratch.")
