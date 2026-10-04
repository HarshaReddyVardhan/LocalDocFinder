"""Always-on watcher: light (no ML, no LanceDB). Records changes in the SQLite queue and spawns
the worker when the machine is on AC power, settled and idle.

    pythonw -m localdoc_finder.watcher  # normally started at logon (see scripts/install_task.ps1)
    python -m localdoc_finder.watcher --status
"""

import argparse
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch

from localdoc_finder.core.idle import IdleGate
from localdoc_finder.core.logging_setup import configure_logging, rotate_if_large
from localdoc_finder.core.models.hardware import on_ac_power
from localdoc_finder.core.ollama_http import unload_model
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.process import self_command, single_instance, stop_requested
from localdoc_finder.core.projects import Projects
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.scope_roots import resolve_roots
from localdoc_finder.core.settings import Settings, SettingsError, load_settings
from localdoc_finder.core.store.sqlite import CHAT_LOCK, INDEXING_PAUSED_KEY, PROGRESS_KEY, StateDb
from localdoc_finder.core.wiring import build_projects, build_scope, log_dir

logger = logging.getLogger("watcher")

TICK_SECONDS = 10
_BELOW_NORMAL = 0x00004000
_NO_WINDOW = 0x08000000
_FAILURE_BACKOFF_SECONDS = 600
_WORKER_STOP_GRACE_SECONDS = 15.0
_RECONCILE_BACKOFF_SECONDS = 300
_MAX_EVENTS_PER_SECOND = 400  # beyond this, events are dropped and a reconcile picks them up
_WATCH_REFRESH_TICKS = 6  # re-list the top-level folders about once a minute
_TREE_QUEUE_SIZE = 64  # directories waiting to be walked
_TREE_IDLE_SECONDS = 5.0  # the tree-walking thread exits after this long with nothing to do
_NO_PROGRESS_BACKOFF_SECONDS = 120  # a worker that finishes without shrinking the queue
_UNPLUG_GRACE_SECONDS = 20


def quick_reject(path: str, blocked_dirs: frozenset[str]) -> bool:
    """Cheapest possible filter for the flood of events from node_modules, caches, ..."""
    return any(part in blocked_dirs for part in Path(path.lower()).parts[:-1])


class ChangeHandler(FileSystemEventHandler):
    """Turns file-system events into queue rows. Cheap: no hashing, no parsing."""

    def __init__(
        self, state: StateDb, projects: Projects, scope: ScopePolicy, settings: Settings
    ) -> None:
        self.state = state
        self.projects = projects
        self.scope = scope
        self._blocked = settings.scope.blocked_dirs
        self._debounce = settings.idle.file_debounce_seconds
        self._window_start = time.monotonic()
        self._window_events = 0
        self._dropped = 0
        self._trees: queue.Queue[str] = queue.Queue(maxsize=_TREE_QUEUE_SIZE)
        self._tree_worker: threading.Thread | None = None
        self._tree_lock = threading.Lock()

    def _admit(self) -> bool:
        """Rate limit: a storm of events (an unpacked archive, a build) is dropped, not queued.

        Dropped events are not lost for good: ``take_dropped`` tells the watcher to reconcile,
        which finds whatever changed by comparing the disk with the manifest.
        """
        now = time.monotonic()
        if now - self._window_start >= 1.0:
            self._window_start, self._window_events = now, 0
        self._window_events += 1
        if self._window_events > _MAX_EVENTS_PER_SECOND:
            self._dropped += 1
            return False
        return True

    def take_dropped(self) -> int:
        """How many events were dropped since the last call (and reset the count)."""
        dropped, self._dropped = self._dropped, 0
        return dropped

    def _upsert(self, path: str) -> None:
        if quick_reject(path, self._blocked) or not self._admit():
            return
        if self.scope.is_valid_file(path, is_ignored=self.projects.is_ignored):
            # Re-queueing resets the debounce timer, so a file saved repeatedly is processed once,
            # ``file_debounce_seconds`` after the last write.
            self.state.enqueue(path, "upsert", priority=time.time(), delay=self._debounce)

    def _delete(self, path: str) -> None:
        """A path vanished. On Windows a removed *directory* often arrives as a plain file-deleted
        event, so anything indexed below the path goes too."""
        if self.state.manifest_get(path) is not None:
            self.state.enqueue(path, "delete", priority=time.time(), delay=0)
        self._delete_tree(path)

    def _delete_tree(self, directory: str) -> None:
        for path in self.state.manifest_under(directory):
            self.state.enqueue(path, "delete", priority=time.time(), delay=0)

    def _enqueue_tree(self, directory: str) -> None:
        try:
            for path in self.projects.iter_files([directory]):
                self.state.enqueue(str(path), "upsert", priority=time.time(), delay=self._debounce)
        except Exception:  # a failing scan must not kill the watcher thread
            logger.exception("tree enqueue failed for %s", directory)

    def _enqueue_tree_async(self, directory: str) -> None:
        """Walk a directory that appeared (created or moved in): its files send no events.

        One bounded worker thread does all the walking. A directory the scope would not enter is
        skipped, and if too many pile up, an early reconcile picks up what was dropped.
        """
        if not self.scope.should_descend(directory, self.projects.is_ignored):
            return
        try:
            self._trees.put_nowait(directory)
        except queue.Full:
            logger.warning("watcher: tree backlog is full; scheduling an early reconcile")
            self.state.set_meta("last_reconcile", "0")
            return
        with self._tree_lock:
            if self._tree_worker is None or not self._tree_worker.is_alive():
                self._tree_worker = threading.Thread(target=self._walk_trees, daemon=True)
                self._tree_worker.start()

    def _walk_trees(self) -> None:
        while True:
            try:
                directory = self._trees.get(timeout=_TREE_IDLE_SECONDS)
            except queue.Empty:
                return  # idle: the thread ends, and a new one starts when needed
            self._enqueue_tree(directory)

    @staticmethod
    def _text(value: str | bytes) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else value

    def on_created(self, event: FileSystemEvent) -> None:
        path = self._text(event.src_path)
        if event.is_directory:
            self._enqueue_tree_async(path)
        else:
            self._upsert(path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._upsert(self._text(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        path = self._text(event.src_path)
        if event.is_directory:
            self._delete_tree(path)
        else:
            self._delete(path)

    def on_moved(self, event: FileSystemEvent) -> None:
        src, dest = self._text(event.src_path), self._text(event.dest_path)
        if event.is_directory:
            self._delete_tree(src)
            self._enqueue_tree_async(dest)
        else:
            self._delete(src)
            if Path(dest).is_dir():  # Windows may report a moved directory as a plain move
                self._enqueue_tree_async(dest)
            else:
                self._upsert(dest)


class WorkerHandle(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


class WorkerLauncher(Protocol):
    def start(self, reconcile: bool) -> WorkerHandle: ...


class SubprocessLauncher:
    """Starts ``python -m localdoc_finder.worker`` at below-normal priority, without a window."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir

    def start(self, reconcile: bool, now: bool = False) -> WorkerHandle:
        """``now`` skips the idle wait and debounce: the user asked for indexing to start."""
        command = self_command("worker")
        if reconcile:
            command.append("--reconcile")
        if now:
            command.append("--now")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        output = self._log_dir / "worker.out.log"
        rotate_if_large(output)  # the worker redirects its streams here, so it cannot rotate it
        with output.open("a", encoding="utf-8") as out:
            logger.info("starting worker: %s", " ".join(command))
            return subprocess.Popen(  # noqa: S603  # fixed argv, our own interpreter and module
                command,
                stdout=out,
                stderr=out,
                creationflags=_BELOW_NORMAL | _NO_WINDOW,
            )


class StartGate(Protocol):
    def update(self) -> None: ...

    @property
    def on_ac(self) -> bool: ...

    def ready(self, allow_battery: bool = False) -> tuple[bool, str]: ...


class Watcher:
    """Scheduling brain: one ``tick`` every ``TICK_SECONDS`` decides whether to start a worker."""

    def __init__(
        self,
        settings: Settings,
        state: StateDb,
        gate: StartGate,
        launcher: WorkerLauncher,
        unload: Callable[[], None],
        *,
        projects: Projects,
        scope: ScopePolicy,
        clock: Callable[[], float] = time.time,
        roots: Sequence[str] | None = None,
        reload_settings: Callable[[], Settings] | None = None,
    ) -> None:
        self._reload_settings = reload_settings
        self.settings = settings
        self.state = state
        self.gate = gate
        self.launcher = launcher
        self._unload = unload
        self.projects = projects
        self.scope = scope
        self._clock = clock
        self.roots = [r for r in (roots or resolve_roots(settings.scope)) if Path(r).is_dir()]
        self._scope_key = (tuple(self.roots), settings.scope.enabled_kinds())
        self.observer = Observer()
        self.handle: WorkerHandle | None = None
        self.next_spawn = 0.0
        self.last_reason = ""
        self.unplugged_at: float | None = None
        self._progress_at_start: str | None = None
        self._watches: dict[str, ObservedWatch] = {}
        self._ticks = 0
        self.handler = ChangeHandler(state, projects, scope, settings)
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def watch_targets(self) -> dict[str, bool]:
        """Folders to watch -> recursive.

        A root is watched for its own files, and each top-level folder the scope would enter is
        watched recursively. Folders it would not enter (AppData, hidden folders, ``node_modules``)
        are never watched, so their constant churn costs nothing.
        """
        targets: dict[str, bool] = {}
        for root in self.roots:
            try:
                children = [e.path for e in os.scandir(root) if e.is_dir(follow_symlinks=False)]
            except OSError:
                targets[root] = True  # cannot list it: fall back to watching it whole
                continue
            targets[root] = False
            for child in children:
                if self.scope.should_descend(child, self.projects.is_ignored):
                    targets[child] = True
        return targets

    def start_observers(self) -> None:
        self.refresh_watches()
        self.observer.start()

    def refresh_watches(self) -> None:
        """Start watching folders that appeared and stop watching ones that went away."""
        wanted = self.watch_targets()
        for path in [p for p in self._watches if p not in wanted]:
            self.observer.unschedule(self._watches.pop(path))
            logger.info("no longer watching %s", path)
        for path, recursive in wanted.items():
            if path not in self._watches:
                self._watches[path] = self.observer.schedule(
                    self.handler, path, recursive=recursive
                )
                logger.info("watching %s%s", path, " (recursive)" if recursive else "")

    def stop(self) -> None:
        self._stop.set()

    def request_startup_reconcile(self) -> None:
        """Changes made while the watcher was not running (a reboot, a crash) sent no events."""
        self.state.set_meta("last_reconcile", "0")

    def run(self) -> None:
        self.request_startup_reconcile()
        self.start_observers()
        try:
            while not self._stop.wait(TICK_SECONDS):
                try:
                    self.tick()
                except Exception:  # keep watching even if one scheduling pass fails
                    logger.exception("tick failed")
        finally:
            self.observer.stop()
            self.observer.join(timeout=5)
            self._stop_worker()

    def _stop_worker(self) -> None:
        """Let a running worker finish its batch and unload itself; kill it only if it hangs."""
        if self.handle is None or self.handle.poll() is not None:
            return
        deadline = time.monotonic() + _WORKER_STOP_GRACE_SECONDS
        while self.handle.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.handle.poll() is None:
            self.handle.terminate()
        self._unload()  # a hard kill skips the worker's own unload

    # ------------------------------------------------------------------ scheduling
    def reconcile_due(self) -> bool:
        last = float(self.state.get_meta("last_reconcile", "0") or 0)
        return self._clock() - last > self.settings.idle.reconcile_interval_hours * 3600

    def _supervise(self) -> bool:
        """Watch a running worker. Returns True while one is still running."""
        if self.handle is None:
            self.unplugged_at = None
            return False
        code = self.handle.poll()
        if code is not None:
            logger.info("worker exited", extra={"code": code})
            self.handle = None
            if code != 0:
                self.next_spawn = self._clock() + _FAILURE_BACKOFF_SECONDS  # e.g. Ollama is down
                self._unload()  # a crashed worker never reached its own unload
            elif self.state.get_meta(PROGRESS_KEY) == self._progress_at_start:
                # It ran and exited cleanly without finishing a single queue item (every file kept
                # failing, or the gates stopped it at once): do not start another straight away.
                self.next_spawn = self._clock() + _NO_PROGRESS_BACKOFF_SECONDS
            self.unplugged_at = None
            return False
        # The worker checks power itself before each batch; this is the backstop if it is stuck
        # inside a long extraction when the charger is pulled.
        if not self.gate.on_ac and self.settings.power.require_ac_power:
            self.unplugged_at = self.unplugged_at or self._clock()
            if self._clock() - self.unplugged_at > _UNPLUG_GRACE_SECONDS:
                logger.warning("unplugged: terminating worker")
                self.handle.terminate()
                self._unload()
        else:
            self.unplugged_at = None
        return True

    def _housekeeping(self) -> None:
        """New top-level folders get watched; a storm of dropped events schedules a reconcile."""
        self._ticks += 1
        if self.handler.take_dropped():
            logger.warning("watcher: events were dropped under load; scheduling a reconcile")
            self.state.set_meta("last_reconcile", "0")
        if self._ticks % _WATCH_REFRESH_TICKS == 0:
            self.reload_roots()
            self.refresh_watches()

    def reload_roots(self) -> bool:
        """Pick up folders added or removed in Settings; True if the roots changed.

        The scope, the project finder and the event handler were built from the old roots, so they
        are rebuilt, and a reconcile is scheduled: a folder that was just added is scanned, not
        waited on for events that will never come for files already there.
        """
        if self._reload_settings is None:
            return False
        try:
            fresh = self._reload_settings()
        except SettingsError:
            logger.warning("watcher: settings are unreadable; keeping the current folders")
            return False
        roots = [r for r in resolve_roots(fresh.scope) if Path(r).is_dir()]
        scope_key = (tuple(roots), fresh.scope.enabled_kinds())  # new file kinds need a re-scan
        if scope_key == self._scope_key:
            return False
        self._scope_key = scope_key
        logger.info("watcher: folders changed", extra={"roots": roots})
        self.settings = fresh
        self.roots = roots
        self.scope = build_scope(fresh)
        self.projects = build_projects(fresh, self.scope)
        self.handler = ChangeHandler(self.state, self.projects, self.scope, fresh)
        self.observer.unschedule_all()  # the old handler is gone: every watch is re-created
        self._watches.clear()
        self.state.set_meta("last_reconcile", "0")
        return True

    def tick(self) -> None:
        if stop_requested(self.settings.storage.data_dir):
            logger.info("stop requested")
            self.stop()
            return
        self._housekeeping()
        self.gate.update()
        if self._supervise():
            return
        ready, why = self.gate.ready()
        if why != self.last_reason:
            self.last_reason = why
            logger.info("gate: %s", why or "ready", extra={"queued": self.state.queue_size()})
        paused = self.state.get_meta(INDEXING_PAUSED_KEY) == "1"
        if not ready or paused or self._clock() < self.next_spawn:
            return
        reconcile = self.reconcile_due()
        if reconcile or self.state.queue_size(due_only=True) > 0:
            self._progress_at_start = self.state.get_meta(PROGRESS_KEY)
            self.handle = self.launcher.start(reconcile)
            if reconcile:  # the worker records success; do not re-trigger while it runs
                self.next_spawn = self._clock() + _RECONCILE_BACKOFF_SECONDS


def status(settings: Settings) -> str:
    """Human-readable state of the index and the queue."""
    with StateDb(settings.storage.data_dir) as state:
        last = float(state.get_meta("last_reconcile", "0") or 0)
        lines = [
            f"data dir        : {settings.storage.data_dir}",
            f"model           : {state.get_meta('model_id')} (dim {state.get_meta('dim')})",
            f"indexed files   : {state.manifest_count()}",
            f"queue           : {state.queue_size()} total, {state.queue_size(due_only=True)} due",
            f"last reconcile  : {time.ctime(last) if last else 'never'}",
            f"on AC power     : {on_ac_power()}",
        ]
    return "\n".join(lines)


def _current_embed_model(startup: Settings) -> str:
    """The embedder named in the settings file *now*: it may have been changed since startup."""
    try:
        return load_settings().embedding.model
    except SettingsError:
        return startup.embedding.model


def build_watcher(settings: Settings, state: StateDb) -> Watcher:
    scope = build_scope(settings)
    gate = IdleGate(
        PowerGate(settings.power),
        settings.idle,
        chat_active=lambda: state.lock_held(CHAT_LOCK),
    )
    return Watcher(
        settings,
        state,
        gate,
        SubprocessLauncher(log_dir(settings)),
        lambda: unload_model(settings.ollama_host, _current_embed_model(settings)),
        projects=build_projects(settings, scope),
        scope=scope,
        reload_settings=lambda: load_settings(settings.settings_path()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.status:
        sys.stdout.write(status(settings) + "\n")
        return 0
    configure_logging("watcher", log_dir(settings), settings.log_level)
    with single_instance("watcher", settings.storage.data_dir) as acquired:
        if not acquired:
            logger.info("watcher already running")
            return 0
        with StateDb(settings.storage.data_dir) as state:
            watcher = build_watcher(settings, state)
            signal.signal(signal.SIGINT, lambda *_: watcher.stop())
            signal.signal(signal.SIGTERM, lambda *_: watcher.stop())
            watcher.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
