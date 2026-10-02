"""Always-on watcher: light (no ML, no LanceDB). Records changes in the SQLite queue and spawns
the worker when the machine is on AC power, settled and idle.

    pythonw -m vector_embed.watcher     # normally started at logon (see scripts/install_task.ps1)
    python -m vector_embed.watcher --status
"""

import argparse
import logging
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

from vector_embed.core import runtime
from vector_embed.core.idle import IdleGate
from vector_embed.core.logging_setup import configure_logging
from vector_embed.core.models.hardware import on_ac_power
from vector_embed.core.ollama_http import unload_model
from vector_embed.core.power import PowerGate
from vector_embed.core.process import self_command, single_instance, stop_requested
from vector_embed.core.projects import Projects
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import Settings, load_settings
from vector_embed.core.store.sqlite import CHAT_LOCK, PROGRESS_KEY, StateDb

logger = logging.getLogger("watcher")

TICK_SECONDS = 10
_BELOW_NORMAL = 0x00004000
_NO_WINDOW = 0x08000000
_FAILURE_BACKOFF_SECONDS = 600
_WORKER_STOP_GRACE_SECONDS = 15.0
_RECONCILE_BACKOFF_SECONDS = 300
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

    def _upsert(self, path: str) -> None:
        if quick_reject(path, self._blocked):
            return
        if self.scope.is_valid_file(path, is_ignored=self.projects.is_ignored):
            # Re-queueing resets the debounce timer, so a file saved repeatedly is processed once,
            # ``file_debounce_seconds`` after the last write.
            self.state.enqueue(path, "upsert", priority=time.time(), delay=self._debounce)

    def _delete(self, path: str) -> None:
        if self.state.manifest_get(path) is not None:
            self.state.enqueue(path, "delete", priority=time.time(), delay=0)

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
        threading.Thread(target=self._enqueue_tree, args=(directory,), daemon=True).start()

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
            self._upsert(dest)


class WorkerHandle(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


class WorkerLauncher(Protocol):
    def start(self, reconcile: bool) -> WorkerHandle: ...


class SubprocessLauncher:
    """Starts ``python -m vector_embed.worker`` at below-normal priority, without a window."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir

    def start(self, reconcile: bool) -> WorkerHandle:
        command = self_command("worker")
        if reconcile:
            command.append("--reconcile")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        with (self._log_dir / "worker.out.log").open("a", encoding="utf-8") as out:
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
    ) -> None:
        self.settings = settings
        self.state = state
        self.gate = gate
        self.launcher = launcher
        self._unload = unload
        self.projects = projects
        self.scope = scope
        self._clock = clock
        self.roots = [r for r in (roots or settings.scope.roots) if Path(r).is_dir()]
        self.observer = Observer()
        self.handle: WorkerHandle | None = None
        self.next_spawn = 0.0
        self.last_reason = ""
        self.unplugged_at: float | None = None
        self._progress_at_start: str | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start_observers(self) -> None:
        handler = ChangeHandler(self.state, self.projects, self.scope, self.settings)
        for root in self.roots:
            self.observer.schedule(handler, root, recursive=True)
            logger.info("watching %s", root)
        self.observer.start()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
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

    def tick(self) -> None:
        if stop_requested(self.settings.storage.data_dir):
            logger.info("stop requested")
            self.stop()
            return
        self.gate.update()
        if self._supervise():
            return
        ready, why = self.gate.ready()
        if why != self.last_reason:
            self.last_reason = why
            logger.info("gate: %s", why or "ready", extra={"queued": self.state.queue_size()})
        if not ready or self._clock() < self.next_spawn:
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


def build_watcher(settings: Settings, state: StateDb) -> Watcher:
    scope = runtime.build_scope(settings)
    gate = IdleGate(
        PowerGate(settings.power),
        settings.idle,
        chat_active=lambda: state.lock_held(CHAT_LOCK),
    )
    return Watcher(
        settings,
        state,
        gate,
        SubprocessLauncher(runtime.log_dir(settings)),
        lambda: unload_model(settings.ollama_host, settings.embedding.model),
        projects=runtime.build_projects(settings, scope),
        scope=scope,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.status:
        sys.stdout.write(status(settings) + "\n")
        return 0
    configure_logging("watcher", runtime.log_dir(settings), settings.log_level)
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
