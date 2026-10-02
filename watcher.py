"""Always-on watcher (light: no torch, no lancedb). Records changes in the SQLite queue and spawns
worker.py when the machine is on AC power, settled, and idle.

    pythonw watcher.py            # normally started at logon by Task Scheduler (see install_task.ps1)
    python watcher.py --status    # print queue / power / index state
"""
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import indexer_config as cfg
import power
from projects import Projects
from store import State

log = logging.getLogger("watcher")
TICK_SECONDS = 10
_BELOW_NORMAL = 0x00004000
_NO_WINDOW = 0x08000000


def quick_reject(path: str) -> bool:
    """Cheapest possible filter for the flood of events from node_modules, AppData, caches..."""
    for part in path.lower().split(os.sep)[:-1]:
        if part in cfg.BLOCKED_DIRS:
            return True
    return False


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, state: State, projects: Projects):
        self.state = state
        self.projects = projects

    # -- helpers -------------------------------------------------------------
    def _upsert(self, path: str) -> None:
        if quick_reject(path):
            return
        if cfg.is_valid_file(path, is_ignored=self.projects.is_ignored):
            # Re-enqueueing resets the debounce timer, so a file being saved repeatedly is
            # processed once, FILE_DEBOUNCE_SECONDS after the last write.
            self.state.enqueue(path, "upsert", priority=time.time(), delay=cfg.FILE_DEBOUNCE_SECONDS)

    def _delete(self, path: str) -> None:
        if self.state.manifest_get(path) is not None:
            self.state.enqueue(path, "delete", priority=time.time(), delay=0)

    def _delete_tree(self, d: str) -> None:
        for p in self.state.manifest_under(d):
            self.state.enqueue(p, "delete", priority=time.time(), delay=0)

    def _enqueue_tree(self, d: str) -> None:
        try:
            for p in self.projects.iter_files([d]):
                self.state.enqueue(p, "upsert", priority=time.time(), delay=cfg.FILE_DEBOUNCE_SECONDS)
        except Exception:
            log.exception("tree enqueue failed for %s", d)

    # -- watchdog callbacks --------------------------------------------------
    def on_created(self, e):
        if e.is_directory:
            threading.Thread(target=self._enqueue_tree, args=(e.src_path,), daemon=True).start()
        else:
            self._upsert(e.src_path)

    def on_modified(self, e):
        if not e.is_directory:
            self._upsert(e.src_path)

    def on_deleted(self, e):
        if e.is_directory:
            self._delete_tree(e.src_path)
        else:
            self._delete(e.src_path)

    def on_moved(self, e):
        if e.is_directory:
            self._delete_tree(e.src_path)
            threading.Thread(target=self._enqueue_tree, args=(e.dest_path,), daemon=True).start()
        else:
            self._delete(e.src_path)
            self._upsert(e.dest_path)


def unload_model(model: str) -> None:
    """keep_alive=0 straight over HTTP (avoids importing the ollama package in the watcher)."""
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/embed", method="POST",
            data=json.dumps({"model": model, "input": "x", "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


class Watcher:
    def __init__(self, data_dir: Path = None, roots=None, worker_cmd=None):
        self.data_dir = Path(data_dir or cfg.DATA_DIR)
        self.state = State(self.data_dir)
        self.projects = Projects()
        self.gate = power.IdleGate()
        self.roots = [r for r in (roots or cfg.WATCH_ROOTS) if os.path.isdir(r)]
        self.observer = Observer()
        self.proc: Optional[subprocess.Popen] = None
        self.worker_cmd = worker_cmd or [sys.executable, str(Path(__file__).with_name("worker.py"))]
        self.next_spawn = 0.0
        self.last_reason = ""
        self.unplugged_at: Optional[float] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ run
    def start_observers(self) -> None:
        handler = ChangeHandler(self.state, self.projects)
        for r in self.roots:
            self.observer.schedule(handler, r, recursive=True)
            log.info("watching %s", r)
        self.observer.start()

    def stop(self) -> None:
        self._stop.set()

    def reconcile_due(self) -> bool:
        last = float(self.state.get_meta("last_reconcile", "0") or 0)
        return time.time() - last > cfg.RECONCILE_INTERVAL_HOURS * 3600

    def spawn_worker(self, reconcile: bool) -> None:
        cmd = list(self.worker_cmd) + (["--reconcile"] if reconcile else [])
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        out = open(log_dir / "worker.out.log", "a", encoding="utf-8")
        log.info("starting worker: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=out, cwd=str(Path(__file__).parent),
                                     creationflags=_BELOW_NORMAL | _NO_WINDOW)

    def tick(self) -> None:
        """One scheduling decision; called every TICK_SECONDS."""
        self.gate.update()
        on_ac = self.gate.on_ac

        # Worker supervision. The worker checks power itself before every file/batch; this is the
        # backstop if it is stuck inside a long extraction when the charger is pulled.
        if self.proc is not None:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                log.info("worker exited (%s)", code)
                self.proc = None
                if code not in (0, None):
                    self.next_spawn = time.time() + 600  # back off after failures (e.g. Ollama down)
            else:
                if not on_ac and cfg.REQUIRE_AC_POWER:
                    self.unplugged_at = self.unplugged_at or time.time()
                    if time.time() - self.unplugged_at > 20:
                        log.warning("unplugged: terminating worker")
                        self.proc.terminate()
                        unload_model(cfg.EMBED_MODEL)
                else:
                    self.unplugged_at = None
                return
        self.unplugged_at = None

        ready, why = self.gate.ready()
        if why != self.last_reason:
            self.last_reason = why
            log.info("gate: %s (queued=%d)", why or "ready", self.state.queue_size())
        if not ready or time.time() < self.next_spawn:
            return
        recon = self.reconcile_due()
        if recon or self.state.queue_size(due_only=True) > 0:
            self.spawn_worker(recon)
            if recon:  # the worker records success; don't re-trigger while it runs
                self.next_spawn = time.time() + 300

    def run(self) -> None:
        self.start_observers()
        try:
            while not self._stop.wait(TICK_SECONDS):
                try:
                    self.tick()
                except Exception:
                    log.exception("tick failed")
        finally:
            self.observer.stop()
            self.observer.join(timeout=5)
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                unload_model(cfg.EMBED_MODEL)


def status(data_dir: Path) -> None:
    st = State(data_dir)
    print(f"data dir        : {data_dir}")
    print(f"model           : {st.get_meta('model_id')} (dim {st.get_meta('dim')})")
    print(f"indexed files   : {st.manifest_count()}")
    print(f"queue           : {st.queue_size()} total, {st.queue_size(due_only=True)} due")
    last = float(st.get_meta("last_reconcile", "0") or 0)
    print(f"last reconcile  : {time.ctime(last) if last else 'never'}")
    print(f"on AC power     : {power.on_ac_power()}")
    print(f"idle input (s)  : {power.seconds_since_input():.0f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--data-dir")
    ap.add_argument("--root", action="append", help="watch this directory instead of the defaults (testing)")
    a = ap.parse_args(argv)
    data_dir = Path(a.data_dir or cfg.DATA_DIR)
    if a.status:
        status(data_dir)
        return 0
    from store import single_instance
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_dir / "watcher.log", encoding="utf-8")]
    if sys.stderr:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers)
    with single_instance("watcher", data_dir) as got:
        if not got:
            log.info("watcher already running")
            return 0
        w = Watcher(data_dir, a.root)
        signal.signal(signal.SIGINT, lambda *_: w.stop())
        signal.signal(signal.SIGTERM, lambda *_: w.stop())
        w.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
