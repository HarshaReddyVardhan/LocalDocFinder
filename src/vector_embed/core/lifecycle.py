"""Install, update and uninstall chores: stop old processes, start the watcher, wipe the data.

Velopack runs the app exe with special flags at these moments; ``hooks`` wires them to the
functions here. Every chore logs and carries on: a failure must never block an install or update.
"""

import logging
import os
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import psutil

from vector_embed.core.autostart import Autostart
from vector_embed.core.process import self_command
from vector_embed.core.store.sqlite import STATE_FILENAME

logger = logging.getLogger(__name__)

APP_EXE_NAME = "VectorEmbed.exe"
CLI_EXE_NAME = "ve.exe"
DELETE_ATTEMPTS = 5
STOP_WAIT_SECONDS = 5.0
_DETACHED = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
_QUOTE = "'"


class DataDeletionRefusedError(RuntimeError):
    """The folder asked to be wiped does not look like our data folder."""


class ProcessLike(Protocol):
    @property
    def pid(self) -> int: ...

    def name(self) -> str: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


ProcessLister = Callable[[], Iterable[ProcessLike]]
WaitForExit = Callable[[list[ProcessLike], float], list[ProcessLike]]  # returns those still alive


def list_processes() -> Iterable[ProcessLike]:
    return psutil.process_iter()


def wait_for_exit(processes: list[ProcessLike], timeout: float) -> list[ProcessLike]:
    _, alive = psutil.wait_procs(processes, timeout=timeout)  # type: ignore[arg-type]  # same API
    return list(alive)


def stop_other_instances(
    *,
    exe_names: tuple[str, ...] = (APP_EXE_NAME, CLI_EXE_NAME),
    current_pid: int | None = None,
    lister: ProcessLister = list_processes,
    wait: WaitForExit = wait_for_exit,
) -> int:
    """Stop every other running copy of the app (tray, watcher, worker, ve); returns how many."""
    names = {name.lower() for name in exe_names}
    me = os.getpid() if current_pid is None else current_pid
    victims: list[ProcessLike] = []
    for process in lister():
        try:
            if process.pid != me and process.name().lower() in names:
                victims.append(process)
        except psutil.Error:  # the process ended or is not ours to inspect
            continue
    for process in victims:
        try:
            process.terminate()
        except psutil.Error:
            logger.debug("lifecycle: could not terminate pid %s", process.pid)
    for process in wait(victims, STOP_WAIT_SECONDS):
        try:
            process.kill()
        except psutil.Error:
            logger.warning("lifecycle: pid %s would not stop", process.pid)
    return len(victims)


def start_watcher(spawn: Callable[..., object] = subprocess.Popen) -> None:
    """Launch the watcher detached. It exits at once if another one already holds its lock."""
    try:
        spawn(
            self_command("watcher", windowless=True),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_DETACHED,
        )
    except OSError:
        logger.warning("lifecycle: could not start the watcher", exc_info=True)


def ensure_data_folder(data_dir: Path) -> None:
    if not (data_dir / STATE_FILENAME).is_file():
        raise DataDeletionRefusedError(
            f"{data_dir} does not look like the Vector Embed data folder"
        )


def schedule_data_deletion(
    data_dir: Path,
    pid: int | None = None,
    spawn: Callable[..., object] = subprocess.Popen,
) -> None:
    """Delete ``data_dir`` after this process exits (its database file is open until then).

    Refuses anything without ``state.db``: this must never be pointed at a user's own folder.
    """
    ensure_data_folder(data_dir)
    target = str(data_dir).replace(_QUOTE, _QUOTE * 2)
    script = (
        f"Wait-Process -Id {os.getpid() if pid is None else pid} -ErrorAction SilentlyContinue; "
        f"for ($i = 0; $i -lt {DELETE_ATTEMPTS}; $i++) {{ "
        "Start-Sleep -Seconds 1; "
        f"try {{ Remove-Item -LiteralPath {_QUOTE}{target}{_QUOTE} -Recurse -Force "
        "-ErrorAction Stop; break } catch { } }"
    )
    spawn(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_DETACHED,
    )


# ---------------------------------------------------------------------- Velopack hooks
def _safely(action: Callable[[], object], what: str) -> None:
    try:
        action()
    except Exception:  # hook boundary: an install or update must not fail because of a chore
        logger.exception("lifecycle: %s failed", what)


def after_install(autostart: Autostart, enabled: Callable[[], bool]) -> None:
    if enabled():
        _safely(autostart.register, "registering startup tasks")


def before_update() -> None:
    _safely(stop_other_instances, "stopping the old version")


def after_update(autostart: Autostart, enabled: Callable[[], bool]) -> None:
    if enabled():
        _safely(autostart.register, "re-registering startup tasks")
    _safely(start_watcher, "starting the watcher")


def before_uninstall(autostart: Autostart) -> None:
    _safely(autostart.unregister, "removing startup tasks")
    _safely(stop_other_instances, "stopping the app")


class StartupApp(Protocol):
    """``velopack.App``: a builder whose hooks take the version string."""

    def on_after_install_fast_callback(self, callback: Callable[[str], None]) -> "StartupApp": ...
    def on_after_update_fast_callback(self, callback: Callable[[str], None]) -> "StartupApp": ...
    def on_before_update_fast_callback(self, callback: Callable[[str], None]) -> "StartupApp": ...
    def on_before_uninstall_fast_callback(
        self, callback: Callable[[str], None]
    ) -> "StartupApp": ...
    def run(self) -> None: ...


def run_startup_hooks(
    app_factory: Callable[[], StartupApp] | None = None,
    *,
    autostart: Autostart | None = None,
    enabled: Callable[[], bool] = lambda: True,
) -> None:
    """Must run before anything else in the frozen exe: Velopack calls it with install flags.

    Returns normally on an ordinary start; on an install/update/uninstall call Velopack exits the
    process after the matching hook has run.
    """
    if app_factory is None:
        import velopack

        app_factory = velopack.App
    tasks = autostart or Autostart()
    app = app_factory()
    (
        app.on_after_install_fast_callback(lambda _version: after_install(tasks, enabled))
        .on_after_update_fast_callback(lambda _version: after_update(tasks, enabled))
        .on_before_update_fast_callback(lambda _version: before_update())
        .on_before_uninstall_fast_callback(lambda _version: before_uninstall(tasks))
        .run()
    )
