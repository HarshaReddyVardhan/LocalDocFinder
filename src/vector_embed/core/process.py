"""Cross-process single-instance lock (Windows ``msvcrt`` file lock) and our own command lines."""

import msvcrt
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path


@contextmanager
def single_instance(name: str, data_dir: Path) -> Iterator[bool]:
    """Yield ``True`` if this process got the named lock, ``False`` if another holds it.

    The lock is released when the context exits or the process dies.
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / f"{name}.lock").open("a+")
    acquired = False
    try:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            with suppress(OSError):
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


ENTRY_POINTS = ("app", "watcher", "worker", "setup")  # what ``vector_embed.__main__`` dispatches


def is_frozen() -> bool:
    """True inside the PyInstaller build."""
    return bool(getattr(sys, "frozen", False))


def self_command(entry: str, *, windowless: bool = False) -> list[str]:
    """Command line that starts one of our entry points, from source or from the frozen build.

    Frozen (PyInstaller): ``VectorEmbed.exe <entry>``. From source: ``python -m vector_embed
    <entry>``, using ``pythonw`` when ``windowless`` so no console flashes up.
    """
    if entry not in ENTRY_POINTS:
        raise ValueError(f"unknown entry point {entry!r}")
    if is_frozen():
        return [sys.executable, entry]
    interpreter = Path(sys.executable)
    if windowless and (quiet := interpreter.with_name("pythonw.exe")).is_file():
        interpreter = quiet
    return [str(interpreter), "-m", "vector_embed", entry]


STOP_REQUEST_FILENAME = "stop.request"
STOP_REQUEST_MAX_AGE_SECONDS = 60.0  # an old flag from a crashed updater must not stop us forever


def request_stop(data_dir: Path) -> None:
    """Ask the worker and watcher to finish what they are doing and exit (and unload the model).

    ``Process.terminate`` on Windows is a hard kill that skips ``finally`` blocks, so a worker
    stopped that way would leave the model on the GPU.
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STOP_REQUEST_FILENAME).write_text("stop", encoding="ascii")


def clear_stop_request(data_dir: Path) -> None:
    (Path(data_dir) / STOP_REQUEST_FILENAME).unlink(missing_ok=True)


def stop_requested(data_dir: Path, now: float | None = None) -> bool:
    try:
        modified = (Path(data_dir) / STOP_REQUEST_FILENAME).stat().st_mtime
    except OSError:
        return False
    return (time.time() if now is None else now) - modified <= STOP_REQUEST_MAX_AGE_SECONDS
