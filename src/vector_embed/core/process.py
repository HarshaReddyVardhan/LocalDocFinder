"""Cross-process single-instance lock (Windows ``msvcrt`` file lock) and our own command lines."""

import msvcrt
import sys
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


def self_command(entry: str, *, windowless: bool = False) -> list[str]:
    """Command line that starts one of our entry points, from source or from the frozen build.

    Frozen (PyInstaller): ``VectorEmbed.exe <entry>``. From source: ``python -m vector_embed
    <entry>``, using ``pythonw`` when ``windowless`` so no console flashes up.
    """
    if entry not in ENTRY_POINTS:
        raise ValueError(f"unknown entry point {entry!r}")
    if getattr(sys, "frozen", False):
        return [sys.executable, entry]
    interpreter = Path(sys.executable)
    if windowless and (quiet := interpreter.with_name("pythonw.exe")).is_file():
        interpreter = quiet
    return [str(interpreter), "-m", "vector_embed", entry]
