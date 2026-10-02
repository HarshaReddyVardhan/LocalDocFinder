"""Cross-process single-instance lock (Windows ``msvcrt`` file lock)."""

import msvcrt
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
