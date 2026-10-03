"""Writing user settings back to ``settings.toml`` (model overrides, embedder changes).

Only the keys the user changed are stored, so new defaults keep flowing in. A change is
validated against the full schema before the file is replaced, and the replace is atomic.

Several processes write this file (the tray app, the command line, the setup wizard), so a
change is a locked read-modify-write: two of them cannot overwrite each other's edit.
"""

import msvcrt
import os
import time
import tomllib
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import tomli_w

from vector_embed.core.settings import SCHEMA_VERSION, Settings, SettingsError, migrate

LOCK_TIMEOUT_SECONDS = 10.0
LOCK_POLL_SECONDS = 0.05
REPLACE_ATTEMPTS = 6  # Windows refuses to replace a file another program has open
REPLACE_RETRY_SECONDS = 0.05


def set_setting(path: Path, keys: Sequence[str], value: object) -> None:
    """Set ``keys`` (a path such as ``["models", "overrides", "chat"]``) to ``value``.

    ``value=None`` removes the key. Raises ``SettingsError`` and leaves the file untouched if
    the result would not validate.
    """
    with _locked(path):
        data = _read(path)
        node = data
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        if value is None:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = value
        # What is stored is the upgraded layout: writing the old one back under a new version
        # number would leave a file that claims to be current and is not.
        _write_atomic(path, _validated(data, path))


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock on ``<path>.lock`` for the whole change."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.with_name(path.name + ".lock").open("a+")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise SettingsError(
                        f"another program is saving {path.name}; try again"
                    ) from None
                time.sleep(LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            with suppress(OSError):
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"cannot read {path}: {exc}") from exc


def _validated(data: dict[str, Any], path: Path) -> dict[str, Any]:
    """The migrated settings, checked against the full schema."""
    try:
        migrated = migrate(data)
        migrated.setdefault("schema_version", SCHEMA_VERSION)
        Settings(**migrated)
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise SettingsError(f"refusing to write invalid settings to {path}: {exc}") from exc
    return migrated


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write beside the target under a name nobody else uses, flush to disk, then swap it in."""
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as handle:
            tomli_w.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())  # a crash after the swap must not leave an empty file
        _replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _replace_with_retry(temp: Path, path: Path) -> None:
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            temp.replace(path)
            return
        except PermissionError:  # an antivirus scan or a reader has the target open for a moment
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_RETRY_SECONDS * (attempt + 1))
