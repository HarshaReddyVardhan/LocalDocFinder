"""Writing user settings back to ``settings.toml`` (model overrides, embedder changes).

Only the keys the user changed are stored, so new defaults keep flowing in. A change is
validated against the full schema before the file is replaced, and the replace is atomic.
"""

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tomli_w

from vector_embed.core.settings import SCHEMA_VERSION, Settings, SettingsError, migrate


def set_setting(path: Path, keys: Sequence[str], value: object) -> None:
    """Set ``keys`` (a path such as ``["models", "overrides", "chat"]``) to ``value``.

    ``value=None`` removes the key. Raises ``SettingsError`` and leaves the file untouched if
    the result would not validate.
    """
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
    data.setdefault("schema_version", SCHEMA_VERSION)
    _validate(data, path)
    _write_atomic(path, data)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"cannot read {path}: {exc}") from exc


def _validate(data: dict[str, Any], path: Path) -> None:
    try:
        Settings(**migrate(data))
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise SettingsError(f"refusing to write invalid settings to {path}: {exc}") from exc


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        tomli_w.dump(data, handle)
    temp.replace(path)
