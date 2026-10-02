r"""One-time move of user data out of the Velopack install folder.

Releases up to 0.1 kept the index in ``%LOCALAPPDATA%\VectorEmbed``, which is also where Velopack
installs the app, so uninstalling deleted the index. Data now lives in ``VectorEmbedData``.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Everything Velopack puts in the install folder; never data, never moved.
_INSTALLER_ENTRIES = frozenset({"current", "packages", "update.exe", "sq.version", ".betaid"})
_DATA_MARKERS = ("state.db", "settings.toml")


def migrate_legacy_data(legacy_dir: Path, data_dir: Path) -> bool:
    """Move data from ``legacy_dir`` into an empty ``data_dir``; True when something moved.

    Skips when the new folder already has content or the old one holds no recognisable data, so
    it is safe to call on every start. A failure leaves the old files in place and is logged.
    """
    if not legacy_dir.is_dir() or _has_content(data_dir):
        return False
    if not any((legacy_dir / marker).exists() for marker in _DATA_MARKERS):
        return False
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        for entry in legacy_dir.iterdir():
            if entry.name.lower() in _INSTALLER_ENTRIES or entry.name.lower().startswith("app-"):
                continue
            shutil.move(str(entry), str(data_dir / entry.name))
    except OSError:
        logger.warning("could not move the old data folder", exc_info=True)
        return False
    logger.info("moved user data to %s", data_dir)
    return True


def _has_content(directory: Path) -> bool:
    return directory.is_dir() and any(directory.iterdir())
