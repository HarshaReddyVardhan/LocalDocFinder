r"""One-time moves of user data to where this release keeps it.

Releases up to 0.1 kept the index in ``%LOCALAPPDATA%\VectorEmbed``, which is also where Velopack
installs the app, so uninstalling deleted the index. Releases named Vector Embed then used
``VectorEmbedData``. Data now lives in ``LocalDocFinderData``.
"""

import logging
import shutil
from pathlib import Path

from localdoc_finder.core.store.sqlite import STATE_FILENAME

logger = logging.getLogger(__name__)

# Everything Velopack puts in the install folder; never data, never moved.
_INSTALLER_ENTRIES = frozenset({"current", "packages", "update.exe", "sq.version", ".betaid"})
_DATA_MARKERS = (STATE_FILENAME, "settings.toml")


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


def rename_data_dir(old_dir: Path, new_dir: Path) -> bool:
    """Rename the whole data folder in one step; True when it moved.

    A rename either happens completely or not at all, so an old copy of the app holding a file
    open can never leave the index split across two folders: the rename is simply tried again
    on the next start.
    """
    if not old_dir.is_dir() or new_dir.exists():
        return False
    try:
        old_dir.rename(new_dir)
    except OSError:
        logger.warning("could not rename the old data folder; it is in use", exc_info=True)
        return False
    logger.info("renamed the data folder to %s", new_dir)
    return True


def _has_content(directory: Path) -> bool:
    return directory.is_dir() and any(directory.iterdir())
