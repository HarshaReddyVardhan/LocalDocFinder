r"""Which folders are scanned, from the coverage the user picked.

"Entire PC" means every fixed drive; a chosen drive or folder means just that. The Windows
drive is special: scanning all of ``C:\`` would walk Windows, programs and every profile, so
it is replaced by the user's own folder plus any ordinary folders the user made on it.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path

from vector_embed.core.drives import fixed_drives
from vector_embed.core.protection import SystemProtection
from vector_embed.core.scope import is_reparse_point
from vector_embed.core.settings import ScopeSettings

logger = logging.getLogger(__name__)

_PROFILES_DIR = "users"


def _top_level_dirs(drive_root: str) -> list[str]:
    try:
        return sorted(e.path for e in os.scandir(drive_root) if e.is_dir(follow_symlinks=False))
    except OSError:
        logger.warning("cannot list %s", drive_root, exc_info=True)
        return []


def expand_system_drive(
    drive_root: str,
    scope: ScopeSettings,
    protection: SystemProtection,
    home: Path,
    list_dirs: Callable[[str], list[str]] = _top_level_dirs,
) -> list[str]:
    """The user's folder and the ordinary top-level folders of the Windows drive."""
    roots = [str(home)] if home.is_dir() else []
    for folder in list_dirs(drive_root):
        name = Path(folder).name.lower()
        if (
            name == _PROFILES_DIR
            or name in scope.blocked_dirs
            or name.startswith((".", "$"))
            or protection.is_protected(folder)
            or is_reparse_point(folder)
        ):
            continue
        roots.append(folder)
    return roots


def resolve_roots(
    scope: ScopeSettings,
    *,
    drives: Callable[[], list[str]] = fixed_drives,
    protection: SystemProtection | None = None,
    home: Path | None = None,
    list_dirs: Callable[[str], list[str]] = _top_level_dirs,
) -> tuple[str, ...]:
    """The folders to scan: protected ones removed, the Windows drive made safe, no duplicates."""
    protection = protection or SystemProtection()
    home = home if home is not None else Path.home()
    wanted = list(scope.roots) if scope.coverage == "chosen" else drives()
    resolved: list[str] = []
    for root in wanted:
        if protection.is_system_drive_root(root):
            resolved.extend(expand_system_drive(root, scope, protection, home, list_dirs))
        elif not protection.is_protected(root):
            resolved.append(root)
    return _without_nested(resolved)


def _without_nested(roots: list[str]) -> tuple[str, ...]:
    """Drop duplicates and folders that lie inside another root (they would be scanned twice)."""
    keyed = {os.path.normcase(os.path.normpath(root)): root for root in roots}
    kept = [
        original
        for key, original in keyed.items()
        if not any(key.startswith(other.rstrip(os.sep) + os.sep) for other in keyed if other != key)
    ]
    return tuple(kept)
