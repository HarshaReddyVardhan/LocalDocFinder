r"""Places that are never indexed, whatever the user chooses.

Choosing the whole Windows drive (or ``C:\Windows`` itself) must not put the operating system,
installed programs, other users' profiles or the recycle bin into the index. These rules are
code, not settings: nothing in ``settings.toml`` can switch them off.
"""

import os
from collections.abc import Mapping
from pathlib import Path

# A directory with one of these names is skipped wherever it is, on any drive.
PROTECTED_NAMES = frozenset({"$recycle.bin", "system volume information", "windows.old"})
_ENV_DIRS = ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData")
_PROFILES_DIR = "users"


def _key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


class SystemProtection:
    """Answers "is this path part of the operating system or someone else's profile?"."""

    def __init__(self, environ: Mapping[str, str] | None = None, home: Path | None = None) -> None:
        env = os.environ if environ is None else environ
        self._dirs = tuple(_key(env[name]) for name in _ENV_DIRS if env.get(name))
        self.system_drive = (env.get("SystemDrive") or "C:").rstrip(r"\/").lower()
        self._home = _key(home if home is not None else Path.home())
        self._profiles = _key(self.system_drive + os.sep + _PROFILES_DIR)

    def reason(self, path: str | Path) -> str | None:
        """Why ``path`` is never indexed, or ``None`` if it may be."""
        candidate = _key(path)
        parts = set(Path(candidate).parts)
        if parts & PROTECTED_NAMES:
            return "a recycle bin or system restore folder"
        for directory in self._dirs:
            if candidate == directory or candidate.startswith(directory + os.sep):
                return "a Windows or program folder"
        if candidate.startswith(self._profiles + os.sep):
            if candidate == self._home or candidate.startswith(self._home + os.sep):
                return None
            return "another user's profile"
        return None

    def is_protected(self, path: str | Path) -> bool:
        return self.reason(path) is not None

    def is_system_drive_root(self, path: str | Path) -> bool:
        return _key(path).rstrip(r"\/") == self.system_drive
