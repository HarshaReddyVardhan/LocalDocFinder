"""Self-update through Velopack and GitHub Releases.

``Updater`` checks for a newer release, downloads only what changed (delta packages) and then
waits for the user to restart. The Velopack ``UpdateManager`` is injected, so tests never touch the
network and a dev checkout (not installed by Velopack) reports "not installed" instead of failing.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from typing import Protocol, cast

from localdoc_finder.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

SOURCE_FILENAME = "update_source.txt"  # written by scripts/build.ps1 for release builds
LAST_CHECK_KEY = "last_update_check"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60


class UpdateKind(StrEnum):
    UP_TO_DATE = "up_to_date"
    READY = "ready"  # downloaded; applying needs a restart
    NOT_INSTALLED = "not_installed"  # running from source or an unpacked build
    NOT_CONFIGURED = "not_configured"  # no release source known
    FAILED = "failed"


@dataclass(frozen=True)
class UpdateOutcome:
    kind: UpdateKind
    message: str
    version: str | None = None


class ReleaseLike(Protocol):
    @property
    def Version(self) -> str: ...  # noqa: N802  # mirrors velopack's attribute names


class UpdateInfoLike(Protocol):
    @property
    def TargetFullRelease(self) -> ReleaseLike: ...  # noqa: N802


class UpdateManagerLike(Protocol):
    """The slice of ``velopack.UpdateManager`` that we use."""

    def get_current_version(self) -> str: ...
    def check_for_updates(self) -> UpdateInfoLike | None: ...
    def download_updates(
        self, update_info: UpdateInfoLike, progress_callback: Callable[[int], None] | None = None
    ) -> None: ...
    def apply_updates_and_restart(self, update: UpdateInfoLike) -> None: ...


ManagerFactory = Callable[[str], UpdateManagerLike]


def velopack_manager(repo_url: str) -> UpdateManagerLike:
    import velopack  # imported late: it loads a native module the watcher never needs

    # velopack's classes are nominal; they have exactly the methods of UpdateManagerLike
    return cast(UpdateManagerLike, velopack.UpdateManager(velopack.GithubSource(repo_url)))


def bundled_source() -> str:
    """The release source baked into this build ("" for a dev checkout)."""
    try:
        return (
            resources.files("localdoc_finder").joinpath(SOURCE_FILENAME).read_text("utf-8").strip()
        )
    except (OSError, ModuleNotFoundError):
        return ""


def resolve_source(configured: str) -> str:
    """The user's ``updates.repo_url`` wins over the one built into the release."""
    return configured.strip() or bundled_source()


class Updater:
    def __init__(
        self,
        repo_url: str,
        *,
        state: StateDb | None = None,
        factory: ManagerFactory = velopack_manager,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._repo_url = repo_url
        self._state = state
        self._factory = factory
        self._clock = clock
        self._manager: UpdateManagerLike | None = None
        self._pending: UpdateInfoLike | None = None
        self._lock = threading.Lock()
        self.outcome: UpdateOutcome | None = None

    def due(self) -> bool:
        """True when the last check is more than a day old (or there has never been one)."""
        last = self._state.get_meta(LAST_CHECK_KEY) if self._state else None
        return last is None or self._clock() - float(last) >= CHECK_INTERVAL_SECONDS

    def check(self, progress: Callable[[int], None] | None = None) -> UpdateOutcome:
        """Look for a newer release and download it. Never raises: the result says what happened.

        One check at a time: the daily scheduler and the "Check now" button may overlap, and two
        downloads of the same release would only fight over the same files.
        """
        with self._lock:
            try:
                return self._check(progress)
            except Exception as exc:  # the updater's boundary: the native layer raises anything
                logger.exception("updates: unexpected failure")
                return self._finish(
                    UpdateOutcome(UpdateKind.FAILED, f"Could not check for updates: {exc}")
                )

    def _check(self, progress: Callable[[int], None] | None) -> UpdateOutcome:
        if not self._repo_url:
            return self._finish(
                UpdateOutcome(UpdateKind.NOT_CONFIGURED, "No update source is set.")
            )
        if self._pending is not None:
            return self._finish(self._ready(self._pending))
        try:
            manager = self._manager or self._factory(self._repo_url)
        except RuntimeError as exc:  # velopack: "not properly installed"
            logger.info("updates: not an installed build (%s)", exc)
            return self._finish(
                UpdateOutcome(UpdateKind.NOT_INSTALLED, "Updates only work in the installed app.")
            )
        self._manager = manager
        return self._finish(self._look(manager, progress))

    def _look(
        self, manager: UpdateManagerLike, progress: Callable[[int], None] | None
    ) -> UpdateOutcome:
        try:
            info = manager.check_for_updates()
            if info is None:
                version = manager.get_current_version()
                return UpdateOutcome(
                    UpdateKind.UP_TO_DATE, f"You are up to date ({version}).", version
                )
            manager.download_updates(info, progress)
        except (RuntimeError, OSError, ValueError) as exc:  # network down, bad package, ...
            logger.warning("updates: check failed: %s", exc)
            return UpdateOutcome(UpdateKind.FAILED, f"Could not check for updates: {exc}")
        self._pending = info
        return self._ready(info)

    @staticmethod
    def _ready(info: UpdateInfoLike) -> UpdateOutcome:
        version = str(info.TargetFullRelease.Version)
        return UpdateOutcome(
            UpdateKind.READY, f"Version {version} is ready. Restart to update.", version
        )

    def _finish(self, outcome: UpdateOutcome) -> UpdateOutcome:
        if self._state is not None and outcome.kind is not UpdateKind.FAILED:
            self._state.set_meta(LAST_CHECK_KEY, str(self._clock()))
        self.outcome = outcome
        return outcome

    def restart_to_update(self) -> None:
        """Apply the downloaded update and relaunch (the process exits)."""
        if self._manager is None or self._pending is None:
            raise RuntimeError("no update has been downloaded")
        self._manager.apply_updates_and_restart(self._pending)
