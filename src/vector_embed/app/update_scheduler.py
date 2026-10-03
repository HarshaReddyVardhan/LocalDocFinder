"""Background update checks: shortly after start, then whenever a day has passed.

The check downloads the release (only the changed parts), so it runs on a worker thread. The
tray offers "Restart to update" when ``ready`` fires; nothing restarts without the user.
"""

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from vector_embed.core.updates import UpdateKind, UpdateOutcome, Updater

logger = logging.getLogger(__name__)

STARTUP_DELAY_MS = 30_000  # let the app settle before touching the network
POLL_INTERVAL_MS = 60 * 60 * 1000  # "is a day up?" is checked hourly; the check itself is daily


class _Signals(QObject):
    done = Signal(object)  # UpdateOutcome


class _CheckJob(QRunnable):
    def __init__(self, updater: Updater, signals: _Signals) -> None:
        super().__init__()
        self._updater = updater
        self._signals = signals

    def run(self) -> None:
        try:
            outcome = self._updater.check()
        except Exception as exc:  # worker boundary: whatever happens, the scheduler must hear back
            logger.exception("updates: the check crashed")
            outcome = UpdateOutcome(UpdateKind.FAILED, f"Could not check for updates: {exc}")
        self._signals.done.emit(outcome)


class UpdateScheduler(QObject):
    ready = Signal(str)  # version of a downloaded update

    def __init__(
        self,
        updater: Updater,
        enabled: Callable[[], bool],
        pool: QThreadPool | None = None,
        startup_delay_ms: int = STARTUP_DELAY_MS,
        poll_interval_ms: int = POLL_INTERVAL_MS,
    ) -> None:
        super().__init__()
        self._updater = updater
        self._enabled = enabled
        self._pool = pool or QThreadPool.globalInstance()
        self._startup_delay_ms = startup_delay_ms
        self._busy = False
        self._announced: str | None = None
        self._signals = _Signals()
        self._signals.done.connect(self._on_done)
        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self.tick)

    def start(self) -> None:
        QTimer.singleShot(self._startup_delay_ms, self.tick)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def tick(self) -> None:
        """Check now if auto-check is on and the last check is more than a day old."""
        if self._busy or not self._enabled() or not self._updater.due():
            return
        self._busy = True
        self._pool.start(_CheckJob(self._updater, self._signals))

    def _on_done(self, outcome: UpdateOutcome) -> None:
        self._busy = False
        logger.info("updates: %s", outcome.kind.value)
        if outcome.kind is UpdateKind.READY and outcome.version != self._announced:
            self._announced = outcome.version
            self.ready.emit(outcome.version or "")
