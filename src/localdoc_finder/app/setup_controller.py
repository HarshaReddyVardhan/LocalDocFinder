"""Runs ``SetupFlow`` on a worker thread and reports progress to the wizard with signals.

The flow is Qt-free; this is the thin bridge. The one blocking interaction, the "this model is
slow, switch to a smaller one?" question, is posted to the GUI thread and waited on here.
"""

import logging
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.settings import Settings
from localdoc_finder.core.setup.flow import (
    EnvironmentProbe,
    SetupCancelled,
    SetupError,
    SetupFlow,
    SetupOptions,
    SetupPreview,
    SlowOffer,
)
from localdoc_finder.core.setup.ollama_install import OllamaState
from localdoc_finder.core.setup.plan import SetupChoices, SetupPlanError
from localdoc_finder.core.setup.wiring import FlowBuilder
from localdoc_finder.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

ANSWER_TIMEOUT_SECONDS = 600.0  # an unanswered question must not hang the worker forever
_KNOWN_ERRORS = (SetupError, SetupPlanError, ProviderError, OSError)


class SetupController(QObject):
    progressed = Signal(object)  # SetupEvent
    finished = Signal(object)  # SetupResult
    failed = Signal(str)
    downgrade_offered = Signal(object)  # SlowOffer
    probed = Signal(object)  # EnvironmentProbe
    cancelled = Signal()  # the user closed the wizard and the flow stopped

    def __init__(
        self,
        build: FlowBuilder,
        settings: Settings,
        state: StateDb,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__()
        self._pool = pool or QThreadPool.globalInstance()
        self._answered = threading.Event()
        self._accepted = False
        self._flow: SetupFlow = build(settings, state, self.progressed.emit, self._ask_downgrade)
        self._cancel = threading.Event()
        self._flow.cancelled = self._cancel.is_set
        self._lock = threading.Lock()  # start() may be called from two places: one run only
        self.running = False
        self._probe: EnvironmentProbe | None = None

    def probe_async(self) -> None:
        """Look at Ollama and the disk in the background; ``probed`` fires when it is known."""
        self._pool.start(_ProbeJob(self))

    def run_probe(self) -> None:
        try:
            probe = self._flow.probe()
        except _KNOWN_ERRORS as exc:
            logger.warning("setup probe failed: %s", exc)
            probe = EnvironmentProbe(OllamaState.MISSING, (), 0)
        self._probe = probe
        self.probed.emit(probe)

    @property
    def environment(self) -> EnvironmentProbe | None:
        """The last probe, or ``None`` before the first one has finished."""
        return self._probe

    def preview(self, choices: SetupChoices = SetupChoices()) -> SetupPreview | None:  # noqa: B008
        """What setup would do for ``choices``; instant, from the last probe (no changes).

        ``None`` until a probe has finished: the caller shows "checking..." and waits for
        ``probed``. It never touches the network or the disk itself.
        """
        if self._probe is None:
            return None
        return self._flow.preview_from(self._probe, SetupOptions(choices))

    def start(self, options: SetupOptions) -> None:
        with self._lock:
            if self.running:
                return
            self.running = True
            self._cancel.clear()
        self._pool.start(_FlowJob(self, options))

    def cancel(self) -> None:
        """Stop a running setup (between steps and mid-download). Safe to call when idle."""
        self._cancel.set()
        self._accepted = False
        self._answered.set()  # a pending "switch to a smaller model?" question gets a "no"

    def answer_downgrade(self, accepted: bool) -> None:
        self._accepted = accepted
        self._answered.set()

    def _ask_downgrade(self, offer: SlowOffer) -> bool:
        self._answered.clear()
        self.downgrade_offered.emit(offer)
        if not self._answered.wait(ANSWER_TIMEOUT_SECONDS):
            return False
        return self._accepted

    def run_flow(self, options: SetupOptions) -> None:
        try:
            result = self._flow.run(options)
        except SetupCancelled:
            logger.info("setup cancelled by the user")
            self.running = False
            self.cancelled.emit()
            return
        except _KNOWN_ERRORS as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # worker boundary: surface anything unexpected, never die silently
            logger.exception("setup crashed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)
        finally:
            self.running = False


class _ProbeJob(QRunnable):
    def __init__(self, controller: SetupController) -> None:
        super().__init__()
        self._controller = controller

    def run(self) -> None:
        self._controller.run_probe()


class _FlowJob(QRunnable):
    def __init__(self, controller: SetupController, options: SetupOptions) -> None:
        super().__init__()
        self._controller = controller
        self._options = options

    def run(self) -> None:
        self._controller.run_flow(self._options)
