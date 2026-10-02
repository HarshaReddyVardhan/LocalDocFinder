"""Runs ``SetupFlow`` on a worker thread and reports progress to the wizard with signals.

The flow is Qt-free; this is the thin bridge. The one blocking interaction, the "this model is
slow, switch to a smaller one?" question, is posted to the GUI thread and waited on here.
"""

import logging
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from vector_embed.core.providers.base import ProviderError
from vector_embed.core.settings import Settings
from vector_embed.core.setup.flow import (
    SetupError,
    SetupFlow,
    SetupOptions,
    SetupPreview,
    SlowOffer,
)
from vector_embed.core.setup.plan import SetupChoices, SetupPlanError
from vector_embed.core.setup.wiring import FlowBuilder
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

ANSWER_TIMEOUT_SECONDS = 600.0  # an unanswered question must not hang the worker forever
_KNOWN_ERRORS = (SetupError, SetupPlanError, ProviderError, OSError)


class SetupController(QObject):
    progressed = Signal(object)  # SetupEvent
    finished = Signal(object)  # SetupResult
    failed = Signal(str)
    downgrade_offered = Signal(object)  # SlowOffer

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
        self.running = False

    def preview(self, choices: SetupChoices = SetupChoices()) -> SetupPreview:  # noqa: B008
        """What setup would do; a quick, read-only probe (no changes)."""
        return self._flow.preview(SetupOptions(choices))

    def start(self, options: SetupOptions) -> None:
        if self.running:
            return
        self.running = True
        self._pool.start(_FlowJob(self, options))

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
        except _KNOWN_ERRORS as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # worker boundary: surface anything unexpected, never die silently
            logger.exception("setup crashed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)
        finally:
            self.running = False


class _FlowJob(QRunnable):
    def __init__(self, controller: SetupController, options: SetupOptions) -> None:
        super().__init__()
        self._controller = controller
        self._options = options

    def run(self) -> None:
        self._controller.run_flow(self._options)
