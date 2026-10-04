"""The Features tab: switch Ask, Chat and Match on or off after setup.

Search is always on. The others answer with a chat model, which setup only downloads when one of
them was picked, so switching one on here offers that download if no chat model is installed.
"""

import logging
from collections.abc import Callable
from typing import Protocol

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from localdoc_finder.app.settings_controller import SettingsController
from localdoc_finder.app.settings_tabs import SettingsTab
from localdoc_finder.core.features import FEATURE_TITLES, enabled_features
from localdoc_finder.core.models.manager import ProgressCallback
from localdoc_finder.core.models.starter import StarterPick
from localdoc_finder.core.providers.base import ProviderError

logger = logging.getLogger(__name__)

FEATURES_HINT = (
    "Search is always on. Ask, Chat and Match answer with a local chat model, a larger download "
    "that is only needed once you switch one of them on."
)
_KNOWN_ERRORS = (ProviderError, RuntimeError, OSError, ValueError)
_MB_PER_GB = 1024


class ChatModels(Protocol):
    """What the tab needs from the Models controller."""

    def missing_chat_model(self) -> StarterPick | None: ...

    def pull(self, name: str, progress: ProgressCallback | None = None) -> None: ...


class _Signals(QObject):
    missing = Signal(object)  # StarterPick | None
    progress = Signal(str, float)
    pulled = Signal(str)
    failed = Signal(str)


class _Job(QRunnable):
    def __init__(self, fn: Callable[[], None], failed: Callable[[str], None]) -> None:
        super().__init__()
        self._fn, self._failed = fn, failed

    def run(self) -> None:
        try:
            self._fn()
        except _KNOWN_ERRORS as exc:
            self._failed(str(exc))
        except Exception as exc:  # unexpected: surface it rather than die in the pool
            logger.exception("features job crashed")
            self._failed(f"{type(exc).__name__}: {exc}")


class FeaturesTab(SettingsTab):
    def __init__(
        self,
        controller: SettingsController,
        models: ChatModels,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(controller)
        self._models = models
        self._pool = pool or QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.missing.connect(self._show_missing)
        self._signals.progress.connect(self._on_progress)
        self._signals.pulled.connect(self._on_pulled)
        self._signals.failed.connect(self._on_failed)
        self._pick: StarterPick | None = None
        self._pulling = False

        hint = QLabel(FEATURES_HINT)
        hint.setWordWrap(True)
        self.boxes = {name: QCheckBox(title) for name, title in FEATURE_TITLES.items()}
        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.download = QPushButton("")
        self.progress = QProgressBar()
        download_row = QWidget()
        row = QHBoxLayout(download_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.model_note, 1)
        row.addWidget(self.download)
        self.download_row = download_row

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        for box in self.boxes.values():
            layout.addWidget(box)
        layout.addWidget(download_row)
        layout.addWidget(self.progress)
        layout.addStretch(1)
        self._hide_download()

        self.refresh()
        for name, box in self.boxes.items():
            box.clicked.connect(lambda on, n=name: self._toggle(n, on))
        self.download.clicked.connect(self._pull)

    def refresh(self) -> None:
        enabled = enabled_features(self._controller.settings())
        for name, box in self.boxes.items():
            box.setChecked(name in enabled)
        self._check_model(bool(enabled))

    def _toggle(self, name: str, on: bool) -> None:
        state = "on" if on else "off"
        if not self._guard(lambda: self._controller.set_feature(name, on), f"{name} is {state}"):
            self.boxes[name].setChecked(not on)
            return
        self._check_model(any(box.isChecked() for box in self.boxes.values()))

    # ------------------------------------------------------------------ chat model
    def _check_model(self, wanted: bool) -> None:
        """Asks Ollama in the background whether a chat model still has to be downloaded."""
        if not wanted:
            if not self._pulling:
                self._hide_download()
            return
        signals = self._signals
        self._pool.start(
            _Job(lambda: signals.missing.emit(self._models.missing_chat_model()), self._fail)
        )

    def _fail(self, message: str) -> None:
        self._signals.failed.emit(message)

    def _show_missing(self, pick: StarterPick | None) -> None:
        self._pick = pick
        if pick is None or self._pulling:
            if not self._pulling:
                self._hide_download()
            return
        size = f"{pick.download_mb / _MB_PER_GB:.1f} GB"
        self.model_note.setText(
            f"No chat model is installed yet, so these features cannot answer. "
            f"Recommended for this PC: {pick.model} ({size}; {pick.reason})."
        )
        self.download.setText(f"Download {pick.model}")
        self.download.setEnabled(True)
        self.download_row.setVisible(True)

    def _hide_download(self) -> None:
        self.download_row.setVisible(False)
        self.progress.setVisible(False)

    def _pull(self) -> None:
        pick = self._pick
        if pick is None or self._pulling:
            return
        self._pulling = True
        self.download.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        signals = self._signals

        def work() -> None:
            self._models.pull(pick.model, lambda p: signals.progress.emit(p.status, p.fraction))
            signals.pulled.emit(pick.model)

        self._pool.start(_Job(work, self._fail))

    def _on_progress(self, status: str, fraction: float) -> None:
        self.progress.setValue(int(fraction * 100))
        self.message.emit(status)

    def _on_pulled(self, name: str) -> None:
        self._pulling = False
        self._hide_download()
        self.message.emit(f"downloaded {name}: Ask, Chat and Match are ready")

    def _on_failed(self, message: str) -> None:
        self._pulling = False
        self.progress.setVisible(False)
        self.download.setEnabled(True)
        self.message.emit(message)
