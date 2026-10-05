"""Add a cloud provider: pick a service, paste the key, test it, save.

The key lives only in the password field and, once the test passes, in Credential Manager. It is
cleared from the field as soon as the dialog is done, and error messages reach this window already
scrubbed of it (the provider does that).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from localdoc_finder.app.settings_controller import CloudModel, SettingsController
from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.providers.presets import CUSTOM, PRESETS
from localdoc_finder.core.secrets import KeyStoreError
from localdoc_finder.core.settings import SettingsError

logger = logging.getLogger(__name__)

_KNOWN_ERRORS = (ProviderError, SettingsError, KeyStoreError, OSError)


@dataclass(frozen=True)
class ModelsResult:
    """What a background model listing came back with; ``owner`` says which request it answers."""

    owner: str
    models: list[CloudModel]
    error: str = ""


class ModelsSignals(QObject):
    done = Signal(object)  # ModelsResult


class ModelsJob(QRunnable):
    """Run one blocking model listing off the UI thread and report it through ``signals``."""

    def __init__(
        self, owner: str, fetch: Callable[[], list[CloudModel]], signals: ModelsSignals
    ) -> None:
        super().__init__()
        self._owner, self._fetch, self._signals = owner, fetch, signals

    def run(self) -> None:
        try:
            result = ModelsResult(self._owner, self._fetch())
        except _KNOWN_ERRORS as exc:
            result = ModelsResult(self._owner, [], str(exc))
        except Exception as exc:  # unexpected: say so rather than leave the UI waiting
            # Only the type is logged and shown: the text of a stray exception may hold the key.
            logger.warning("cloud model listing crashed: %s", type(exc).__name__)
            result = ModelsResult(self._owner, [], f"unexpected error ({type(exc).__name__})")
        self._signals.done.emit(result)


class CloudProviderDialog(QDialog):
    """Preset, key and (for Custom) address; "Test & save" lists the models, then saves."""

    def __init__(
        self,
        controller: SettingsController,
        parent: QWidget | None = None,
        pool: QThreadPool | None = None,
        open_url: Callable[[str], object] = lambda url: QDesktopServices.openUrl(QUrl(url)),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add a cloud provider")
        self._controller = controller
        self._pool = pool or QThreadPool.globalInstance()
        self._open_url = open_url
        self._signals = ModelsSignals()
        self._signals.done.connect(self._tested)
        self.added: str | None = None  # the saved provider's name once the dialog was accepted
        self._closed = False

        self.preset = QComboBox()
        for key, preset in PRESETS.items():
            self.preset.addItem(preset.label, key)
        self.key_link = QPushButton("Get a key ↗")
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("API key")
        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("https://host/v1")
        self.base_url_label = QLabel("Base URL")
        self.test = QPushButton("Test && save")
        self.cancel = QPushButton("Cancel")
        self.status = QLabel("")
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Service", self.preset)
        form.addRow("", self.key_link)
        form.addRow("API key", self.key)
        form.addRow(self.base_url_label, self.base_url)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(self.test)
        layout.addWidget(self.cancel)

        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.key_link.clicked.connect(self._open_key_page)
        self.test.clicked.connect(self._start_test)
        self.key.returnPressed.connect(self._start_test)
        self.cancel.clicked.connect(self.reject)
        self._preset_changed()

    @property
    def preset_key(self) -> str:
        return str(self.preset.currentData())

    def _preset_changed(self) -> None:
        chosen = PRESETS[self.preset_key]
        self.key_link.setVisible(bool(chosen.key_url))
        custom = self.preset_key == CUSTOM
        self.base_url.setVisible(custom)
        self.base_url_label.setVisible(custom)
        self.status.setText("")

    def _open_key_page(self) -> None:
        self._open_url(PRESETS[self.preset_key].key_url)

    def _start_test(self) -> None:
        if not self.test.isEnabled():
            return
        self._busy(True, "Testing the key and listing models…")
        preset, key, base_url = self.preset_key, self.key.text(), self.base_url.text()
        job = ModelsJob(
            preset, lambda: self._controller.probe_provider(preset, key, base_url), self._signals
        )
        self._pool.start(job)

    def _busy(self, busy: bool, text: str = "") -> None:
        self.test.setEnabled(not busy)
        self.preset.setEnabled(not busy)
        self.status.setText(text)

    def done(self, result: int) -> None:
        self._closed = True  # a test still running must not save anything afterwards
        super().done(result)

    def _tested(self, result: ModelsResult) -> None:
        if self._closed:
            return
        self._busy(False)
        if result.error:
            self.status.setText(result.error)  # nothing was saved
            return
        try:
            self.added = self._controller.add_provider(
                self.preset_key, self.key.text(), self.base_url.text()
            )
        except (SettingsError, KeyStoreError, OSError) as exc:
            self.status.setText(str(exc))
            return
        self.key.clear()
        self.accept()


def run_add_provider_dialog(controller: SettingsController, parent: QWidget | None) -> str | None:
    """Show the dialog; returns the new provider's name, or ``None`` when cancelled."""
    dialog = CloudProviderDialog(controller, parent)
    return dialog.added if dialog.exec() == QDialog.DialogCode.Accepted else None
