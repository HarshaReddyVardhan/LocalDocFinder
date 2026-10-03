"""The Settings window's tabs. Each is a small widget over ``SettingsController``."""

from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vector_embed.app.models_panel import ModelsPanel
from vector_embed.app.settings_controller import SettingsController, app_version
from vector_embed.core.models.benchmark import BenchKind, Verdict, judge
from vector_embed.core.settings import SettingsError
from vector_embed.core.updates import UpdateKind, UpdateOutcome

ADD_FOLDER_TITLE = "Add a folder to index"
KEY_PROMPT_TITLE = "API key"
DEFAULT_BUDGET_USD = 10.0


def pick_folder() -> str | None:
    path = QFileDialog.getExistingDirectory(None, ADD_FOLDER_TITLE)
    return path or None


def ask_secret(provider: str) -> str | None:
    """Password-style prompt; the key is never shown or logged."""
    text, accepted = QInputDialog.getText(
        None, KEY_PROMPT_TITLE, f"API key for {provider}:", QLineEdit.EchoMode.Password
    )
    return text if accepted and text.strip() else None


class SettingsTab(QWidget):
    """A tab reports problems and confirmations through ``message``."""

    message = Signal(str)

    def __init__(self, controller: SettingsController) -> None:
        super().__init__()
        self._controller = controller

    def refresh(self) -> None:
        """Re-read settings from disk and redraw."""

    def _guard(self, action: Callable[[], object], done: str = "") -> bool:
        try:
            action()
        except (SettingsError, OSError, RuntimeError) as exc:
            self.message.emit(str(exc))
            return False
        if done:
            self.message.emit(done)
        return True


class GeneralTab(SettingsTab):
    hotkey_changed = Signal(str)

    def __init__(
        self, controller: SettingsController, choose_folder: Callable[[], str | None] = pick_folder
    ) -> None:
        super().__init__(controller)
        self._choose_folder = choose_folder
        self.hotkey = QLineEdit()
        self.apply_hotkey = QPushButton("Apply")
        self.start_with_windows = QCheckBox("Start with Windows")
        self.folders = QListWidget()
        self.add_folder = QPushButton("Add folder…")
        self.remove_folder = QPushButton("Remove selected")

        hotkey_row = QHBoxLayout()
        hotkey_row.addWidget(self.hotkey, 1)
        hotkey_row.addWidget(self.apply_hotkey)
        folder_buttons = QHBoxLayout()
        folder_buttons.addWidget(self.add_folder)
        folder_buttons.addWidget(self.remove_folder)
        folder_buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Search hotkey"))
        layout.addLayout(hotkey_row)
        layout.addWidget(self.start_with_windows)
        layout.addWidget(QLabel("Folders to index"))
        layout.addWidget(self.folders, 1)
        layout.addLayout(folder_buttons)

        self.refresh()
        self.apply_hotkey.clicked.connect(self._apply_hotkey)
        self.hotkey.returnPressed.connect(self._apply_hotkey)
        self.start_with_windows.clicked.connect(self._toggle_autostart)
        self.add_folder.clicked.connect(self._add_folder)
        self.remove_folder.clicked.connect(self._remove_folder)

    def refresh(self) -> None:
        settings = self._controller.settings()
        self.hotkey.setText(settings.search.hotkey)
        self.start_with_windows.setChecked(settings.app.start_with_windows)
        self.folders.clear()
        self.folders.addItems(list(settings.scope.roots))

    def _roots(self) -> list[str]:
        return [self.folders.item(i).text() for i in range(self.folders.count())]

    def _apply_hotkey(self) -> None:
        holder: list[str] = []
        if self._guard(lambda: holder.append(self._controller.set_hotkey(self.hotkey.text()))):
            self.hotkey.setText(holder[0])
            self.hotkey_changed.emit(holder[0])
            self.message.emit(f"hotkey set to {holder[0]}")

    def _toggle_autostart(self, checked: bool) -> None:
        if not self._guard(lambda: self._controller.set_start_with_windows(checked)):
            self.refresh()

    def _add_folder(self) -> None:
        folder = self._choose_folder()
        if not folder or folder in self._roots():
            return
        added = self._guard(
            lambda: self._controller.set_roots([*self._roots(), folder]),
            "folder added; the watcher starts on it within a minute and scans it when idle",
        )
        if added:
            self.refresh()

    def _remove_folder(self) -> None:
        row = self.folders.currentRow()
        if row < 0:
            return
        remaining = [root for i, root in enumerate(self._roots()) if i != row]
        if self._guard(lambda: self._controller.set_roots(remaining), "folder removed"):
            self.refresh()


class ModelsTab(SettingsTab):
    """The existing Models/Health panel, plus the speed-test results from setup."""

    def __init__(self, controller: SettingsController, panel: ModelsPanel) -> None:
        super().__init__(controller)
        self.panel = panel
        self.speed = QLabel("")
        self.speed.setWordWrap(True)
        panel.status_changed.connect(self.message)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(panel, 1)
        layout.addWidget(QLabel("Speed tests (from setup)"))
        layout.addWidget(self.speed)

    def refresh(self) -> None:
        self.panel.activate()  # loads data and starts the health dashboard refresh
        self.speed.setText(self._speed_text())

    def _speed_text(self) -> str:
        results = self._controller.speed_tests()
        if not results:
            return "Not measured yet. Run setup again to measure your models."
        lines = []
        for result in sorted(results, key=lambda r: (r.kind is BenchKind.CHAT, r.model)):
            note = "  (slow on this machine)" if judge(result) is Verdict.SLOW else ""
            lines.append(f"{result.model}: {result.rate:.1f} {result.unit}{note}")
        return "\n".join(lines)


class CloudTab(SettingsTab):
    def __init__(
        self,
        controller: SettingsController,
        ask_key: Callable[[str], str | None] = ask_secret,
    ) -> None:
        super().__init__(controller)
        self._ask_key = ask_key
        self.redact = QCheckBox("Hide my name, email and phone from cloud models")
        self.mask_local = QCheckBox("Also mask ID numbers for local models")
        self.limit = QCheckBox("Limit monthly cloud spend")
        self.budget = QDoubleSpinBox()
        self.budget.setPrefix("$ ")
        self.budget.setRange(1.0, 100000.0)
        self.keys = QVBoxLayout()
        self.key_labels: list[QLabel] = []
        self.key_buttons: list[QPushButton] = []
        budget_row = QHBoxLayout()
        budget_row.addWidget(self.limit)
        budget_row.addWidget(self.budget)
        budget_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Cloud providers (add them with `ve cloud add`)"))
        layout.addLayout(self.keys)
        layout.addWidget(QLabel("Privacy"))
        layout.addWidget(self.redact)
        layout.addWidget(self.mask_local)
        layout.addLayout(budget_row)
        layout.addStretch(1)

        self.refresh()
        self.redact.clicked.connect(lambda on: self._save(self._controller.set_redact_personal, on))
        self.mask_local.clicked.connect(
            lambda on: self._save(self._controller.set_mask_ids_locally, on)
        )
        self.limit.clicked.connect(lambda _on: self._save_budget())
        self.budget.editingFinished.connect(self._save_budget)

    def refresh(self) -> None:
        settings = self._controller.settings()
        self.redact.setChecked(settings.privacy.redact_personal)
        self.mask_local.setChecked(settings.privacy.mask_ids_locally)
        limit = settings.cloud.monthly_budget_usd
        self.limit.setChecked(limit is not None)
        self.budget.setValue(limit if limit is not None else DEFAULT_BUDGET_USD)
        self.budget.setEnabled(limit is not None)
        self._fill_keys()

    def _fill_keys(self) -> None:
        while self.keys.count():
            item = self.keys.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self.key_labels, self.key_buttons = [], []
        statuses = self._controller.key_statuses()
        if not statuses:
            self.keys.addWidget(QLabel("No cloud provider is configured; everything stays local."))
        for status in statuses:
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            label = QLabel(f"{status.label}: {'key stored' if status.has_key else 'no key'}")
            line.addWidget(label, 1)
            button = QPushButton("Set key…")
            self.key_labels.append(label)
            self.key_buttons.append(button)
            button.clicked.connect(lambda _c=False, name=status.provider: self._set_key(name))
            line.addWidget(button)
            self.keys.addWidget(row)

    def _set_key(self, provider: str) -> None:
        key = self._ask_key(provider)
        if key and self._guard(lambda: self._controller.set_key(provider, key), "key saved"):
            self._fill_keys()

    def _save(self, setter: Callable[[bool], None], value: bool) -> None:
        self._guard(lambda: setter(value), "saved")

    def _save_budget(self) -> None:
        self.budget.setEnabled(self.limit.isChecked())
        value = self.budget.value() if self.limit.isChecked() else None
        self._guard(lambda: self._controller.set_monthly_budget(value), "budget saved")


class _CheckSignals(QObject):
    done = Signal(object)  # UpdateOutcome


class _CheckJob(QRunnable):
    def __init__(self, controller: SettingsController, signals: _CheckSignals) -> None:
        super().__init__()
        self._controller = controller
        self._signals = signals

    def run(self) -> None:
        self._signals.done.emit(self._controller.check_now())


class UpdatesTab(SettingsTab):
    def __init__(self, controller: SettingsController, pool: QThreadPool | None = None) -> None:
        super().__init__(controller)
        self._pool = pool or QThreadPool.globalInstance()
        self._signals = _CheckSignals()
        self._signals.done.connect(self._show_outcome)
        self.auto_check = QCheckBox("Check for updates automatically")
        self.check_now = QPushButton("Check now")
        self.restart = QPushButton("Restart to update")
        self.restart.setVisible(False)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.auto_check)
        layout.addWidget(self.check_now)
        layout.addWidget(self.result)
        layout.addWidget(self.restart)
        layout.addStretch(1)
        self.refresh()
        self.auto_check.clicked.connect(
            lambda on: self._guard(lambda: self._controller.set_auto_check(on), "saved")
        )
        self.check_now.clicked.connect(self._check)
        self.restart.clicked.connect(lambda: self._guard(self._controller.restart_to_update))

    def refresh(self) -> None:
        self.auto_check.setChecked(self._controller.settings().updates.auto_check)

    def _check(self) -> None:
        """The check downloads a release, so it runs in the background."""
        self.check_now.setEnabled(False)
        self.result.setText("Checking…")
        self._pool.start(_CheckJob(self._controller, self._signals))

    def _show_outcome(self, outcome: UpdateOutcome) -> None:
        self.check_now.setEnabled(True)
        self.result.setText(outcome.message)
        self.restart.setVisible(outcome.kind is UpdateKind.READY)


def confirm_delete_dialog() -> bool:
    answer = QMessageBox.warning(
        None,
        "Delete my data",
        "This removes the search index, settings and logs, and closes Vector Embed.\n"
        "Your files and Ollama models are not touched. Continue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return answer == QMessageBox.StandardButton.Yes


class AboutTab(SettingsTab):
    def __init__(
        self,
        controller: SettingsController,
        confirm_delete: Callable[[], bool] = confirm_delete_dialog,
        quit_app: Callable[[], None] = lambda: QApplication.quit(),  # noqa: PLW0108
    ) -> None:
        super().__init__(controller)
        self._confirm_delete = confirm_delete
        self._quit_app = quit_app
        self.version = QLabel(f"Vector Embed {app_version()}")
        self.data_folder = QLabel("")
        self.data_folder.setWordWrap(True)
        self.open_folder = QPushButton("Open data folder")
        self.delete_data = QPushButton("Delete my data…")
        layout = QFormLayout(self)
        layout.addRow(self.version)
        layout.addRow("Data folder", self.data_folder)
        layout.addRow(self.open_folder)
        layout.addRow(self.delete_data)
        self.refresh()
        self.open_folder.clicked.connect(self._open)
        self.delete_data.clicked.connect(self._delete)

    def refresh(self) -> None:
        self.data_folder.setText(str(self._controller.settings_path.parent))

    def _delete(self) -> None:
        if self._confirm_delete() and self._guard(self._controller.delete_my_data):
            self._quit_app()

    def _open(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._controller.settings_path.parent)))
