"""First-run setup wizard: hardware, Ollama, models, download, speed test, settings, done.

The wizard only collects choices and shows progress; ``SetupFlow`` (via ``SetupController``)
does the work, so ``ldf setup`` and the wizard behave the same.
"""

import ctypes
import sys
from collections.abc import Callable, Collection

from PySide6.QtCore import QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from localdoc_finder.app.cloud_tab import CloudTab
from localdoc_finder.app.scope_editor import ScopeEditor
from localdoc_finder.app.settings_controller import SettingsController
from localdoc_finder.app.settings_tabs import GeneralTab, UpdatesTab
from localdoc_finder.app.setup_controller import SetupController
from localdoc_finder.app.theme import scheme_in_use, style_check_boxes
from localdoc_finder.core.features import FEATURE_TITLES, FEATURES, enabled_features
from localdoc_finder.core.models.benchmark import Verdict, judge
from localdoc_finder.core.models.catalog import ROLE_CHAT, ROLE_EMBED, Catalog
from localdoc_finder.core.models.fit import budget_mb, fits
from localdoc_finder.core.models.hardware import Hardware
from localdoc_finder.core.providers.base import ModelInfo
from localdoc_finder.core.settings import SettingsError
from localdoc_finder.core.setup.flow import (
    EnvironmentProbe,
    SetupEvent,
    SetupOptions,
    SetupPreview,
    SetupResult,
    SlowOffer,
    Stage,
)
from localdoc_finder.core.setup.ollama_install import INSTALLER_SIZE_MB, OllamaState
from localdoc_finder.core.setup.plan import EXTRA_ROLES, SetupChoices
from localdoc_finder.core.terms import TERMS_TEXT, TERMS_TITLE

WIZARD_TITLE = "Set up LocalDoc Finder"
WIZARD_SIZE = (680, 520)
FRONT_HOLD_MS = 3000
HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW = 0x0002, 0x0001, 0x0040
PROGRESS_SCALE = 100
_DOWNLOAD_STAGES = {Stage.OLLAMA, Stage.PLAN, Stage.DISK, Stage.PULL}  # before the speed test


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def ask_downgrade_with_dialog(offer: SlowOffer) -> bool:
    answer = QMessageBox.question(
        None,
        "Slow model",
        f"{offer.model} runs at {offer.result.rate:.1f} {offer.result.unit} on this PC.\n"
        f"Switch to the smaller {offer.alternative}?",
    )
    return answer == QMessageBox.StandardButton.Yes


class WelcomePage(QWizardPage):
    def __init__(self, hardware: Hardware) -> None:
        super().__init__()
        self.setTitle("Welcome")
        gpu = (
            f"{hardware.gpu_name} with {hardware.vram_total_mb} MB of video memory"
            if hardware.has_gpu
            else "no NVIDIA graphics card (models will run on the CPU)"
        )
        self.summary = _label(
            "LocalDoc Finder searches your files by meaning, and answers questions about them, "
            "entirely on this PC.\n\n"
            f"Your PC: {gpu}, {hardware.ram_total_mb} MB of memory.\n\n"
            "Setup will choose models that fit this hardware, download them, and test their "
            "speed. You can change everything."
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)


class TermsPage(QWizardPage):
    """The Terms and Conditions; Next stays off until they are accepted, and the choice is saved."""

    def __init__(self, record_acceptance: Callable[[], None]) -> None:
        super().__init__()
        self.setTitle(TERMS_TITLE)
        self._record = record_acceptance
        self.text = QTextBrowser()
        self.text.setPlainText(TERMS_TEXT)
        self.accept_box = QCheckBox("I have read and accept the Terms and Conditions")
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addWidget(self.accept_box)
        self.accept_box.toggled.connect(lambda _on: self.completeChanged.emit())

    def isComplete(self) -> bool:  # noqa: N802
        return self.accept_box.isChecked()

    def validatePage(self) -> bool:  # noqa: N802
        self._record()
        return True


class ScopePage(QWizardPage):
    """What to index. Defaults to the documents of the whole PC; saved when Next is pressed."""

    def __init__(self, controller: SettingsController) -> None:
        super().__init__()
        self.setTitle("What should be searchable?")
        self._controller = controller
        self.editor = ScopeEditor()
        self.status = _label()
        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addWidget(self.status)
        self.editor.changed.connect(self.completeChanged)

    def initializePage(self) -> None:  # noqa: N802
        self.editor.load(self._controller.settings().scope)

    def isComplete(self) -> bool:  # noqa: N802
        return self.editor.is_valid()

    def validatePage(self) -> bool:  # noqa: N802
        choice = self.editor.choice()
        try:
            self._controller.set_scope(
                choice.coverage, choice.roots, choice.file_types, custom_kinds=choice.kinds
            )
        except SettingsError as exc:
            self.status.setText(str(exc))
            return False
        return True


class OllamaPage(QWizardPage):
    """Ollama runs the models. Installing it needs the user's explicit tick."""

    def __init__(self, controller: SetupController) -> None:
        super().__init__()
        self.setTitle("Ollama")
        self._controller = controller
        self._needs_install = False
        self.status = _label()
        self.consent = QCheckBox(
            f"Download and install Ollama from ollama.com (about {INSTALLER_SIZE_MB} MB)"
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.consent)
        self.consent.toggled.connect(lambda _on: self.completeChanged.emit())
        self._controller.probed.connect(self._on_probed)

    def initializePage(self) -> None:  # noqa: N802
        self._needs_install = True  # not complete until we know: Next stays off while checking
        self.consent.setVisible(False)
        self.status.setText("Checking for Ollama…")
        self.completeChanged.emit()
        self._controller.probe_async()  # network and disk, so not on the UI thread

    def _on_probed(self, probe: EnvironmentProbe) -> None:
        state = probe.ollama
        self._needs_install = state is OllamaState.MISSING
        self.consent.setVisible(self._needs_install)
        self.status.setText(
            {
                OllamaState.RUNNING: "Ollama is installed and running.",
                OllamaState.INSTALLED_NOT_RUNNING: "Ollama is installed; setup will start it.",
                OllamaState.MISSING: "Ollama, the program that runs the models, is not installed. "
                "Nothing is downloaded until you agree below.",
            }[state]
        )
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return not self._needs_install or self.consent.isChecked()

    @property
    def install_consented(self) -> bool:
        return self._needs_install and self.consent.isChecked()


class ModelsPage(QWizardPage):
    """Pre-filled with the auto-picks; every catalog model is listed with its size."""

    def __init__(
        self,
        controller: SetupController,
        catalog: Catalog,
        hardware: Hardware,
        initial_features: Collection[str] = (),
    ) -> None:
        super().__init__()
        self.setTitle("Models")
        self._controller = controller
        self._catalog = catalog
        self._hardware = hardware
        self._preview: SetupPreview | None = None
        self._filled = False
        self.embed = QComboBox()
        self.chat = QComboBox()
        self.features = {name: QCheckBox(title) for name, title in FEATURE_TITLES.items()}
        for name, box in self.features.items():
            box.setChecked(name in initial_features)
        self.features_note = _label(
            "Search is always included. Ask, Chat and Match answer with a chat model, which is "
            "a larger download, so it is only fetched if you pick one of them. You can turn "
            "them on later in Settings."
        )
        self.chat_label = QLabel("Chat model (answers questions)")
        self.extras = {
            role: QCheckBox(f"Also download a {role.replace('_', ' ')} model")
            for role in EXTRA_ROLES
        }
        self.disk = _label()
        self.installed_note = _label()
        layout = QVBoxLayout(self)
        layout.addWidget(self.installed_note)
        layout.addWidget(QLabel("Embedding model (finds your files)"))
        layout.addWidget(self.embed)
        layout.addWidget(self.features_note)
        for box in self.features.values():
            layout.addWidget(box)
        layout.addWidget(self.chat_label)
        layout.addWidget(self.chat)
        for box in self.extras.values():
            layout.addWidget(box)
        layout.addWidget(self.disk)
        self._filling = False
        self.setCommitPage(True)  # downloading starts after this page, so there is no way back
        for combo in (self.embed, self.chat):
            combo.currentIndexChanged.connect(self._update_disk)
        for box in (*self.extras.values(), *self.features.values()):
            box.toggled.connect(self._update_disk)
        self._show_chat_picker()
        self._controller.probed.connect(self._on_probed)

    def _installed(self) -> set[str]:
        probe = self._controller.environment
        return {n.removesuffix(":latest") for n in probe.installed} if probe else set()

    def _fill(self, combo: QComboBox, role: str, selected: str | None) -> None:
        """Every catalog model, each marked installed or needs-download, then the user's own."""
        combo.clear()
        budget = budget_mb(self._hardware, total=True)
        installed = self._installed()
        listed: set[str] = set()
        for name in self._catalog.preferences(role):
            entry = self._catalog.entry(name)
            if entry is None:
                continue
            listed.add(name.removesuffix(":latest"))
            size = f"{entry.download_mb} MB" if entry.download_mb else "size unknown"
            state = (
                "already installed"
                if name.removesuffix(":latest") in installed
                else (f"needs download, {size}")
            )
            note = "" if fits(entry, entry.vram_mb, self._hardware, budget) else " - won't fit"
            combo.addItem(f"{name} - {state}{note}", name)
        for info in self._own_models(role):
            if info.name.removesuffix(":latest") not in listed:
                combo.addItem(f"{info.name} - already installed (your own model)", info.name)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(index, 0))

    def _own_models(self, role: str) -> list[ModelInfo]:
        """Installed models that are not in the catalog and suit ``role``."""
        probe = self._controller.environment
        if probe is None:
            return []
        wanted_embedding = role == ROLE_EMBED
        return [m for m in probe.models if m.is_embedding == wanted_embedding]

    def _describe_installed(self) -> None:
        probe = self._controller.environment
        names = sorted(probe.installed) if probe else []
        self.installed_note.setText(
            "Already on this PC (no download needed): " + ", ".join(names)
            if names
            else "No models are installed yet, so the models you pick will be downloaded."
        )

    def initializePage(self) -> None:  # noqa: N802
        self._preview = None
        self.disk.setText("Checking free disk space…")
        self.completeChanged.emit()
        if self._controller.environment is not None:
            self._on_probed(self._controller.environment)  # the Ollama page already looked
        else:
            self._controller.probe_async()

    def _on_probed(self, _probe: EnvironmentProbe) -> None:
        """The machine has been looked at: pre-fill the picks, then keep the numbers live."""
        if self._filled:  # a later probe only refreshes the disk numbers
            self._update_disk()
            return
        preview = self._controller.preview()
        if preview is None:
            return
        self._filling = True
        self._describe_installed()
        embed = preview.plan.model_for(ROLE_EMBED)
        # The chat model is auto-picked even while no feature is ticked, so ticking one is ready.
        with_chat = self._controller.preview(SetupChoices(features=FEATURES))
        chat = with_chat.plan.model_for(ROLE_CHAT) if with_chat else None
        self._fill(self.embed, ROLE_EMBED, embed.model if embed else None)
        self._fill(self.chat, ROLE_CHAT, chat.model if chat else None)
        self._filling = False
        self._filled = True
        self._update_disk()

    def _chosen_features(self) -> tuple[str, ...]:
        return tuple(name for name, box in self.features.items() if box.isChecked())

    def _show_chat_picker(self) -> None:
        """The chat model is only relevant (and only downloaded) when a feature uses it."""
        wanted = bool(self._chosen_features())
        self.chat_label.setVisible(wanted)
        self.chat.setVisible(wanted)

    def choices(self) -> SetupChoices:
        features = self._chosen_features()
        return SetupChoices(
            embed=self.embed.currentData(),
            chat=self.chat.currentData() if features else None,
            extras=tuple(role for role, box in self.extras.items() if box.isChecked()),
            features=features,
        )

    def _update_disk(self) -> None:
        if self._filling:
            return
        self._show_chat_picker()
        preview = self._controller.preview(self.choices())
        self._preview = preview
        if preview is None:
            self.disk.setText("Checking free disk space…")
            self.completeChanged.emit()
            return
        if preview.to_download:
            text = (
                f"Needs download: {', '.join(preview.to_download)}. To download: "
                f"{preview.download_mb} MB. Free disk space: {preview.free_disk_mb} MB."
            )
        else:
            text = "Nothing to download: the chosen models are already installed."
        if not preview.enough_disk:
            text += " Not enough free space; choose smaller models or free some space."
        text += "".join(f"\n{warning}" for warning in preview.plan.warnings)
        self.disk.setText(text)
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return self._preview is not None and self._preview.enough_disk


class RunPage(QWizardPage):
    """Shared by the download and speed-test pages: a progress bar, a log line and a retry."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setTitle(title)
        self.line = _label()
        self.bar = QProgressBar()
        self.bar.setRange(0, PROGRESS_SCALE)
        self.retry = QPushButton("Try again")
        self.retry.setVisible(False)
        layout = QVBoxLayout(self)
        layout.addWidget(self.line)
        layout.addWidget(self.bar)
        layout.addWidget(self.retry)
        self._done = False

    def show_event(self, event: SetupEvent) -> None:
        self.line.setText(event.message)
        if event.fraction is not None:
            self.bar.setValue(int(event.fraction * PROGRESS_SCALE))

    def mark_done(self) -> None:
        self._done = True
        self.completeChanged.emit()

    def show_failure(self, message: str) -> None:
        self.line.setText(f"Setup stopped: {message}\nIt keeps what it already downloaded.")
        self.retry.setVisible(True)

    def isComplete(self) -> bool:  # noqa: N802
        return self._done


class DownloadPage(RunPage):
    def __init__(self) -> None:
        super().__init__("Downloading")


class SpeedTestPage(RunPage):
    def __init__(self) -> None:
        super().__init__("Speed test")
        self.results = _label()
        self.layout().addWidget(self.results)  # type: ignore[union-attr]

    def show_result(self, result: SetupResult) -> None:
        lines = [
            f"{r.model}: {r.rate:.1f} {r.unit}"
            + ("  (slow on this PC)" if judge(r) is Verdict.SLOW else "")
            for r in result.bench
        ]
        lines.extend(result.warnings)
        self.results.setText("\n".join(lines) or "Speed test skipped.")


class SettingsPage(QWizardPage):
    """Every user setting, once: the same tabs as the Settings window."""

    def __init__(self, controller: SettingsController) -> None:
        super().__init__()
        self.setTitle("Your settings")
        self.general = GeneralTab(controller, show_scope=False)
        self.cloud = CloudTab(controller)
        self.updates = UpdatesTab(controller)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.general, "General")
        self.tabs.addTab(self.cloud, "Cloud && Privacy")  # && shows one &, not a shortcut
        self.tabs.addTab(self.updates, "Updates")
        self.status = _label()
        for tab in (self.general, self.cloud, self.updates):
            tab.message.connect(self.status.setText)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.status)

    def initializePage(self) -> None:  # noqa: N802
        for tab in (self.general, self.cloud, self.updates):
            tab.refresh()


class DonePage(QWizardPage):
    def __init__(self, hotkey: Callable[[], str]) -> None:
        super().__init__()
        self.setTitle("All set")
        self._hotkey = hotkey
        self.text = _label()
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)

    def initializePage(self) -> None:  # noqa: N802
        self.text.setText(
            f"Press {self._hotkey()} anywhere to search your files.\n\n"
            "Indexing happens in the background while the PC is idle and plugged in. "
            "Settings are in the tray icon's menu."
        )


class SetupWizard(QWizard):
    def __init__(
        self,
        controller: SetupController,
        settings_controller: SettingsController,
        catalog: Catalog,
        hardware: Hardware,
        *,
        ask_downgrade: Callable[[SlowOffer], bool] = ask_downgrade_with_dialog,
        record_terms: Callable[[], None] = lambda: None,
    ) -> None:
        super().__init__()
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setWindowTitle(WIZARD_TITLE)
        self.resize(*WIZARD_SIZE)
        self._controller = controller
        self._settings = settings_controller
        self._ask_downgrade = ask_downgrade
        self.welcome = WelcomePage(hardware)
        self.terms = TermsPage(record_terms)
        self.scope = ScopePage(settings_controller)
        self.ollama = OllamaPage(controller)
        self.models = ModelsPage(
            controller, catalog, hardware, enabled_features(settings_controller.settings())
        )
        self.download = DownloadPage()
        self.speed = SpeedTestPage()
        self.settings_page = SettingsPage(settings_controller)
        self.done_page = DonePage(lambda: settings_controller.settings().search.hotkey)
        self._pages = (
            self.welcome,
            self.terms,
            self.scope,
            self.ollama,
            self.models,
            self.download,
            self.speed,
            self.settings_page,
            self.done_page,
        )
        for page in self._pages:
            self.addPage(page)
        controller.progressed.connect(self._on_event)
        controller.finished.connect(self._on_finished)
        controller.failed.connect(self._on_failed)
        controller.downgrade_offered.connect(self._on_downgrade)
        self.download.retry.clicked.connect(self._start)

        self.speed.retry.clicked.connect(self._start)
        self.setButtonText(QWizard.WizardButton.CommitButton, "Download")
        self.currentIdChanged.connect(self._on_page)
        style_check_boxes(self, scheme_in_use())  # last: only reaches widgets that exist by now

    # ------------------------------------------------------------------ closing
    def reject(self) -> None:
        """Cancel / the window's close button: stop a running download instead of leaving it."""
        self._controller.cancel()
        super().reject()

    # ------------------------------------------------------------------ flow control
    def _on_page(self, page_id: int) -> None:
        if self.page(page_id) is self.download:
            self._start()

    def _start(self) -> None:
        self.download.retry.setVisible(False)
        self.speed.retry.setVisible(False)
        options = SetupOptions(
            self.models.choices(), install_ollama=self.ollama.install_consented, run_bench=True
        )
        self._controller.start(options)

    def _on_event(self, event: SetupEvent) -> None:
        if event.stage is Stage.BENCH and self.currentPage() is self.download:
            self.download.mark_done()
            self.next()
        in_download = event.stage in _DOWNLOAD_STAGES and not self.download.isComplete()
        (self.download if in_download else self.speed).show_event(event)

    def _on_finished(self, result: SetupResult) -> None:
        self.download.mark_done()
        self.speed.show_result(result)
        self.speed.mark_done()
        if self.currentPage() is self.download:
            self.next()

    def _on_failed(self, message: str) -> None:
        page = self.currentPage()
        if isinstance(page, RunPage):
            page.show_failure(message)

    def _on_downgrade(self, offer: SlowOffer) -> None:
        self._controller.answer_downgrade(self._ask_downgrade(offer))

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._bring_to_front)
        QTimer.singleShot(FRONT_HOLD_MS, self._release_front)

    def _bring_to_front(self) -> None:
        """Right after install the app may not take focus, so Windows can leave the wizard behind
        other windows with nothing to show it is running: sit on top until it has been seen."""
        self._set_topmost(True)
        self.raise_()
        self.activateWindow()
        QApplication.alert(self, 0)  # flashes the taskbar button until the user looks

    def _release_front(self) -> None:
        if self.isVisible():
            self._set_topmost(False)

    def _set_topmost(self, on: bool) -> None:
        """Win32 topmost without Qt's window flag, which rebuilds the wizard's buttons."""
        if sys.platform != "win32":
            return
        insert_after = HWND_TOPMOST if on else HWND_NOTOPMOST
        ctypes.windll.user32.SetWindowPos(  # type: ignore[attr-defined,unused-ignore]
            int(self.winId()), insert_after, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
        )
