"""The Settings window: one tab per concern, opened from the tray."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from vector_embed.app.models_controller import ModelsController
from vector_embed.app.models_panel import ModelsPanel
from vector_embed.app.settings_controller import SettingsController
from vector_embed.app.settings_tabs import (
    AboutTab,
    CloudTab,
    GeneralTab,
    ModelsTab,
    SettingsTab,
    UpdatesTab,
)

WINDOW_TITLE = "Vector Embed settings"
WINDOW_SIZE = (720, 560)


class SettingsWindow(QWidget):
    hotkey_changed = Signal(str)

    def __init__(
        self,
        controller: SettingsController,
        models: ModelsController,
        *,
        general: GeneralTab | None = None,
        cloud: CloudTab | None = None,
        about: AboutTab | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*WINDOW_SIZE)
        self.general = general or GeneralTab(controller)
        self.models = ModelsTab(controller, ModelsPanel(models))
        self.cloud = cloud or CloudTab(controller)
        self.updates = UpdatesTab(controller)
        self.about = about or AboutTab(controller)
        self.tabs = QTabWidget()
        self._pages: list[SettingsTab] = []
        for title, page in (
            ("General", self.general),
            ("Models & Health", self.models),
            ("Cloud & Privacy", self.cloud),
            ("Updates", self.updates),
            ("About", self.about),
        ):
            self.tabs.addTab(page, title)
            self._pages.append(page)
            page.message.connect(self._show_status)
        self.general.hotkey_changed.connect(self.hotkey_changed)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status)

    def _show_status(self, text: str) -> None:
        self.status.setText(text)

    def open(self) -> None:
        """Show the window with fresh data (settings may have changed since it last opened)."""
        for page in self._pages:
            page.refresh()
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self.models.panel.deactivate()
        super().closeEvent(event)  # type: ignore[arg-type]
