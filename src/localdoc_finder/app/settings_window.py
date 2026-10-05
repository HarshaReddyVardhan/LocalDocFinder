"""The Settings window: one tab per concern, opened from the tray."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from localdoc_finder.app.cloud_tab import CloudTab
from localdoc_finder.app.models_controller import ModelsController
from localdoc_finder.app.models_panel import ModelsPanel
from localdoc_finder.app.settings_controller import SettingsController
from localdoc_finder.app.settings_features import FeaturesTab
from localdoc_finder.app.settings_tabs import (
    AboutTab,
    AdvancedTab,
    GeneralTab,
    IndexingTab,
    ModelsTab,
    SettingsTab,
    UpdatesTab,
)
from localdoc_finder.app.theme import scheme_in_use, style_check_boxes
from localdoc_finder.core.indexing_control import IndexingControl

WINDOW_TITLE = "LocalDoc Finder settings"
WINDOW_SIZE = (720, 560)
MIN_WINDOW_SIZE = (420, 320)  # smaller than any tab's content: a tab scrolls instead of growing


def tab_title(text: str) -> str:
    """Qt reads ``&`` as a keyboard-shortcut marker (shown as ``_``); ``&&`` is a literal one."""
    return text.replace("&", "&&")


def _scrolling(page: QWidget) -> QScrollArea:
    """A tab inside a scroll area, so its content can never force the window wider or taller."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(page)
    return area


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
        indexing: IndexingControl | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*WINDOW_SIZE)
        self.setMinimumSize(*MIN_WINDOW_SIZE)
        self.general = general or GeneralTab(controller)
        self.features = FeaturesTab(controller, models)
        self.models = ModelsTab(controller, ModelsPanel(models))
        self.cloud = cloud or CloudTab(controller)
        self.updates = UpdatesTab(controller)
        self.advanced = AdvancedTab(controller)
        self.about = about or AboutTab(controller)
        self.tabs = QTabWidget()
        self._pages: list[SettingsTab] = []
        self.indexing = IndexingTab(controller, indexing) if indexing else None
        tabs: list[tuple[str, SettingsTab]] = [("General", self.general)]
        if self.indexing:
            tabs.append(("Indexing", self.indexing))
        for title, page in (
            *tabs,
            ("Features", self.features),
            ("Models & Health", self.models),
            ("Cloud & Privacy", self.cloud),
            ("Updates", self.updates),
            ("Advanced", self.advanced),
            ("About", self.about),
        ):
            self.tabs.addTab(_scrolling(page), tab_title(title))
            self._pages.append(page)
            page.message.connect(self._show_status)
        self.general.hotkey_changed.connect(self.hotkey_changed)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status)
        style_check_boxes(self, scheme_in_use())

    def _show_status(self, text: str) -> None:
        self.status.setText(text)

    def open(self, page: SettingsTab | None = None) -> None:
        """Show the window with fresh data (settings may have changed since it last opened),
        on ``page`` if one is given."""
        for tab in self._pages:
            tab.refresh()
        if page is not None:
            self.tabs.setCurrentIndex(self._pages.index(page))
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self.models.panel.deactivate()
        super().closeEvent(event)  # type: ignore[arg-type]
