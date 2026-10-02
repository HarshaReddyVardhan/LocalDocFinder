"""The hotkey popup: type to search, Enter opens, Ctrl+Enter reveals, Shift+Enter opens in VS Code.

Keys: Enter open · Ctrl+Enter reveal in Explorer · Shift+Enter code -g file:line · Esc hide
"""

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vector_embed.app.controller import (
    Launcher,
    SearchOutcome,
    SearchService,
    foreground_title,
    guess_project,
    result_label,
)
from vector_embed.core.extractors.image import thumbnail_path
from vector_embed.core.skills.search import SearchResult

DEBOUNCE_MS = 180
PLACEHOLDER = (
    "Search code, notes, PDFs, images…   type:code  proj:name  ext:py  in:D:\\x  after:2026-01"
)
STYLE = """
QWidget { background:#1e1f24; color:#e6e6e6; font-size:13px; }
QLineEdit { background:#2a2c33; border:1px solid #3b3e47; border-radius:6px;
            padding:9px 12px; font-size:16px; }
QListWidget { background:#1e1f24; border:none; outline:0; }
QListWidget::item { padding:6px 8px; border-bottom:1px solid #2a2c33; }
QListWidget::item:selected { background:#33405a; }
QPlainTextEdit { background:#17181c; border:1px solid #2a2c33; font-family:Consolas;
                 font-size:12px; }
QLabel#status { color:#8a8f9c; padding:2px 6px; }
"""


class _Signals(QObject):
    done = Signal(int, object)  # generation, SearchOutcome


class _SearchJob(QRunnable):
    def __init__(
        self,
        generation: int,
        query: str,
        project: str | None,
        service: SearchService,
        signals: _Signals,
    ) -> None:
        super().__init__()
        self._args = (generation, query, project, service, signals)

    def run(self) -> None:
        generation, query, project, service, signals = self._args
        signals.done.emit(generation, service.search(query, project))


class _WarmJob(QRunnable):
    def __init__(self, service: SearchService) -> None:
        super().__init__()
        self._service = service

    def run(self) -> None:
        try:
            self._service.warm()
        except Exception:  # warm-up is an optimisation only
            return


class SearchWindow(QWidget):
    def __init__(self, service: SearchService, launcher: Launcher, thumbs_dir: Path) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._service = service
        self._launcher = launcher
        self._thumbs_dir = thumbs_dir
        self._generation = 0
        self._results: list[SearchResult] = []
        self._project: str | None = None
        self._pool = QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.done.connect(self._on_outcome)

        self.setWindowTitle("Vector Embed")
        self.resize(1000, 560)
        self.setStyleSheet(STYLE)
        self.input = QLineEdit()
        self.input.setPlaceholderText(PLACEHOLDER)
        self.list = QListWidget()
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pane = QStackedWidget()
        self.pane.addWidget(self.preview)
        self.pane.addWidget(self.image)
        self.status = QLabel("")
        self.status.setObjectName("status")

        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(self.pane)
        split.setSizes([520, 480])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 6)
        layout.addWidget(self.input)
        layout.addWidget(split, 1)
        layout.addWidget(self.status)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.run_search)
        self.input.textChanged.connect(lambda _text: self._timer.start())
        self.list.currentRowChanged.connect(self._show_preview)
        self.list.itemActivated.connect(lambda _item: self.open_selected())
        self.input.installEventFilter(self)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.hide)

    # ------------------------------------------------------------------ show / hide
    def summon(self) -> None:
        self._project = guess_project(foreground_title())
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()
        self.input.selectAll()
        self._pool.start(_WarmJob(self._service))  # hides the model-load delay while typing
        mode = "  ·  on battery: searching on CPU" if self._on_battery() else ""
        self.status.setText((f"project: {self._project}" if self._project else "") + mode)

    def _on_battery(self) -> bool:
        try:
            return self._service.on_battery()
        except Exception:  # no index or model server yet: the status line is cosmetic
            return False

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        focus_lost = event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow()
        if focus_lost and self.isVisible():  # hide after a grace period for transient focus
            QTimer.singleShot(150, self._hide_if_inactive)

    def _hide_if_inactive(self) -> None:
        if not self.isActiveWindow():
            self.hide()

    # ------------------------------------------------------------------ search
    def run_search(self) -> None:
        query = self.input.text().strip()
        self._generation += 1
        if not query:
            self.list.clear()
            self.preview.clear()
            return
        job = _SearchJob(self._generation, query, self._project, self._service, self._signals)
        self._pool.start(job)

    def _on_outcome(self, generation: int, outcome: SearchOutcome) -> None:
        if generation != self._generation:
            return  # a newer query is already running
        self.show_results(outcome)

    def show_results(self, outcome: SearchOutcome) -> None:
        self._results = outcome.results
        self.list.clear()
        for result in outcome.results:
            self.list.addItem(QListWidgetItem(result_label(result)))
        if outcome.results:
            self.list.setCurrentRow(0)
        else:
            self.preview.setPlainText(outcome.message or "No results.")
            self.pane.setCurrentIndex(0)
        self.status.setText(
            outcome.message or f"{len(outcome.results)} results in {outcome.milliseconds:.0f} ms"
        )

    def _show_preview(self, row: int) -> None:
        if not 0 <= row < len(self._results):
            return
        result = self._results[row]
        if result.kind == "image" and not result.page:
            try:
                pixmap = QPixmap(str(thumbnail_path(self._thumbs_dir, Path(result.path))))
            except OSError:
                pixmap = QPixmap()
            if not pixmap.isNull():
                self.image.setPixmap(pixmap)
                self.pane.setCurrentIndex(1)
                return
        self.preview.setPlainText(result.text)
        self.pane.setCurrentIndex(0)

    # ------------------------------------------------------------------ actions
    def selected(self) -> SearchResult | None:
        row = self.list.currentRow()
        return self._results[row] if 0 <= row < len(self._results) else None

    def open_selected(self) -> None:
        if (result := self.selected()) is not None:
            self.hide()
            self._launcher.open_file(result)

    def reveal_selected(self) -> None:
        if (result := self.selected()) is not None:
            self.hide()
            self._launcher.reveal(result)

    def code_selected(self) -> None:
        if (result := self.selected()) is not None:
            self.hide()
            self._launcher.open_in_editor(result)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            return self._handle_key(event)
        return super().eventFilter(obj, event)

    def _handle_key(self, event: object) -> bool:
        key = event.key()  # type: ignore[attr-defined]
        modifiers = event.modifiers()  # type: ignore[attr-defined]
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            row = self.list.currentRow() + (1 if key == Qt.Key.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                self.reveal_selected()
            elif modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.code_selected()
            else:
                self.open_selected()
            return True
        return False
