"""Models tab (installed models, role choices, one-click pull) and Health tab (dashboard)."""

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, SignalInstance
from PySide6.QtGui import QHideEvent, QShowEvent, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vector_embed.app.models_controller import ModelsController
from vector_embed.core.models.catalog import (
    ROLE_CAPTION,
    ROLE_CHAT,
    ROLE_CODE_CHAT,
    ROLE_EMBED,
    ROLE_MATCH_SCORER,
    ROLE_RERANKER,
    ROLE_SUMMARIZER,
    ROLES,
)
from vector_embed.core.models.registry import BETTER_OPTION, FLAG_OK, Report
from vector_embed.core.providers.base import ProviderError

logger = logging.getLogger(__name__)

AUTOMATIC = "(automatic)"
HEALTH_REFRESH_MS = 15_000  # the dashboard is a glance, not a monitor
MODEL_HEADERS = ["Model", "Size", "Used as", "Status"]
ROLE_HEADERS = ["Role", "Used for", "Model in use", "Why", "Choose"]
ROLE_COLUMN, MODEL_COLUMN, CHOICE_COLUMN = 0, 2, 4
# What each role is called here and which feature asks for it, so the table says what runs where.
ROLE_LABELS: dict[str, tuple[str, str]] = {
    ROLE_EMBED: ("Embeddings", "Search"),
    ROLE_CHAT: ("Chat", "Ask, Chat, Match verdict"),
    ROLE_MATCH_SCORER: ("Match scoring", "Match requirements"),
    ROLE_CODE_CHAT: ("Code chat", "Ask and Chat about code"),
    ROLE_CAPTION: ("Image captions", "Not used yet"),
    ROLE_SUMMARIZER: ("Summaries", "Not used yet"),
    ROLE_RERANKER: ("Reranker", "Not used yet"),
}
# The registry's short reasons, said in words a user knows.
REASON_LABELS = {
    "pinned": "the index uses it",
    "override": "you chose it",
    "preferred": "automatic",
}
_GB = 1024**3
_KNOWN_ERRORS = (ProviderError, RuntimeError, OSError, ValueError)


class _Signals(QObject):
    report = Signal(object)
    health = Signal(str)
    pull_progress = Signal(str, float)
    pull_done = Signal(str)
    failed = Signal(str)  # models, roles and pulls
    health_failed = Signal(str)  # the dashboard has its own: its hiccups must not cancel a pull


class _Job(QRunnable):
    def __init__(
        self, fn: Callable[[], None], signals: _Signals, failed: SignalInstance | None = None
    ) -> None:
        super().__init__()
        self._fn, self._signals = fn, signals
        self._failed = failed if failed is not None else signals.failed

    def run(self) -> None:
        try:
            self._fn()
        except _KNOWN_ERRORS as exc:
            self._failed.emit(str(exc))
        except Exception as exc:  # unexpected: surface it rather than die in the pool
            logger.exception("models job crashed")
            self._failed.emit(f"{type(exc).__name__}: {exc}")


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, (role, ""))[0]


def reason_text(reason: str) -> str:
    """``qwen3-embedding cannot serve chat; preferred`` -> the same with the parts in words."""
    return "; ".join(REASON_LABELS.get(part, part) for part in reason.split("; "))


def recommendation_text(role: str, model: str, reason: str) -> str:
    text = f"{role_label(role)}: {model} would be a better fit"
    return text if reason == BETTER_OPTION else f"{text} ({reason})"


class _ChoiceBox(QComboBox):
    """A combo box in a table: the mouse wheel scrolls the table, never changes the model."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


def _fit_columns(table: QTableWidget, stretch: int) -> None:
    """Columns as wide as their text (names are never cut off); ``stretch`` takes the rest."""
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setWordWrap(False)


def _confirm_with_dialog(message: str) -> bool:
    answer = QMessageBox.question(None, "Change embedding model?", message)
    return answer == QMessageBox.StandardButton.Yes


class ModelsPanel(QWidget):
    status_changed = Signal(str)

    def __init__(
        self,
        controller: ModelsController,
        confirm: Callable[[str], bool] = _confirm_with_dialog,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._confirm = confirm
        self._pool = pool or QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.report.connect(self._show_report)
        self._signals.health.connect(self._show_health)
        self._signals.pull_progress.connect(self._on_progress)
        self._signals.pull_done.connect(self._on_pulled)
        self._signals.failed.connect(self._on_failed)
        self._signals.health_failed.connect(self._on_health_failed)
        self._health_running = False  # one dashboard refresh at a time
        self._updating = False
        self._pulling: set[str] = set()
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(HEALTH_REFRESH_MS)
        self._timer.timeout.connect(self.refresh_health)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.hardware = QLabel("")
        top = QHBoxLayout()
        top.addWidget(self.hardware, 1)
        top.addWidget(self.refresh_button)

        self.roles = QTableWidget(0, len(ROLE_HEADERS))
        self.roles.setHorizontalHeaderLabels(ROLE_HEADERS)
        _fit_columns(self.roles, stretch=3)
        # All seven roles at once: this table answers "which model does what" at a glance.
        self.roles.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
        self.roles.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.models = QTableWidget(0, len(MODEL_HEADERS))
        self.models.setHorizontalHeaderLabels(MODEL_HEADERS)
        _fit_columns(self.models, stretch=3)
        self.recommendations = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        models_tab = QWidget()
        layout = QVBoxLayout(models_tab)
        layout.addLayout(top)
        layout.addWidget(QLabel("Which model does what"))
        layout.addWidget(self.roles)
        layout.addWidget(QLabel("Installed models"))
        layout.addWidget(self.models, 2)
        layout.addWidget(QLabel("Recommendations"))
        layout.addLayout(self.recommendations)
        layout.addWidget(self.progress)

        self.health_view = QPlainTextEdit()
        self.health_view.setReadOnly(True)
        self.health_view.setStyleSheet("font-family:Consolas;")
        self.tabs = QTabWidget()
        self.tabs.addTab(models_tab, "Models")
        self.tabs.addTab(self.health_view, "Health")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.tabs)

    # ------------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        """Called when the mode is shown: load data and start refreshing the dashboard."""
        self.refresh()
        self._timer.start()

    def deactivate(self) -> None:
        self._timer.stop()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()  # refresh only while somebody can see the numbers

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        self._timer.stop()

    def refresh(self) -> None:
        self.status_changed.emit("reading installed models…")
        self._pool.start(
            _Job(lambda: self._signals.report.emit(self._controller.report()), self._signals)
        )
        self.refresh_health()

    def refresh_health(self) -> None:
        if self._health_running:  # a slow Ollama must not stack up refreshes behind it
            return
        self._health_running = True
        self._pool.start(
            _Job(
                lambda: self._signals.health.emit(self._controller.health_text()),
                self._signals,
                self._signals.health_failed,
            )
        )

    # ------------------------------------------------------------------ rendering
    def _show_health(self, text: str) -> None:
        self._health_running = False
        self.health_view.setPlainText(text)

    def _on_health_failed(self, message: str) -> None:
        self._health_running = False
        self.health_view.setPlainText(f"could not read the health report: {message}")

    def _show_report(self, report: Report) -> None:
        hw = report.hardware
        gpu = (
            f"{hw.gpu_name}: {hw.vram_free_mb}/{hw.vram_total_mb} MB VRAM free"
            if hw.has_gpu
            else "no NVIDIA GPU"
        )
        self.hardware.setText(
            f"{gpu} · {hw.ram_free_mb} MB RAM free · {'AC' if hw.on_ac else 'battery'}"
        )
        self._fill_models(report)
        self._fill_roles(report)
        self._fill_recommendations(report)
        self.status_changed.emit(f"{len(report.rows)} models installed")

    def _fill_models(self, report: Report) -> None:
        self.models.setRowCount(len(report.rows))
        for row, item in enumerate(report.rows):
            status = "; ".join(f.message for f in item.flags if f.kind != FLAG_OK) or "ok"
            cells = [
                item.info.name,
                f"{(item.info.size_bytes or 0) / _GB:.1f} GB",
                ", ".join(role_label(role) for role in item.roles) or "—",
                status,
            ]
            for col, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setToolTip(text)
                self.models.setItem(row, col, cell)

    def _fill_roles(self, report: Report) -> None:
        self._updating = True
        self.roles.setRowCount(len(ROLES))
        for row, role in enumerate(ROLES):
            resolution = report.resolutions[role]
            title, used_for = ROLE_LABELS.get(role, (role, ""))
            cells = [
                title,
                used_for,
                resolution.model or "(none)",
                reason_text(resolution.reason),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if col == ROLE_COLUMN:
                    item.setData(Qt.ItemDataRole.UserRole, role)  # the id, for code and tests
                self.roles.setItem(row, col, item)
            combo = _ChoiceBox()
            combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            combo.addItem(AUTOMATIC if role != ROLE_EMBED else (resolution.model or AUTOMATIC))
            for name in report.candidates.get(role, []):  # an embedder cannot chat, and so on
                if name != resolution.model or role != ROLE_EMBED:
                    combo.addItem(name)
            if resolution.reason == "override" and resolution.model:
                combo.setCurrentText(resolution.model)
            combo.currentTextChanged.connect(lambda text, r=role: self._on_role_choice(r, text))
            self.roles.setCellWidget(row, CHOICE_COLUMN, combo)
        self._updating = False

    def _fill_recommendations(self, report: Report) -> None:
        while self.recommendations.count():
            item = self.recommendations.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        if not report.recommendations:
            self.recommendations.addWidget(QLabel("Nothing to suggest: you have the best fit."))
        for rec in report.recommendations:
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(QLabel(recommendation_text(rec.role, rec.model, rec.reason)), 1)
            button = QPushButton(f"Pull {rec.model}")
            button.clicked.connect(lambda _checked=False, name=rec.model: self.pull(name))
            line.addWidget(button)
            self.recommendations.addWidget(row)

    # ------------------------------------------------------------------ actions
    def _on_role_choice(self, role: str, choice: str) -> None:
        if self._updating:
            return
        if role == ROLE_EMBED:
            self._change_embedder(choice)
            return
        model = None if choice == AUTOMATIC else choice
        self._pool.start(_Job(lambda: self._override(role, model), self._signals))

    def _override(self, role: str, model: str | None) -> None:
        self._controller.set_override(role, model)
        self._signals.report.emit(self._controller.report())

    def _change_embedder(self, model: str) -> None:
        notice = self._controller.embed_notice(model)
        if notice is None:
            return
        if not self._confirm(notice.message + " Continue?"):
            self.refresh()  # put the combo box back
            return
        self._controller.change_embedder(model)
        self.status_changed.emit("embedder changed: the index will be rebuilt when the PC is idle")
        self.refresh()

    def pull(self, name: str) -> None:
        """One-click pull with a progress bar."""
        if name in self._pulling:
            return
        self._pulling.add(name)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        def work() -> None:
            self._controller.pull(
                name, lambda p: self._signals.pull_progress.emit(p.status, p.fraction)
            )
            self._signals.pull_done.emit(name)

        self._pool.start(_Job(work, self._signals))

    def _on_progress(self, status: str, fraction: float) -> None:
        self.progress.setValue(int(fraction * 100))
        self.status_changed.emit(status)

    def _on_pulled(self, name: str) -> None:
        self._pulling.discard(name)
        self.progress.setVisible(False)
        self.status_changed.emit(f"pulled {name}")
        self.refresh()

    def _on_failed(self, message: str) -> None:
        self._pulling.clear()
        self.progress.setVisible(False)
        self.status_changed.emit(message)
