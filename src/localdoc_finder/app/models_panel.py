"""Models tab (installed models, role choices, one-click pull) and Health tab (dashboard)."""

import logging
from collections.abc import Callable, Sequence

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, SignalInstance
from PySide6.QtGui import QHideEvent, QPalette, QShowEvent, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
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

from localdoc_finder.app.models_controller import ModelsController
from localdoc_finder.app.theme import secondary_text
from localdoc_finder.core.models.catalog import (
    ROLE_CAPTION,
    ROLE_CHAT,
    ROLE_CODE_CHAT,
    ROLE_EMBED,
    ROLE_MATCH_SCORER,
    ROLE_RERANKER,
    ROLE_SUMMARIZER,
    ROLES,
)
from localdoc_finder.core.models.registry import BETTER_OPTION, FLAG_OK, Report
from localdoc_finder.core.providers.base import ProviderError

logger = logging.getLogger(__name__)

AUTOMATIC = "Automatic"
HEALTH_REFRESH_MS = 15_000  # the dashboard is a glance, not a monitor
MODEL_HEADERS = ["Model", "Size", "Used for", "Status"]
ROLE_HEADERS = ["Job", "Model", "Note"]
ROLE_COLUMN, CHOICE_COLUMN, NOTE_COLUMN = 0, 1, 2
# Each role as the job it does in the app, plus a tooltip that says where that job shows up.
ROLE_LABELS: dict[str, tuple[str, str]] = {
    ROLE_EMBED: (
        "Search",
        "Reads documents and queries for Search; changing it rebuilds the index",
    ),
    ROLE_CHAT: ("Answers", "Writes the answers in Ask and Chat, and the Match verdict"),
    ROLE_MATCH_SCORER: ("Match scoring", "Scores each requirement in Match"),
    ROLE_CODE_CHAT: ("Answers about code", "Ask and Chat when the question is about code"),
    ROLE_CAPTION: ("Image captions", "Not used by any feature yet"),
    ROLE_SUMMARIZER: ("Summaries", "Not used by any feature yet"),
    ROLE_RERANKER: ("Reranking", "Not used by any feature yet"),
}
# Jobs no feature asks for yet: hidden unless asked for, so the table shows only what runs.
UNUSED_ROLES = frozenset({ROLE_CAPTION, ROLE_SUMMARIZER, ROLE_RERANKER})
# The registry's short reasons, said in words a user knows.
REASON_LABELS = {
    "pinned": "changing it rebuilds the index",
    "override": "you chose it",
    "preferred": "best fit for this PC",
    "fallback: best installed model with the capability": "best installed model that can do it",
}
HINT_STRENGTH = 0.6  # how far the explanation under a heading fades toward the background
_GB = 1024**3
_MB_PER_GB = 1024
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
    text = f"For {role_label(role)}: {model} would be a better fit"
    return text if reason == BETTER_OPTION else f"{text} ({reason})"


def used_for_text(roles: Sequence[str]) -> str:
    """The jobs a model does; a job no feature asks for yet says so."""
    names = [
        f"{role_label(role)} (not used yet)" if role in UNUSED_ROLES else role_label(role)
        for role in roles
    ]
    return ", ".join(names) or "—"


def automatic_text(role: str, model: str | None, reason: str) -> str:
    """The first combo entry: automatic, naming the model it picks when automatic is in charge."""
    if role == ROLE_EMBED:
        return model or AUTOMATIC  # the index is built with one model: there is no "automatic"
    if model and reason not in ("override", "pinned"):
        return f"{AUTOMATIC} ({model})"
    return AUTOMATIC


def hardware_text(report: Report) -> str:
    hw = report.hardware
    gpu = (
        f"{hw.gpu_name}: {hw.vram_free_mb / _MB_PER_GB:.1f} of "
        f"{hw.vram_total_mb / _MB_PER_GB:.1f} GB video memory free"
        if hw.has_gpu
        else "No NVIDIA GPU (models run on the CPU)"
    )
    power = "plugged in" if hw.on_ac else "on battery"
    return f"{gpu} · {hw.ram_free_mb / _MB_PER_GB:.1f} GB RAM free · {power}"


class _ChoiceBox(QComboBox):
    """A combo box in a table: the mouse wheel scrolls the table, never changes the model."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


def _plain_table(headers: list[str]) -> QTableWidget:
    """A read-only table whose columns the user can drag wider; the last one fills the rest."""
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setWordWrap(False)
    return table


def _fit_to_text(table: QTableWidget) -> None:
    """Size each column to its text once, after filling; the user may still drag it after."""
    for column in range(table.columnCount() - 1):  # the last one stretches
        table.resizeColumnToContents(column)


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight:600; margin-top:6px;")
    return label


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    palette = label.palette()
    palette.setColor(QPalette.ColorRole.WindowText, secondary_text(palette, HINT_STRENGTH))
    label.setPalette(palette)
    return label


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

        self.roles = _plain_table(ROLE_HEADERS)
        # Every job at once: this table answers "which model does what" without scrolling.
        self.roles.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
        self.roles.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.show_unused = QCheckBox("Show jobs no feature uses yet")
        self.show_unused.toggled.connect(self._show_unused_roles)
        self.models = _plain_table(MODEL_HEADERS)
        self._fitted: set[QTableWidget] = set()  # sized to text once; then the user's widths stay
        self.recommendations = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        models_tab = QWidget()
        layout = QVBoxLayout(models_tab)
        layout.addLayout(top)
        layout.addWidget(_heading("Which model does each job"))
        layout.addWidget(
            _hint(
                "Automatic picks the best installed model that fits this PC. "
                "Pick a model to override it; hover a job to see where it is used."
            )
        )
        layout.addWidget(self.roles)
        layout.addWidget(self.show_unused)
        layout.addWidget(_heading("Installed models"))
        layout.addWidget(self.models, 2)
        layout.addWidget(_heading("Suggested downloads"))
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
        self.hardware.setText(hardware_text(report))
        self._fill_models(report)
        self._fill_roles(report)
        self._fill_recommendations(report)
        self.status_changed.emit(f"{len(report.rows)} models installed")

    def _fit_once(self, table: QTableWidget) -> None:
        if table not in self._fitted and table.rowCount():
            _fit_to_text(table)
            self._fitted.add(table)

    def _fill_models(self, report: Report) -> None:
        self.models.setRowCount(len(report.rows))
        for row, item in enumerate(report.rows):
            status = "; ".join(f.message for f in item.flags if f.kind != FLAG_OK) or "ok"
            cells = [
                item.info.name,
                f"{(item.info.size_bytes or 0) / _GB:.1f} GB",
                used_for_text(item.roles),
                status,
            ]
            for col, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setToolTip(text)
                self.models.setItem(row, col, cell)
        self._fit_once(self.models)

    def _fill_roles(self, report: Report) -> None:
        self._updating = True
        self.roles.setRowCount(len(ROLES))
        for row, role in enumerate(ROLES):
            resolution = report.resolutions[role]
            title, where = ROLE_LABELS.get(role, (role, ""))
            job = QTableWidgetItem(title)
            job.setToolTip(where)
            job.setData(Qt.ItemDataRole.UserRole, role)  # the id, for code and tests
            self.roles.setItem(row, ROLE_COLUMN, job)
            note = reason_text(resolution.reason)
            if role in UNUSED_ROLES:
                note = "not used by any feature yet"
            note_item = QTableWidgetItem(note)
            note_item.setToolTip(note)
            self.roles.setItem(row, NOTE_COLUMN, note_item)
            self.roles.setCellWidget(row, CHOICE_COLUMN, self._choice_box(role, report))
        self._show_unused_roles(self.show_unused.isChecked())
        self._fit_once(self.roles)
        self._updating = False

    def _choice_box(self, role: str, report: Report) -> QComboBox:
        resolution = report.resolutions[role]
        combo = _ChoiceBox()
        combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        combo.addItem(automatic_text(role, resolution.model, resolution.reason))
        for name in report.candidates.get(role, []):  # an embedder cannot chat, and so on
            if name != resolution.model or role != ROLE_EMBED:
                combo.addItem(name)
        if resolution.reason == "override" and resolution.model:
            combo.setCurrentText(resolution.model)
        combo.currentIndexChanged.connect(
            lambda index, r=role, box=combo: self._on_role_choice(
                r, AUTOMATIC if index == 0 and r != ROLE_EMBED else box.itemText(index)
            )
        )
        return combo

    def _show_unused_roles(self, show: bool) -> None:
        for row in range(self.roles.rowCount()):
            item = self.roles.item(row, ROLE_COLUMN)
            role = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            self.roles.setRowHidden(row, role in UNUSED_ROLES and not show)
        self.roles.updateGeometry()  # the fixed-height table grows and shrinks with its rows

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
