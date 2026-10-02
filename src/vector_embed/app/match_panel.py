"""Match panel: candidate checklist -> requirement checklist -> ranked results -> verdict.

Nothing is sent to a chat model until the user presses Score, and only ticked documents are
scored. The footer shows exactly what will be sent and where.
"""

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from vector_embed.app.match_controller import MatchController
from vector_embed.core.match.judge import MatchError
from vector_embed.core.match.pipeline import MatchRun
from vector_embed.core.match.scoring import Requirement
from vector_embed.core.providers.base import ProviderError

logger = logging.getLogger(__name__)

DOC_TYPES = ["resume", "cover_letter", "jd", "invoice", "paper", "plan", "notes", "other"]
PAGE_CANDIDATES, PAGE_CHECKLIST, PAGE_RESULTS = 0, 1, 2
CANDIDATE_HEADERS = ["", "File", "Folder", "Modified", "Versions", "Similarity", "Tokens", "Where"]
CHECKLIST_HEADERS = ["", "Requirement", "Type", "Weight"]
RESULT_HEADERS = ["#", "Score", "File", "Must-haves", "Notes"]
_KNOWN_ERRORS = (MatchError, ProviderError, RuntimeError, OSError)


class _Signals(QObject):
    done = Signal(str, object)  # task name, result
    failed = Signal(str, str)
    progress = Signal(str)
    delta = Signal(str)


class _Task(QRunnable):
    def __init__(self, name: str, fn: Callable[[], object], signals: _Signals) -> None:
        super().__init__()
        self._name, self._fn, self._signals = name, fn, signals

    def run(self) -> None:
        try:
            self._signals.done.emit(self._name, self._fn())
        except _KNOWN_ERRORS as exc:
            self._signals.failed.emit(self._name, str(exc))
        except Exception as exc:  # unexpected: surface it instead of dying in the pool
            logger.exception("match task %s crashed", self._name)
            self._signals.failed.emit(self._name, f"{type(exc).__name__}: {exc}")


class MatchPanel(QWidget):
    chat_requested = Signal(object)  # ChatState with pinned documents and scratch text
    status_changed = Signal(str)

    def __init__(
        self,
        controller: MatchController,
        pick_file: Callable[[], str | None] = lambda: None,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._pick_file = pick_file
        self._pool = pool or QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.done.connect(self._on_done)
        self._signals.failed.connect(self._on_failed)
        self._signals.progress.connect(self.status_changed)
        self._signals.delta.connect(self._on_delta)
        self._verdict = ""
        self._updating = False
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.doc_type = QComboBox()
        self.doc_type.addItems(DOC_TYPES)
        self.buttons = {
            name: QPushButton(label)
            for name, label in {
                "all": "Select all",
                "none": "Select none",
                "top3": "Top 3",
                "add": "+ Add file…",
                "back": "← Candidates",
                "checklist": "Checklist →",
                "score": "Score",
                "chat": "Chat ▶",
            }.items()
        }
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Find best match in:"))
        bar.addWidget(self.doc_type)
        for button in self.buttons.values():
            bar.addWidget(button)
        bar.addStretch(1)

        self.candidates = QTableWidget(0, len(CANDIDATE_HEADERS))
        self.candidates.setHorizontalHeaderLabels(CANDIDATE_HEADERS)
        self.checklist = QTableWidget(0, len(CHECKLIST_HEADERS))
        self.checklist.setHorizontalHeaderLabels(CHECKLIST_HEADERS)
        self.results = QTableWidget(0, len(RESULT_HEADERS))
        self.results.setHorizontalHeaderLabels(RESULT_HEADERS)
        self.verdict = QTextBrowser()
        results_page = QSplitter(Qt.Orientation.Vertical)
        results_page.addWidget(self.results)
        results_page.addWidget(self.verdict)
        self.pages = QStackedWidget()
        for page in (self.candidates, self.checklist, results_page):
            self.pages.addWidget(page)

        self.footer = QLabel("Paste a job description (Ctrl+V), then press Enter.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.pages, 1)
        layout.addWidget(self.footer)

        self.candidates.itemChanged.connect(self._candidate_edited)
        self.checklist.itemChanged.connect(self._checklist_edited)
        b = self.buttons
        b["all"].clicked.connect(lambda: self._tick_all(True))
        b["none"].clicked.connect(lambda: self._tick_all(False))
        b["top3"].clicked.connect(self._tick_top3)
        b["add"].clicked.connect(self.add_file)
        b["back"].clicked.connect(lambda: self._show_page(PAGE_CANDIDATES))
        b["checklist"].clicked.connect(self.request_checklist)
        b["score"].clicked.connect(self.request_score)
        b["chat"].clicked.connect(self.request_chat)
        self._show_page(PAGE_CANDIDATES)

    def _show_page(self, page: int) -> None:
        self.pages.setCurrentIndex(page)
        on_candidates = page == PAGE_CANDIDATES
        has_run = self._controller.run is not None
        for name in ("all", "none", "top3", "add", "checklist"):
            self.buttons[name].setVisible(on_candidates)
        self.buttons["back"].setVisible(page != PAGE_CANDIDATES)
        self.buttons["score"].setVisible(page != PAGE_RESULTS)
        self.buttons["chat"].setVisible(page == PAGE_RESULTS)
        for name in ("score", "checklist", "chat", "add"):
            self.buttons[name].setEnabled(has_run)

    def _say(self, text: str) -> None:
        self.footer.setText(text)
        self.status_changed.emit(text)

    # ------------------------------------------------------------------ step 1: recall
    def begin(self, jd_text: str) -> None:
        """Recall candidate documents for the pasted text (local embeddings only)."""
        self._say("finding candidates…")
        doc_type = self.doc_type.currentText()
        self._pool.start(
            _Task("recall", lambda: self._controller.start(jd_text, doc_type), self._signals)
        )

    def _fill_candidates(self, run: MatchRun) -> None:
        self._updating = True
        self.candidates.setRowCount(len(run.candidates))
        for row, candidate in enumerate(run.candidates):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(
                Qt.CheckState.Checked if candidate.selected else Qt.CheckState.Unchecked
            )
            self.candidates.setItem(row, 0, check)
            for col, text in enumerate(self._controller.candidate_row(candidate), 1):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.candidates.setItem(row, col, cell)
        self._updating = False
        self._show_page(PAGE_CANDIDATES)
        self._refresh_footer()

    def _refresh_footer(self) -> None:
        if self._controller.run is not None:
            self.footer.setText(self._controller.footer())

    def _candidate_edited(self, item: QTableWidgetItem) -> None:
        run = self._controller.run
        if self._updating or run is None or item.column() != 0:
            return
        run.candidates[item.row()].selected = item.checkState() == Qt.CheckState.Checked
        self._refresh_footer()

    def _tick_all(self, state: bool) -> None:
        run = self._controller.run
        if run is None:
            return
        for candidate in run.candidates:
            candidate.selected = state
        self._fill_candidates(run)

    def _tick_top3(self) -> None:
        if self._controller.run is not None:
            self._controller.top(3)
            self._fill_candidates(self._controller.run)

    def add_file(self) -> None:
        """Include a document that recall missed (picked with a file dialog)."""
        path = self._pick_file()
        if not path or self._controller.run is None:
            return
        try:
            self._controller.add_file(path)
        except _KNOWN_ERRORS as exc:
            self._say(str(exc))
            return
        self._fill_candidates(self._controller.run)

    # ------------------------------------------------------------------ step 2: checklist
    def request_checklist(self) -> None:
        self._say("reading the job description…")
        self._pool.start(_Task("checklist", self._controller.checklist, self._signals))

    def _fill_checklist(self, requirements: list[Requirement]) -> None:
        self._updating = True
        self.checklist.setRowCount(len(requirements))
        for row, req in enumerate(requirements):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(Qt.CheckState.Checked if req.enabled else Qt.CheckState.Unchecked)
            self.checklist.setItem(row, 0, check)
            text = QTableWidgetItem(req.text)
            text.setFlags(Qt.ItemFlag.ItemIsEnabled)
            kind = QTableWidgetItem(req.kind)
            kind.setFlags(Qt.ItemFlag.ItemIsEnabled)
            weight = QTableWidgetItem(f"{req.weight:g}")  # editable: re-weighting
            self.checklist.setItem(row, 1, text)
            self.checklist.setItem(row, 2, kind)
            self.checklist.setItem(row, 3, weight)
        self._updating = False
        self._show_page(PAGE_CHECKLIST)
        self._say("tick, untick or re-weight requirements, then press Score")

    def _checklist_edited(self, item: QTableWidgetItem) -> None:
        run = self._controller.run
        if self._updating or run is None or not run.requirements:
            return
        req = run.requirements[item.row()]
        if item.column() == 0:
            req.enabled = item.checkState() == Qt.CheckState.Checked
        elif item.column() == 3:
            try:
                req.weight = max(0.1, float(item.text()))
            except ValueError:
                self._updating = True
                item.setText(f"{req.weight:g}")
                self._updating = False

    # ------------------------------------------------------------------ step 3-4: score + verdict
    def request_score(self) -> None:
        run = self._controller.run
        if run is None:
            return
        self._say(self._controller.footer() + " — scoring…")
        self._pool.start(_Task("score", self._score, self._signals))

    def _score(self) -> object:
        scores = self._controller.score(self._signals.progress.emit)
        self._signals.progress.emit("writing the verdict…")
        parts: list[str] = []
        for delta in self._controller.verdict():
            parts.append(delta)
            self._signals.delta.emit(delta)
        return scores

    def _fill_results(self, run: MatchRun) -> None:
        ranked = run.ranked()
        self.results.setRowCount(len(ranked))
        for row, item in enumerate(ranked):
            breakdown = item.breakdown
            flag = " ⚠" if breakdown and breakdown.unverified else ""
            cut = " (reduced)" if item.judgement and item.judgement.reduced else ""
            cells = [
                str(row + 1),
                f"{item.score}{flag}" if breakdown else "—",
                item.candidate.name,
                breakdown.summary_line if breakdown else item.error,
                (item.judgement.summary if item.judgement else "") + cut,
            ]
            for col, text in enumerate(cells):
                self.results.setItem(row, col, QTableWidgetItem(text))
        self._show_page(PAGE_RESULTS)

    # ------------------------------------------------------------------ step 5: chat
    def request_chat(self) -> None:
        """Hand the JD, the best documents and their results to a follow-up chat."""
        try:
            state = self._controller.chat_state(top=3)
        except _KNOWN_ERRORS as exc:
            self._say(str(exc))
            return
        self.chat_requested.emit(state)

    # ------------------------------------------------------------------ task results
    def _on_done(self, name: str, result: object) -> None:
        run = self._controller.run
        if name == "recall" and run is not None:
            self._fill_candidates(run)
            if not run.candidates:
                self._say("no matching documents found; index some first")
        elif name == "checklist":
            self._fill_checklist(result)  # type: ignore[arg-type]
        elif name == "score" and run is not None:
            self._fill_results(run)
            self._say("done: press Chat to ask follow-up questions")

    def _on_delta(self, text: str) -> None:
        self._verdict += text
        self.verdict.setMarkdown(self._verdict)

    def _on_failed(self, name: str, message: str) -> None:
        self._say(f"{name} failed: {message}")

    def reset(self) -> None:
        self._verdict = ""
        self.verdict.clear()
        self.candidates.setRowCount(0)
        self._show_page(PAGE_CANDIDATES)
