"""The hotkey popup: a search bar and an Explorer-style result list, nothing else.

Search : type; Enter opens, Ctrl+Enter reveals, Shift+Enter opens in VS Code, Ctrl+T chats with it
Ask    : start with ``?`` or press Tab; Enter asks; answers stream with clickable [n] citations
Chat   : Tab again, or Ctrl+T on a result; Ctrl+V pastes long text as a scratch document
Match  : Tab again; paste a job description to rank your documents
Esc closes the window and unloads the model; Tab cycles the modes. The mode label and the answer
pane appear only outside Search. Models, health and settings live in the Settings window.
"""

import enum
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QHideEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from vector_embed.app.assistant import (
    AssistantService,
    ChatState,
    CloudPreview,
    Delta,
    Event,
    Failed,
    Finished,
)
from vector_embed.app.cloud_dialog import confirm_cloud_dialog
from vector_embed.app.controller import (
    Launcher,
    SearchOutcome,
    SearchService,
    foreground_title,
    guess_project,
    result_row,
)
from vector_embed.app.match_controller import MatchController
from vector_embed.app.match_panel import MatchPanel
from vector_embed.app.result_delegate import ROW_ROLE, ResultDelegate
from vector_embed.core.rag import Source
from vector_embed.core.skills.search import SearchResult

COMPACT_HEIGHT = 84  # just the search bar and the status line
EXPANDED_HEIGHT = 520
POPUP_WIDTH = 820
DEBOUNCE_MS = 180
MAINTAIN_MS = 15_000
STREAM_STOP_WAIT_SECONDS = 10.0  # how long ending a chat waits for the answer to stop
SCRATCH_MIN_CHARS = 200  # pasted text longer than this (or multi-line) becomes a scratch document
PLACEHOLDERS = {
    "search": "Search your files…   ? to ask   ·   Tab for modes",
    "ask": "Ask a question about your files…  (Enter to ask)",
    "chat": "Chat about the pinned documents…  (Enter to send, Ctrl+V pastes a document)",
    "match": "Paste a job description (Ctrl+V) and press Enter to rank your documents…",
}
STYLE = """
QWidget { background:#1e1f24; color:#e6e6e6; font-size:13px; }
QLineEdit { background:#2a2c33; border:1px solid #3b3e47; border-radius:6px;
            padding:9px 12px; font-size:16px; }
QListWidget { background:#1e1f24; border:none; outline:0; }
QListWidget::item { padding:6px 8px; border-bottom:1px solid #2a2c33; }
QListWidget::item:selected { background:#33405a; }
QTextBrowser { background:#17181c; border:1px solid #2a2c33; font-size:13px; }
QLabel#status { color:#8a8f9c; padding:2px 6px; }
QLabel#mode { color:#4c7dff; font-weight:bold; padding:0 8px; }
"""


class Mode(enum.Enum):
    SEARCH = "search"
    ASK = "ask"
    CHAT = "chat"
    MATCH = "match"


class _Signals(QObject):
    done = Signal(int, object)  # generation, SearchOutcome
    streamed = Signal(int, object)  # generation, assistant Event
    status = Signal(str)


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


class _StreamJob(QRunnable):
    """Runs an assistant generator off the UI thread, forwarding each event.

    ``cancel`` stops it at the next event and closes the generator, which closes the HTTP stream so
    the model stops generating; ``finished`` lets a caller wait before unloading the model.
    """

    def __init__(self, generation: int, events: Iterator[Event], signals: _Signals) -> None:
        super().__init__()
        self._generation = generation
        self._events = events
        self._signals = signals
        self._cancelled = threading.Event()
        self.finished = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            for event in self._events:
                if self._cancelled.is_set():
                    break
                self._signals.streamed.emit(self._generation, event)
        except Exception as exc:  # unexpected: show it rather than die silently in the pool
            self._signals.streamed.emit(self._generation, Failed(f"{type(exc).__name__}: {exc}"))
        finally:
            close = getattr(self._events, "close", None)
            if callable(close):
                close()
            self.finished.set()


class _CallJob(QRunnable):
    """Runs a blocking call (model load/unload) in the background and reports a status line."""

    def __init__(self, signals: _Signals, call: object, ok: str) -> None:
        super().__init__()
        self._signals, self._call, self._ok = signals, call, ok

    def run(self) -> None:
        try:
            result = self._call()  # type: ignore[operator]
            self._signals.status.emit(str(result) if result else self._ok)
        except Exception as exc:  # report, never crash the pool thread
            self._signals.status.emit(f"{type(exc).__name__}: {exc}")


class SearchWindow(QWidget):
    def __init__(
        self,
        service: SearchService,
        launcher: Launcher,
        thumbs_dir: Path,
        assistant: AssistantService | None = None,
        *,
        matcher: MatchController | None = None,
        pick_file: Callable[[], str | None] = lambda: None,
    ) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._service = service
        self._assistant = assistant
        self._matcher = matcher
        self._pick_file = pick_file
        self._jd_text = ""
        self._last_text = ""
        self.cloud_confirm: Callable[[CloudPreview], bool] = confirm_cloud_dialog
        self._launcher = launcher
        self._thumbs_dir = thumbs_dir
        self._generation = 0
        self._stream_job: _StreamJob | None = None
        self._results: list[SearchResult] = []
        self._sources: list[Source] = []
        self._project: str | None = None
        self._mode = Mode.SEARCH
        self._chat = ChatState()
        self._answer_text = ""
        self._pool = QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.done.connect(self._on_outcome)
        self._signals.streamed.connect(self._on_event)
        self._signals.status.connect(self._set_status)

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.run_search)
        self._maintain = QTimer(self)
        self._maintain.setInterval(MAINTAIN_MS)
        self._maintain.timeout.connect(self.maintain_model)
        self.input.textChanged.connect(self._on_text)
        self.list.currentRowChanged.connect(self._relayout_rows)
        self.list.itemActivated.connect(lambda _item: self.activate_selected())
        self.input.installEventFilter(self)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.dismiss)
        self._apply_mode()

    def reload_context(self) -> None:
        """Settings changed: end any chat (its model/route may differ) and rebuild lazily."""
        if self._mode is Mode.CHAT:
            self.set_mode(Mode.SEARCH)
        elif self._assistant is not None and self._assistant.session_active:
            self._end_chat("settings changed")
        self._service.reset()
        if self._assistant is not None:
            self._assistant.reset()
        if self._matcher is not None:
            self._matcher.reset_context()

    def _build_ui(self) -> None:
        self.setWindowTitle("Vector Embed")
        self.resize(POPUP_WIDTH, COMPACT_HEIGHT)
        self.setStyleSheet(STYLE)
        self.input = QLineEdit()
        self.mode_label = QLabel("SEARCH")
        self.mode_label.setObjectName("mode")
        top = QHBoxLayout()
        top.addWidget(self.input, 1)
        top.addWidget(self.mode_label)
        self.list = QListWidget()
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setItemDelegate(ResultDelegate(self._thumbs_dir, self.list))
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.answer = QTextBrowser()
        bottom = self._build_bottom_bar()

        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(self.answer)
        split.setSizes([520, 480])
        self.panel: MatchPanel | None = None
        self.body = QStackedWidget()
        self.body.addWidget(split)
        if self._matcher is not None:
            self.panel = MatchPanel(self._matcher, self._pick_file)
            self.panel.chat_requested.connect(self._chat_from_match)
            self.panel.status_changed.connect(self._set_status)
            self.body.addWidget(self.panel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 6)
        layout.addLayout(top)
        layout.addWidget(self.body, 1)
        layout.addLayout(bottom)

    def _build_bottom_bar(self) -> QHBoxLayout:
        """Status line plus the "Answer better" button (hidden unless a cloud is configured)."""
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.cloud_button = QPushButton("Answer better ☁")
        self.cloud_button.setVisible(False)
        self.cloud_button.clicked.connect(self.answer_better)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.cloud_button)
        return bottom

    # ------------------------------------------------------------------ modes
    @property
    def mode(self) -> Mode:
        return self._mode

    def available_modes(self) -> list[Mode]:
        modes = [Mode.SEARCH]
        if self._assistant is not None:
            modes += [Mode.ASK, Mode.CHAT]
        if self.panel is not None:
            modes.append(Mode.MATCH)
        return modes

    def _next_mode(self) -> Mode:
        modes = self.available_modes()
        return modes[(modes.index(self._mode) + 1) % len(modes)]

    def set_mode(self, mode: Mode) -> None:
        if mode is self._mode:
            return
        if mode not in self.available_modes():
            return
        leaving_chat = self._mode is Mode.CHAT
        leaving_match = self._mode is Mode.MATCH and mode is not Mode.CHAT  # chat reuses the model
        self._mode = mode
        self._generation += 1  # invalidates anything still streaming
        if not leaving_chat:  # ending a chat cancels the stream itself, and waits for it
            self._cancel_stream()
        self.list.clear()
        self._apply_mode()
        if leaving_chat:
            self._end_chat("left chat mode")
        elif leaving_match:
            self._end_chat("left match mode")
        if mode is Mode.CHAT:
            self._begin_chat()

    def _apply_mode(self) -> None:
        searching = self._mode is Mode.SEARCH
        self.mode_label.setText(self._mode.value.upper())
        self.mode_label.setVisible(not searching)
        self.input.setPlaceholderText(PLACEHOLDERS[self._mode.value])
        self.body.setCurrentIndex(self._body_index())
        self.cloud_button.setVisible(self._cloud_possible())
        self.answer.setVisible(not searching)
        self.answer.clear()
        self._answer_text = ""
        self._fit_height()
        self.input.setFocus()

    def _fit_height(self) -> None:
        """Search shows only the bar until there is something to list; other modes need room."""
        expanded = self._mode is not Mode.SEARCH or bool(self._results)
        self.body.setVisible(expanded)
        if (layout := self.layout()) is not None:
            layout.activate()  # drop the stale minimum height before shrinking
        self.resize(self.width(), EXPANDED_HEIGHT if expanded else COMPACT_HEIGHT)

    def _cloud_possible(self) -> bool:
        """The cloud button shows in Ask/Chat when a provider with a key is configured."""
        if self._assistant is None or self._mode not in (Mode.ASK, Mode.CHAT):
            return False
        try:
            return self._assistant.cloud_available()
        except Exception:  # building the context can fail (no index yet); the button is optional
            return False

    def _body_index(self) -> int:
        if self._mode is Mode.MATCH:
            return self.body.indexOf(self.panel) if self.panel is not None else 0
        return 0

    def _begin_chat(self) -> None:
        assert self._assistant is not None
        self.status.setText("loading the chat model…")
        self._pool.start(_CallJob(self._signals, self._assistant.begin_chat, "chat model ready"))

    def _start_stream(self, events: Iterator[Event]) -> None:
        self._cancel_stream()
        self._stream_job = _StreamJob(self._generation, events, self._signals)
        self._pool.start(self._stream_job)

    def _cancel_stream(self) -> _StreamJob | None:
        job, self._stream_job = self._stream_job, None
        if job is not None:
            job.cancel()
        return job

    def _end_chat(self, reason: str) -> None:
        stopping = self._cancel_stream()  # the answer must stop before its model is unloaded
        if self._assistant is not None:
            assistant = self._assistant

            def end() -> None:
                if stopping is not None:
                    stopping.finished.wait(STREAM_STOP_WAIT_SECONDS)
                assistant.end_chat(reason)

            self._pool.start(_CallJob(self._signals, end, ""))
        self._chat.reset()

    def _set_status(self, text: str) -> None:
        if text:
            self.status.setText(text)

    def maintain_model(self) -> None:
        """Unload the model when idle, unplugged or a fullscreen app starts."""
        if self._assistant is None:
            return
        service = self._assistant

        def check() -> str:
            reason = service.maintain()
            return f"chat model unloaded ({reason})" if reason else ""

        self._pool.start(_CallJob(self._signals, check, ""))

    # ------------------------------------------------------------------ show / hide
    def summon(self) -> None:
        self._project = guess_project(foreground_title())
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()
        self.input.selectAll()
        self._maintain.start()
        if self._mode is Mode.SEARCH:
            self._pool.start(_WarmJob(self._service))  # hides the model-load delay while typing
        mode = "  ·  on battery: searching on CPU" if self._on_battery() else ""
        self.status.setText((f"project: {self._project}" if self._project else "") + mode)

    def dismiss(self) -> None:
        """Esc: hide the window and release the chat model."""
        self.hide()
        self._maintain.stop()
        if self._mode is Mode.CHAT:
            self.set_mode(Mode.SEARCH)
        elif self._assistant is not None:
            self._end_chat("closed")

    def _on_battery(self) -> bool:
        try:
            return self._service.on_battery()
        except Exception:  # no index or model server yet: the status line is cosmetic
            return False

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        if self._assistant is not None:
            self._assistant.revoke_consent()  # consent never outlives the visible window

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        focus_lost = event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow()
        if focus_lost and self.isVisible():  # hide after a grace period for transient focus
            QTimer.singleShot(150, self._hide_if_inactive)

    def _hide_if_inactive(self) -> None:
        if not self.isActiveWindow():
            self.hide()  # the chat session (if any) stays; idle timeout unloads it later

    # ------------------------------------------------------------------ input
    def _on_text(self, text: str) -> None:
        if self._mode is Mode.SEARCH and text.startswith("?") and self._assistant is not None:
            self.input.blockSignals(True)
            self.input.setText(text[1:].lstrip())
            self.input.blockSignals(False)
            self.set_mode(Mode.ASK)
            return
        if self._mode is Mode.SEARCH:
            self._timer.start()

    def submit(self) -> None:
        """Enter in Ask/Chat mode: send the text to the model."""
        text = self.input.text().strip()
        if not text or self._assistant is None:
            return
        self._generation += 1
        self._last_text = text
        self._answer_text = ""
        self.answer.clear()
        self.status.setText("thinking…")
        if self._mode is Mode.ASK:
            events = self._assistant.ask(text)
        else:
            self.answer.setMarkdown(f"**You:** {text}\n\n")
            self._answer_text = f"**You:** {text}\n\n"
            events = self._assistant.chat(text, self._chat)
            self.input.clear()
        self._start_stream(events)

    def answer_better(self) -> None:
        """Re-ask the last question in the cloud, after showing exactly what would be sent."""
        assistant = self._assistant
        if assistant is None or not self._last_text or self._mode not in (Mode.ASK, Mode.CHAT):
            return
        question = self._last_text
        try:
            if self._mode is Mode.ASK:
                preview = assistant.cloud_preview_ask(question)
            else:
                preview = assistant.cloud_preview_chat(question, self._chat)
        except (
            Exception
        ) as exc:  # preview failures (e.g. private pinned file) are shown, not raised
            self.status.setText(str(exc))
            return
        if preview is None:
            self.status.setText("no cloud provider is configured (see: ve keys set)")
            return
        if not self.cloud_confirm(preview):
            self.status.setText("cancelled: nothing was sent")
            return
        self._generation += 1
        self._answer_text = ""
        self.answer.clear()
        self.status.setText(f"asking {preview.destination}…")
        if self._mode is Mode.ASK:
            events = assistant.ask_escalated(question)  # the very request that was previewed
        else:
            events = assistant.chat_escalated(question, self._chat)
        self._start_stream(events)

    def _on_event(self, generation: int, event: Event) -> None:
        if generation != self._generation:
            return
        if isinstance(event, Delta):
            self._answer_text += event.text
            self.answer.setMarkdown(self._answer_text)
            scrollbar = self.answer.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        elif isinstance(event, Finished):
            self._finish_answer(event)
        elif isinstance(event, Failed):
            self.answer.setMarkdown(f"**{event.message}**")
            self.status.setText(event.message)

    def _finish_answer(self, event: Finished) -> None:
        if event.note:
            self._answer_text += event.note
            self.answer.setMarkdown(self._answer_text)
        self._sources = event.sources
        self.list.clear()
        for source in event.sources:
            where = f" ({source.location})" if source.location else ""
            label = f"[{source.n}] {Path(source.path).name}{where}\n{source.path}"
            self.list.addItem(QListWidgetItem(label))
        suffix = f"  ·  {self._chat.describe()}" if self._mode is Mode.CHAT else ""
        self.status.setText("done" + suffix)

    # ------------------------------------------------------------------ search
    def run_search(self) -> None:
        query = self.input.text().strip()
        self._generation += 1
        if not query:
            self._results = []
            self.list.clear()
            self._fit_height()
            return
        job = _SearchJob(self._generation, query, self._project, self._service, self._signals)
        self._pool.start(job)

    def _on_outcome(self, generation: int, outcome: SearchOutcome) -> None:
        if generation != self._generation or self._mode is not Mode.SEARCH:
            return  # a newer query is already running
        self.show_results(outcome)

    def show_results(self, outcome: SearchOutcome) -> None:
        self._results = outcome.results
        self.list.clear()
        for result in outcome.results:
            row = result_row(result)
            item = QListWidgetItem(f"{row.name}\n{row.path}")  # the delegate draws from ROW_ROLE
            item.setData(ROW_ROLE, row)
            self.list.addItem(item)
        if outcome.results:
            self.list.setCurrentRow(0)
        self._fit_height()
        if outcome.message:
            self.status.setText(outcome.message)
        elif outcome.results:
            self.status.setText(f"{len(outcome.results)} results in {outcome.milliseconds:.0f} ms")
        else:
            self.status.setText("No results.")

    def _relayout_rows(self, _row: int) -> None:
        """The selected row grows to show its snippet, so row heights must be recomputed."""
        self.list.scheduleDelayedItemsLayout()

    # ------------------------------------------------------------------ actions
    def selected(self) -> SearchResult | None:
        row = self.list.currentRow()
        return self._results[row] if 0 <= row < len(self._results) else None

    def selected_source(self) -> Source | None:
        row = self.list.currentRow()
        return self._sources[row] if 0 <= row < len(self._sources) else None

    def activate_selected(self) -> None:
        """Enter on a list item: open the result, or jump to the cited source."""
        if self._mode is Mode.SEARCH:
            self.open_selected()
        elif (source := self.selected_source()) is not None:
            self._launcher.open_at(source.path, source.start_line)

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

    def chat_with_selected(self) -> None:
        """Ctrl+T on a search result: pin it and switch to Chat mode."""
        result = self.selected()
        if result is None or self._assistant is None or self._mode is not Mode.SEARCH:
            return
        self._chat.reset()
        self._chat.pinned = [result.path]
        self.set_mode(Mode.CHAT)
        self.status.setText(f"chatting with {Path(result.path).name}")

    def paste_scratch(self) -> bool:
        """Ctrl+V in Chat mode: long or multi-line text becomes a scratch document."""
        text = QGuiApplication.clipboard().text()
        if self._mode is not Mode.CHAT or not (len(text) > SCRATCH_MIN_CHARS or "\n" in text):
            return False
        self._chat.scratch = text
        self.status.setText(f"pasted document attached ({len(text)} chars)")
        return True

    def paste_job_description(self) -> bool:
        """Ctrl+V in Match mode: the pasted text is the job description."""
        text = QGuiApplication.clipboard().text()
        if self._mode is not Mode.MATCH or not text.strip():
            return False
        self._jd_text = text
        self.status.setText(
            f"pasted text attached ({len(text)} chars): press Enter to find matches"
        )
        return True

    def start_match(self) -> None:
        """Enter in Match mode: recall candidate documents for the pasted (or typed) text."""
        text = self._jd_text or self.input.text().strip()
        if not text or self.panel is None:
            self.status.setText("paste the text to match first (Ctrl+V)")
            return
        self._jd_text = text
        self.panel.reset()
        self.panel.begin(text)

    def _chat_from_match(self, state: ChatState) -> None:
        """Follow-up chat: pinned documents plus the job description and their scores."""
        self._chat = state
        self.set_mode(Mode.CHAT)
        self.status.setText("chatting about the match results")

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            return self._handle_key(event)
        return super().eventFilter(obj, event)

    def _handle_key(self, event: object) -> bool:
        key = event.key()  # type: ignore[attr-defined]
        modifiers = event.modifiers()  # type: ignore[attr-defined]
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Tab:
            self.set_mode(self._next_mode())
            return True
        if ctrl and key == Qt.Key.Key_T:
            self.chat_with_selected()
            return True
        if ctrl and key == Qt.Key.Key_V:
            return self.paste_scratch() or self.paste_job_description()
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            row = self.list.currentRow() + (1 if key == Qt.Key.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return self._handle_enter(ctrl, bool(modifiers & Qt.KeyboardModifier.ShiftModifier))
        return False

    def _handle_enter(self, ctrl: bool, shift: bool) -> bool:
        if self._mode is Mode.MATCH:
            self.start_match()
            return True
        if self._mode is not Mode.SEARCH:
            if ctrl:
                self.activate_selected()
            else:
                self.submit()
            return True
        if ctrl:
            self.reveal_selected()
        elif shift:
            self.code_selected()
        else:
            self.open_selected()
        return True
