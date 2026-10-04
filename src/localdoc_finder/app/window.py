"""The hotkey popup: a search bar and an Explorer-style result list, nothing else.

Search : type; Enter opens, Ctrl+Enter reveals, Shift+Enter opens in VS Code, Ctrl+T chats with it
Ask    : start with ``?`` or press Tab; Enter asks; answers stream with clickable [n] citations
Chat   : Tab again, or Ctrl+T on a result; Ctrl+V pastes long text as a scratch document
Match  : Tab again; paste a job description to rank your documents
Esc closes the window and unloads the model; Tab (or a click on a pill) switches the mode. The
window is a rounded card that can be dragged anywhere and resized; Search hides when it loses the
focus, the other modes stay until Esc so an answer can be read beside another app. Models, health
and settings live in the Settings window.
"""

import enum
import re
import threading
import time
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from PySide6.QtCore import (
    QEvent,
    QMimeData,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QHideEvent,
    QKeySequence,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from localdoc_finder.app.assistant import (
    AssistantService,
    ChatState,
    CloudPreview,
    Delta,
    Event,
    Failed,
    Finished,
)
from localdoc_finder.app.cloud_dialog import confirm_cloud_dialog
from localdoc_finder.app.controller import (
    Launcher,
    SearchOutcome,
    SearchService,
    foreground_title,
    guess_project,
    result_row,
)
from localdoc_finder.app.match_controller import MatchController
from localdoc_finder.app.match_panel import MatchPanel
from localdoc_finder.app.mode_bar import ModeBar
from localdoc_finder.app.result_delegate import ROW_ROLE, ResultDelegate
from localdoc_finder.app.theme import (
    Scheme,
    card_colours,
    popup_style,
    scheme_in_use,
    search_icon,
    style_check_boxes,
)
from localdoc_finder.core.documents import DocumentError
from localdoc_finder.core.features import FEATURES
from localdoc_finder.core.rag import CODE_KINDS, Source
from localdoc_finder.core.skills.base import panel_skills
from localdoc_finder.core.skills.chat import SessionSummary
from localdoc_finder.core.skills.search import SearchResult

SHADOW = 18  # transparent margin around the card that the soft shadow is painted into
SHADOW_RINGS = 9  # rings of fading shadow; more is smoother and costs nothing noticeable
CARD_RADIUS = 14  # px; Windows 11 uses 8 for flyouts, launchers go a little rounder
EXPANDED_HEIGHT = 580  # including the shadow margin
POPUP_WIDTH = 860
MIN_WIDTH = 560
TOP_FRACTION = 0.18  # the card opens this far down the screen, where launchers sit
DEBOUNCE_MS = 180
_T = TypeVar("_T")
RENDER_MS = 80  # streamed text is re-rendered at most this often
MAINTAIN_MS = 15_000
STREAM_STOP_WAIT_SECONDS = 10.0  # how long ending a chat waits for the answer to stop
SCRATCH_MIN_CHARS = 200  # pasted text longer than this (or multi-line) becomes a scratch document
PLACEHOLDERS = {
    "search": "Search your files…   ? to ask   ·   Tab for modes",
    "ask": "Ask a question about your files…  (Enter to ask)",
    "chat": "Chat about the pinned documents…  (Enter to send, Ctrl+V pastes a document)",
    "match": "Paste a job description (Ctrl+V) and press Enter to rank your documents…",
}
CHAT_KEY_HINT = "   Ctrl+T chat"  # dropped from the Search hints while Chat is off
KEY_HINTS = {
    "search": f"↵ open   Ctrl+↵ reveal   Shift+↵ VS Code{CHAT_KEY_HINT}   Esc close",
    "ask": "↵ ask   Ctrl+↵ open source   Esc close",
    "chat": "↵ send   Ctrl+V paste a document   drop files to pin   Esc close",
    "match": "Ctrl+V paste   ↵ find matches   Esc close",
}
SKILL_KEY_HINTS = "↵ run   Esc close"
ANSWER_PLACEHOLDERS = {
    "ask": "Answers appear here, with [n] links to the files they came from.",
    "chat": "Drop files here, or press Ctrl+T on a search result, to chat about them.",
}


class Mode(enum.Enum):
    SEARCH = "search"
    ASK = "ask"
    CHAT = "chat"
    MATCH = "match"

    @property
    def title(self) -> str:
        return self.value.capitalize()


@dataclass(frozen=True)
class SkillMode:
    """A mode for any other panel skill in the registry: type, press Enter, read the answer."""

    value: str  # the skill's registry name, like ``Mode.value``
    title: str
    hint: str


AnyMode = Mode | SkillMode
CITE_SCHEME = "cite"
_CITATION = re.compile(r"(?<!\\)\[(\d+)\](?!\()")  # not escaped, and not a link's own text


def link_citations(text: str, sources: list[Source]) -> str:
    """Markdown with every ``[n]`` that names a real source turned into a clickable link."""
    known = {source.n for source in sources}

    def link(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number not in known:
            return match.group(0)
        return f"[\\[{number}\\]]({CITE_SCHEME}:{number})"

    return _CITATION.sub(link, text)


def dropped_files(mime: QMimeData) -> list[str]:
    """The local files (not folders or web links) in a drag."""
    if not mime.hasUrls():
        return []
    paths = (Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile())
    return [str(path) for path in paths if path.is_file()]  # native separators, as indexed


def skill_modes() -> list[SkillMode]:
    """Registered panel skills that have no hand-built mode, in registration order."""
    built_in = {mode.value for mode in Mode}
    return [
        SkillMode(skill.name, skill.title, f"{skill.description}  (Enter to run)")
        for skill in panel_skills()
        if skill.name not in built_in
    ]


class _Signals(QObject):
    done = Signal(int, object)  # generation, SearchOutcome
    streamed = Signal(int, object)  # generation, assistant Event
    status = Signal(str)
    cloud_checked = Signal(int, bool)  # mode-change token, is a cloud provider configured
    preview_ready = Signal(object, str)  # CloudPreview or None, error text


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


class _CloudProbeJob(QRunnable):
    """Asks whether a cloud provider is configured; building the context is not UI-thread work."""

    def __init__(self, token: int, assistant: AssistantService, signals: _Signals) -> None:
        super().__init__()
        self._token, self._assistant, self._signals = token, assistant, signals

    def run(self) -> None:
        try:
            available = self._assistant.cloud_available()
        except Exception:  # no index yet, say: the button is optional
            available = False
        self._signals.cloud_checked.emit(self._token, available)


class _PreviewJob(QRunnable):
    """Builds the "what will be sent" preview (retrieval and masking) off the UI thread."""

    def __init__(self, build: Callable[[], object], signals: _Signals) -> None:
        super().__init__()
        self._build, self._signals = build, signals

    def run(self) -> None:
        try:
            self._signals.preview_ready.emit(self._build(), "")
        except Exception as exc:  # e.g. a private pinned file: shown, not raised
            self._signals.preview_ready.emit(None, str(exc))


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
    settings_requested = Signal()  # the gear in the header: the app owns the Settings window

    def __init__(
        self,
        service: SearchService,
        launcher: Launcher,
        thumbs_dir: Path,
        assistant: AssistantService | None = None,
        *,
        matcher: MatchController | None = None,
        pick_file: Callable[[], str | None] = lambda: None,
        features: Callable[[], Collection[str]] = lambda: FEATURES,
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
        self._features = features  # asked each time: Settings can switch features on or off
        self._pick_file = pick_file
        self._jd_text = ""
        self._last_text = ""
        self.cloud_confirm: Callable[[CloudPreview], bool] = confirm_cloud_dialog
        self.choose_session: Callable[[list[SessionSummary]], int | None] = self._session_menu
        self._launcher = launcher
        self._thumbs_dir = thumbs_dir
        self._generation = 0
        self._dialogs = 0  # dialogs currently open on top of the popup
        self._stream_job: _StreamJob | None = None
        self._results: list[SearchResult] = []
        self._sources: list[Source] = []
        self._project: str | None = None
        self._mode: AnyMode = Mode.SEARCH
        self._skill_modes = skill_modes() if assistant is not None else []
        self._chat = ChatState()
        self._answer_text = ""
        self._pool = QThreadPool.globalInstance()
        self._signals = _Signals()
        self._signals.done.connect(self._on_outcome)
        self._signals.streamed.connect(self._on_event)
        self._signals.status.connect(self._set_status)
        self._signals.cloud_checked.connect(self._on_cloud_checked)
        self._signals.preview_ready.connect(self._on_preview)
        self._mode_token = 0  # which mode switch a cloud probe answers
        self._pending_cloud: tuple[AnyMode, str] | None = None
        self._last_render = 0.0
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(RENDER_MS)
        self._render_timer.timeout.connect(self._render_answer)

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
        if self._mode is Mode.CHAT or self._mode not in self.available_modes():
            self.set_mode(Mode.SEARCH)
        elif self._assistant is not None and self._assistant.session_active:
            self._end_chat("settings changed")
        self._service.reset()
        if self._assistant is not None:
            self._assistant.reset()
        if self._matcher is not None:
            self._matcher.reset_context()
        self._refresh_mode_bar()  # features may have been switched on or off in Settings

    def apply_scheme(self, scheme: Scheme) -> None:
        """Restyle the popup for light or dark (the Settings theme choice changed)."""
        self._scheme = scheme
        self.setStyleSheet(popup_style(scheme))
        style_check_boxes(self, scheme)  # the Match panel's check box
        if hasattr(self, "input"):
            self.input.removeAction(self._search_action)
            self._add_search_icon()
        self.update()

    def _build_ui(self) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # the card has round corners
        self._scheme = scheme_in_use()
        self._placed = False  # the user dragged the card somewhere: summon keeps it there
        self._drag_from: QPoint | None = None
        self._fitting = False  # a resize of our own, not the user's grip
        self._expanded_height = EXPANDED_HEIGHT
        self.setWindowTitle("LocalDoc Finder")
        self.setMinimumWidth(MIN_WIDTH)
        self.resize(POPUP_WIDTH, EXPANDED_HEIGHT)
        self.input = QLineEdit()
        self.input.setObjectName("query")
        self.input.setClearButtonEnabled(True)  # the ✕ at the end empties the box in one click
        self._add_search_icon()
        self.list = QListWidget()
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setItemDelegate(ResultDelegate(self._thumbs_dir, self.list))
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.answer = QTextBrowser()
        self.answer.setOpenLinks(False)  # citation links are handled here, not navigated to
        self.answer.anchorClicked.connect(self._on_link)
        self.input.setAcceptDrops(False)  # dropped files go to the window: they pin to a chat
        self.setAcceptDrops(True)

        split = QSplitter()
        split.setHandleWidth(10)
        split.addWidget(self.list)
        split.addWidget(self.answer)
        split.setSizes([320, 680])  # outside Search the answer matters most
        self.panel: MatchPanel | None = None
        self.body = QStackedWidget()
        self.body.addWidget(split)
        if self._matcher is not None:
            self.panel = MatchPanel(
                self._matcher,
                lambda: self._in_dialog(self._pick_file),
                confirm_cloud=lambda preview: self._in_dialog(lambda: self.cloud_confirm(preview)),
            )
            self.panel.chat_requested.connect(self._chat_from_match)
            self.panel.status_changed.connect(self._set_status)
            self.body.addWidget(self.panel)
        self._card = card = QFrame()
        card.setObjectName("card")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(14, 10, 14, 8)
        inner.setSpacing(8)
        inner.addLayout(self._build_header())
        inner.addWidget(self.input)
        inner.addWidget(self.body, 1)
        inner.addLayout(self._build_bottom_bar())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SHADOW, SHADOW, SHADOW, SHADOW)
        outer.addWidget(card)
        self.apply_scheme(self._scheme)

    def _add_search_icon(self) -> None:
        self._search_action = self.input.addAction(
            search_icon(self._scheme), QLineEdit.ActionPosition.LeadingPosition
        )

    def _build_header(self) -> QHBoxLayout:
        """Mode pills, a Tab key hint, the Settings gear and a close button."""
        self.mode_bar = ModeBar()
        self.mode_bar.chosen.connect(self._choose_mode)
        self.tab_hint = self._build_tab_hint()
        settings = self._header_button("⚙", "settingsButton", "Settings", self.open_settings)
        close = self._header_button("✕", "close", "Close (Esc)", self.dismiss)
        header = QHBoxLayout()
        header.setSpacing(6)
        header.addWidget(self.mode_bar)
        header.addStretch(1)  # empty header space is where the card is dragged from
        header.addWidget(self.tab_hint)
        header.addSpacing(6)
        header.addWidget(settings)
        header.addWidget(close)
        return header

    def _build_tab_hint(self) -> QFrame:
        """A key cap and its meaning in a tinted chip, set apart from the mode pills."""
        chip = QFrame()
        chip.setObjectName("tabHint")
        key = QLabel("Tab")
        key.setObjectName("keyCap")
        text = QLabel("to switch")
        text.setObjectName("tabHintText")
        row = QHBoxLayout(chip)
        row.setContentsMargins(5, 3, 10, 3)
        row.setSpacing(6)
        row.addWidget(key)
        row.addWidget(text)
        return chip

    def _header_button(
        self, glyph: str, name: str, tip: str, action: Callable[[], None]
    ) -> QPushButton:
        button = QPushButton(glyph)
        button.setObjectName(name)
        button.setToolTip(tip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # typing stays in the search bar
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(action)
        return button

    def open_settings(self) -> None:
        """Hide the popup (it stays on top) and ask the app for the Settings window."""
        self.dismiss()
        self.settings_requested.emit()

    def _build_bottom_bar(self) -> QHBoxLayout:
        """Status, the chat history and "Answer better" buttons, key hints and a resize grip."""
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.hints = QLabel("")
        self.hints.setObjectName("hints")
        self.cloud_button = QPushButton("Answer better ☁")
        self.cloud_button.setVisible(False)
        self.cloud_button.clicked.connect(self.answer_better)
        self.history_button = QPushButton("History ▾")
        self.history_button.setToolTip("Reopen an earlier conversation")
        self.history_button.setVisible(False)
        self.history_button.clicked.connect(self.show_history)
        grip = QSizeGrip(self)
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.history_button)
        bottom.addWidget(self.cloud_button)
        bottom.addWidget(self.hints)
        bottom.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        return bottom

    # ------------------------------------------------------------------ the card
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """A soft shadow, then the card: a faint top-to-bottom gloss, a hairline, a sheen."""
        colours = card_colours(self._scheme)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QRectF(self.rect()).adjusted(SHADOW, SHADOW, -SHADOW, -SHADOW)
        painter.setPen(Qt.PenStyle.NoPen)
        for ring in range(SHADOW_RINGS, 0, -1):  # outermost and faintest first
            spread = ring * SHADOW / SHADOW_RINGS
            shade = QColor(colours.shadow)
            shade.setAlphaF(colours.shadow.alphaF() * (1 - ring / (SHADOW_RINGS + 1)) / 3)
            painter.setBrush(shade)
            ring_rect = card.adjusted(-spread, -spread * 0.6, spread, spread * 1.2)
            painter.drawRoundedRect(ring_rect, CARD_RADIUS + spread, CARD_RADIUS + spread)
        gloss = QLinearGradient(card.topLeft(), card.bottomLeft())
        gloss.setColorAt(0, colours.top)
        gloss.setColorAt(1, colours.bottom)
        path = QPainterPath()
        path.addRoundedRect(card, CARD_RADIUS, CARD_RADIUS)
        painter.fillPath(path, gloss)
        painter.setPen(QPen(colours.edge, 1))
        painter.drawPath(path)
        painter.setPen(QPen(colours.highlight, 1))
        inset = CARD_RADIUS * 0.8
        sheen = card.top() + 1
        painter.drawLine(QPointF(card.left() + inset, sheen), QPointF(card.right() - inset, sheen))
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """A press on the card itself (not on a control) drags the window."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        # Moved by hand: Windows' move loop (startSystemMove) ignores this frameless tool window.
        self._placed = True
        self._drag_from = event.globalPosition().toPoint() - self.pos()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_from is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_from = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._fitting and self.body.isVisible():  # the user's grip: remember the height
            self._expanded_height = self.height()

    def _place(self) -> None:
        """Centred near the top of the screen under the mouse, unless the user moved it."""
        if self._placed:
            return
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        self.move(
            area.center().x() - self.width() // 2,
            area.top() + round(area.height() * TOP_FRACTION) - SHADOW,
        )

    # ------------------------------------------------------------------ modes
    @property
    def mode(self) -> AnyMode:
        return self._mode

    def all_modes(self) -> list[AnyMode]:
        """Every mode this window can show, switched on or not, in Tab order."""
        modes: list[AnyMode] = [Mode.SEARCH]
        if self._assistant is not None:
            modes += [Mode.ASK, Mode.CHAT]
        if self.panel is not None:
            modes.append(Mode.MATCH)
        return modes + list(self._skill_modes)

    def available_modes(self) -> list[AnyMode]:
        """Search always; Ask, Chat and Match only when the user switched them on."""
        enabled = self._features()
        return [m for m in self.all_modes() if self._is_on(m, enabled)]

    @staticmethod
    def _is_on(mode: AnyMode, enabled: Collection[str]) -> bool:
        return mode is Mode.SEARCH or isinstance(mode, SkillMode) or mode.value in enabled

    def _has_mode(self, mode: Mode) -> bool:
        return mode in self.available_modes()

    def _next_mode(self) -> AnyMode:
        modes = self.available_modes()
        return modes[(modes.index(self._mode) + 1) % len(modes)]

    def _choose_mode(self, value: str) -> None:
        """A pill was clicked."""
        mode = next((m for m in self.available_modes() if m.value == value), None)
        if mode is not None:
            self.set_mode(mode)
        self.mode_bar.set_current(self._mode.value)  # a refused switch keeps the old pill lit
        self.input.setFocus()

    def set_mode(self, mode: AnyMode) -> None:
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
        mode = self._mode
        self._refresh_mode_bar()
        self.status.clear()  # the last mode's message would read as this one's
        placeholder = mode.hint if isinstance(mode, SkillMode) else PLACEHOLDERS[mode.value]
        self.input.setPlaceholderText(placeholder)
        self.body.setCurrentIndex(self._body_index())
        self._check_cloud_async()
        self.list.setVisible(searching)  # elsewhere it lists sources, once there are some
        self.answer.setVisible(not searching)
        self.answer.setPlaceholderText(ANSWER_PLACEHOLDERS.get(mode.value, ""))
        self.history_button.setVisible(mode is Mode.CHAT)
        self.answer.clear()
        self._answer_text = ""
        self._fit_height()
        self.input.setFocus()

    def _refresh_mode_bar(self) -> None:
        """Pills and key hints for the modes on; with only Search there is nothing to switch."""
        modes = self.available_modes()
        every = self.all_modes()
        self.mode_bar.set_modes([(m.value, m.title, m in modes) for m in every])
        self.mode_bar.set_current(self._mode.value)
        self.mode_bar.setVisible(len(every) > 1)  # switched-off modes show greyed out
        self.tab_hint.setVisible(len(modes) > 1)  # nothing to switch to with Search alone
        if isinstance(self._mode, SkillMode):
            self.hints.setText(SKILL_KEY_HINTS)
            return
        hint = KEY_HINTS[self._mode.value]
        if self._mode is Mode.SEARCH and not self._has_mode(Mode.CHAT):
            hint = hint.replace(CHAT_KEY_HINT, "")
        self.hints.setText(hint)

    def _fit_height(self) -> None:
        """Search shows only the bar until there is something to list; other modes need room."""
        expanded = self._mode is not Mode.SEARCH or bool(self._results)
        self.body.setVisible(expanded)
        for layout in (self._card.layout(), self.layout()):  # inner first: the outer reads it
            if layout is not None:
                layout.activate()  # drop the stale minimum height before shrinking
        height = self._expanded_height if expanded else self.compact_height()
        self._fitting = True
        try:
            self.resize(self.width(), height)
        finally:
            self._fitting = False

    def compact_height(self) -> int:
        """Just the pills, the search bar and the status line (with the body hidden)."""
        layout = self.layout()
        return layout.sizeHint().height() if layout is not None else EXPANDED_HEIGHT

    def _check_cloud_async(self) -> None:
        """The cloud button shows in Ask/Chat when a provider with a key is configured.

        Found out in the background: the answer may need the whole skill context built first.
        """
        self.cloud_button.setVisible(False)
        self._mode_token += 1
        if self._assistant is None or self._mode not in (Mode.ASK, Mode.CHAT):
            return
        self._pool.start(_CloudProbeJob(self._mode_token, self._assistant, self._signals))

    def _on_cloud_checked(self, token: int, available: bool) -> None:
        if token == self._mode_token:  # not a stale answer for a mode we already left
            self.cloud_button.setVisible(available and self._mode in (Mode.ASK, Mode.CHAT))

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
        if not self.isVisible() and not service.session_active:
            self._maintain.stop()  # nothing to watch: no window, no loaded model
            return

        def check() -> str:
            reason = service.maintain()
            return f"chat model unloaded ({reason})" if reason else ""

        self._pool.start(_CallJob(self._signals, check, ""))

    # ------------------------------------------------------------------ show / hide
    def summon(self) -> None:
        self._project = guess_project(foreground_title())
        self._place()
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
        # A dialog opened from here (cloud preview, file picker) takes the focus: the popup must
        # not vanish from under it. Only Search behaves like a launcher; an answer, a chat or a
        # match is read beside other windows (copying a job description from a browser), so those
        # stay until Esc.
        if self._mode is not Mode.SEARCH:
            return
        if self.isActiveWindow() or self._dialogs or QApplication.activeModalWidget() is not None:
            return
        self.hide()  # the chat session (if any) stays; idle timeout unloads it later

    # ------------------------------------------------------------------ chat history
    def show_history(self) -> None:
        """Pick an earlier conversation from a menu and continue it."""
        if self._assistant is None or self._mode is not Mode.CHAT:
            return
        try:
            sessions = self._assistant.recent_sessions()
        except RuntimeError as exc:
            self.status.setText(f"⚠ {exc}")
            return
        if not sessions:
            self.status.setText("no earlier conversations yet")
            return
        chosen = self._in_dialog(lambda: self.choose_session(sessions))
        if chosen is not None:
            self.reopen_session(chosen)

    def _session_menu(self, sessions: list[SessionSummary]) -> int | None:
        menu = QMenu(self)
        for summary in sessions:
            action = menu.addAction(summary.line().strip())
            action.setData(summary.id)
        picked = menu.exec(self.history_button.mapToGlobal(self.history_button.rect().topLeft()))
        return int(picked.data()) if picked is not None else None

    def reopen_session(self, session_id: int) -> None:
        """Show an earlier conversation; the next message continues it."""
        if self._assistant is None:
            return
        self._generation += 1
        self._cancel_stream()
        try:
            transcript = self._assistant.reopen(session_id, self._chat)
        except RuntimeError as exc:
            self.status.setText(f"⚠ {exc}")
            return
        self._answer_text = transcript
        self.answer.setMarkdown(transcript)
        details = self._chat.describe()
        self.status.setText(f"reopened chat {session_id}" + (f"  ·  {details}" if details else ""))

    def _in_dialog(self, call: Callable[[], _T]) -> _T:
        """Run something that opens a dialog, keeping the popup visible meanwhile."""
        self._dialogs += 1
        try:
            return call()
        finally:
            self._dialogs -= 1
            self.activateWindow()  # the focus returns here when the dialog closes

    def shutdown(self) -> None:
        """The app is quitting: stop any answer and unload every model, now.

        Runs on the UI thread and waits: the process is about to end, so a background job
        would never finish, and the GPU would be left holding the model.
        """
        self._maintain.stop()
        job = self._cancel_stream()
        if job is not None:
            job.finished.wait(STREAM_STOP_WAIT_SECONDS)
        if self._matcher is not None:
            self._matcher.revoke_cloud_consent()
        if self._assistant is not None:
            self._assistant.end_chat("quitting")
        self._service.release()

    # ------------------------------------------------------------------ input
    def _on_text(self, text: str) -> None:
        if self._mode is Mode.SEARCH and text.startswith("?") and self._has_mode(Mode.ASK):
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
        elif isinstance(self._mode, SkillMode):
            events = self._assistant.run_skill(self._mode.value, text)
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
        mode, chat = self._mode, self._chat
        self._pending_cloud = (mode, question)
        self.status.setText("preparing what will be sent…")
        self.cloud_button.setEnabled(False)
        build = (
            (lambda: assistant.cloud_preview_ask(question))
            if mode is Mode.ASK
            else (lambda: assistant.cloud_preview_chat(question, chat))
        )
        self._pool.start(_PreviewJob(build, self._signals))

    def _on_preview(self, preview: object, error: str) -> None:
        """The preview is ready: show it, and send only what the user approves."""
        self.cloud_button.setEnabled(True)
        pending, self._pending_cloud = self._pending_cloud, None
        assistant = self._assistant
        if pending is None or assistant is None:
            return
        mode, question = pending
        if mode is not self._mode:  # the user moved on while it was being prepared
            return
        if error:
            self.status.setText(error)
            return
        if preview is None:
            self.status.setText("no cloud provider is configured (see: ldf keys set)")
            return
        assert isinstance(preview, CloudPreview)
        if not self._in_dialog(lambda: self.cloud_confirm(preview)):
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
            self._schedule_render()
        elif isinstance(event, Finished):
            self._render_now()
            self._finish_answer(event)
        elif isinstance(event, Failed):
            self._render_timer.stop()
            self.answer.setMarkdown(f"**{event.message}**")
            self.status.setText(event.message)

    def _schedule_render(self) -> None:
        """Re-parsing the whole answer as Markdown per token is slow: render at most every 80 ms.

        The first token of a burst shows at once; the rest is flushed by a trailing timer.
        """
        if (
            time.monotonic() - self._last_render >= RENDER_MS / 1000
            and not self._render_timer.isActive()
        ):
            self._render_answer()
        elif not self._render_timer.isActive():
            self._render_timer.start()

    def _render_now(self) -> None:
        self._render_timer.stop()
        self._render_answer()

    def _render_answer(self) -> None:
        self._last_render = time.monotonic()
        self.answer.setMarkdown(self._answer_text)
        scrollbar = self.answer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _finish_answer(self, event: Finished) -> None:
        if event.note:
            self._answer_text += event.note
            self._render_now()
        self._sources = event.sources
        if event.sources:  # now the [n] markers can link to their sources
            self.answer.setMarkdown(link_citations(self._answer_text, event.sources))
            scrollbar = self.answer.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        self.list.clear()
        self.list.setVisible(bool(event.sources))
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
            self.open_source(source)

    def open_source(self, source: Source) -> None:
        """Code opens in the editor at its line; documents (PDF, Word, images) in their app."""
        if source.kind in CODE_KINDS:
            self._launcher.open_at(source.path, source.start_line)
        else:
            self._launcher.open_path(source.path)

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() != CITE_SCHEME:
            return
        number = int(url.path()) if url.path().isdigit() else -1
        source = next((s for s in self._sources if s.n == number), None)
        if source is not None:
            self.open_source(source)

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
        if result is None or not self._has_mode(Mode.CHAT) or self._mode is not Mode.SEARCH:
            return
        self._chat.reset()
        self._chat.pinned = [result.path]
        self.set_mode(Mode.CHAT)
        self.status.setText(f"chatting with {Path(result.path).name}")

    # ------------------------------------------------------------------ drag and drop
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._has_mode(Mode.CHAT) and dropped_files(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = dropped_files(event.mimeData())
        if not self._has_mode(Mode.CHAT) or not paths:
            return
        event.acceptProposedAction()
        self.pin_files(paths)

    def pin_files(self, paths: list[str]) -> None:
        """Files dropped on the popup: chat about them (added to the chat already open)."""
        assert self._assistant is not None
        names = ", ".join(Path(p).name for p in paths)
        if self._mode is not Mode.CHAT:
            self._chat.reset()
            self._chat.pinned = list(paths)
            self.set_mode(Mode.CHAT)
            self.status.setText(f"chatting with {names}")
            return
        try:
            self._assistant.pin(paths, self._chat)
        except (DocumentError, RuntimeError) as exc:
            self.status.setText(f"⚠ {exc}")
            return
        self.status.setText(f"pinned {names}  ·  {self._chat.describe()}")

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
