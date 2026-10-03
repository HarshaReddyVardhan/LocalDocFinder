"""Clickable citations, documents opening in their own app, and files dropped onto the popup."""

from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant, source, wait_for
from tests.core.conftest import Chat, Env

from vector_embed.app.assistant import AssistantService, ChatState, Delta, Finished
from vector_embed.app.controller import Launcher
from vector_embed.app.window import Mode, SearchWindow, dropped_files, link_citations
from vector_embed.core.documents import DocumentError
from vector_embed.core.rag import Source
from vector_embed.core.skills.base import SkillContext


def pdf_source(n: int = 2) -> Source:
    return Source(n, r"D:\docs\policy.pdf", "docs", "doc", "", 0, 0, 4, "refunds")


class OpeningLauncher(Launcher):
    def __init__(self) -> None:
        super().__init__(startfile=lambda p: self.opened.append(("app", p, 0)))
        self.opened: list[tuple[str, str, int]] = []

    def open_at(self, path: str, line: int = 0) -> None:
        self.opened.append(("editor", path, line))


class PinningAssistant(FakeAssistant):
    def __init__(self) -> None:
        super().__init__()
        self.refuse = False

    def pin(self, paths: list[str], state: ChatState) -> None:
        if self.refuse:
            raise DocumentError("secrets.env looks like a secret and is never read")
        state.pinned = [*state.pinned, *paths]


@pytest.fixture
def parts(
    qapp: QApplication, tmp_path: Path
) -> tuple[SearchWindow, PinningAssistant, OpeningLauncher]:
    launcher = OpeningLauncher()
    assistant = PinningAssistant()
    window = SearchWindow(
        FakeService(),  # type: ignore[arg-type]
        launcher,
        tmp_path,
        assistant,  # type: ignore[arg-type]
    )
    return window, assistant, launcher


def test_only_citations_of_real_sources_become_links() -> None:
    text = r"Use backoff [1][2]. See [9], not \[1\] or [1](http://x)."
    linked = link_citations(text, [source(1), pdf_source(2)])
    assert r"[\[1\]](cite:1)[\[2\]](cite:2)" in linked
    assert "[9]" in linked and "cite:9" not in linked
    assert r"\[1\] or [1](http://x)" in linked  # escaped text and real links are left alone


def test_a_clicked_citation_opens_code_in_the_editor_and_a_pdf_in_its_app(
    qapp: QApplication, parts: tuple[SearchWindow, PinningAssistant, OpeningLauncher]
) -> None:
    window, assistant, launcher = parts
    assistant.events = [
        Delta("Retry with backoff [1], refunds in [2]."),
        Finished([source(1), pdf_source(2)]),
    ]
    window.show()
    window.set_mode(Mode.ASK)
    window.input.setText("how?")
    window.submit()
    wait_for(qapp, lambda: bool(window._sources))
    html = window.answer.toHtml()
    assert 'href="cite:1"' in html and 'href="cite:2"' in html

    window.answer.anchorClicked.emit(QUrl("cite:2"))
    window.answer.anchorClicked.emit(QUrl("cite:1"))
    window.answer.anchorClicked.emit(QUrl("cite:7"))  # no such source: ignored
    window.answer.anchorClicked.emit(QUrl("https://example.com"))  # not a citation: ignored
    assert launcher.opened == [("app", r"D:\docs\policy.pdf", 0), ("editor", r"D:\p\a.py", 7)]


def drop(window: SearchWindow, paths: list[Path]) -> None:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dropEvent(event)


def test_dropped_files_start_a_chat_about_them(
    tmp_path: Path, parts: tuple[SearchWindow, PinningAssistant, OpeningLauncher]
) -> None:
    window, assistant, _ = parts
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF")
    drop(window, [report, tmp_path])  # the folder is ignored
    assert window.mode is Mode.CHAT
    assert window._chat.pinned == [str(report)]
    assert "chatting with report.pdf" in window.status.text()

    notes = tmp_path / "notes.md"
    notes.write_text("x", encoding="utf-8")
    drop(window, [notes])  # already chatting: added to the conversation
    assert window._chat.pinned == [str(report), str(notes)]
    assert "pinned notes.md" in window.status.text()

    assistant.refuse = True
    drop(window, [notes])
    assert "never read" in window.status.text()


def test_only_local_files_count_as_dropped(tmp_path: Path) -> None:
    real = tmp_path / "a.txt"
    real.write_text("x", encoding="utf-8")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(real)), QUrl("https://example.com/b.pdf")])
    assert dropped_files(mime) == [str(real)]
    assert dropped_files(QMimeData()) == []


def test_pinning_into_a_running_session_is_saved_and_checked(
    env: Env, skill_ctx: SkillContext, chat: Chat
) -> None:
    first = env.root / "a.md"
    first.write_text("# A\n\nalpha\n", encoding="utf-8")
    second = env.root / "b.md"
    second.write_text("# B\n\nbravo\n", encoding="utf-8")
    service = AssistantService(lambda: skill_ctx)
    state = ChatState(pinned=[str(first)])
    list(service.chat("hello", state))
    assert state.session_id is not None
    service.pin([str(second)], state)
    assert state.pinned == [str(first), str(second)]
    context = skill_ctx.state.session_context(state.session_id)
    assert context["pinned"] == [str(first), str(second)]

    secret = env.root / ".env"
    secret.write_text("API_KEY=abc", encoding="utf-8")
    with pytest.raises(DocumentError):
        service.pin([str(secret)], state)
    assert skill_ctx.state.session_context(state.session_id)["pinned"] == state.pinned
