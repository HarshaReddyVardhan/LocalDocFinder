from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant, wait_for
from tests.core.conftest import Chat, Env

from localdoc_finder.app.assistant import AssistantService, ChatState, Event
from localdoc_finder.app.controller import Launcher
from localdoc_finder.app.window import Mode, SearchWindow
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.skills.chat import SessionSummary


def test_reopening_restores_the_pins_and_returns_the_conversation(
    env: Env, skill_ctx: SkillContext, chat: Chat
) -> None:
    path = env.root / "notes.md"
    path.write_text("# Notes\n\nRetry with backoff.\n", encoding="utf-8")
    service = AssistantService(lambda: skill_ctx)
    first = ChatState(pinned=[str(path)])
    chat.client.chat_reply = ["Use backoff."]
    list(service.chat("how do we retry?", first))
    assert first.session_id is not None

    (summary,) = service.recent_sessions()
    assert (summary.id, summary.messages) == (first.session_id, 2)
    later = ChatState()
    transcript = service.reopen(summary.id, later)
    assert transcript == "**You:** how do we retry?\n\nUse backoff.\n\n"
    assert later.session_id == first.session_id
    assert later.pinned == [str(path)]

    with pytest.raises(RuntimeError, match="no chat session 999"):
        service.reopen(999, ChatState())


class HistoryAssistant(FakeAssistant):
    def recent_sessions(self) -> list[SessionSummary]:
        return [SessionSummary(7, "retry plan", 0.0, 2)]

    def reopen(self, session_id: int, state: ChatState) -> str:
        state.session_id, state.pinned = session_id, ["D:/notes.md"]
        return "**You:** old question\n\nold answer\n\n"

    def chat(self, message: str, state: ChatState) -> Iterator[Event]:
        self.calls.append(("chat", (message, state.session_id)))
        yield from self.events


@pytest.fixture
def window(qapp: QApplication, tmp_path: Path) -> tuple[SearchWindow, HistoryAssistant]:
    assistant = HistoryAssistant()
    win = SearchWindow(
        FakeService(),  # type: ignore[arg-type]
        Launcher(),
        tmp_path,
        assistant,  # type: ignore[arg-type]
    )
    return win, assistant


def test_history_is_offered_only_in_chat_mode(
    window: tuple[SearchWindow, HistoryAssistant],
) -> None:
    win, _ = window
    win.show()
    assert not win.history_button.isVisible()
    win.set_mode(Mode.CHAT)
    assert win.history_button.isVisible()
    win.set_mode(Mode.ASK)
    assert not win.history_button.isVisible()


def test_a_reopened_chat_shows_its_history_and_continues_it(
    qapp: QApplication, window: tuple[SearchWindow, HistoryAssistant]
) -> None:
    win, assistant = window
    win.show()
    win.set_mode(Mode.CHAT)
    offered: list[list[SessionSummary]] = []
    win.choose_session = lambda sessions: offered.append(sessions) or 7  # type: ignore[func-returns-value]
    win.history_button.click()
    assert [s.id for s in offered[0]] == [7]
    assert "old answer" in win.answer.toPlainText()
    assert "reopened chat 7" in win.status.text() and "1 file(s) pinned" in win.status.text()

    win.input.setText("and now?")
    win.submit()
    wait_for(qapp, lambda: ("chat", ("and now?", 7)) in assistant.calls)
    assert ("chat", ("and now?", 7)) in assistant.calls


def test_dismissing_the_menu_changes_nothing(
    window: tuple[SearchWindow, HistoryAssistant],
) -> None:
    win, _ = window
    win.set_mode(Mode.CHAT)
    win.choose_session = lambda _sessions: None
    win.show_history()
    assert win._chat.session_id is None
