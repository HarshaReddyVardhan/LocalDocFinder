import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService, result

from vector_embed.app.assistant import ChatState, Delta, Event, Failed, Finished
from vector_embed.app.controller import Launcher
from vector_embed.app.window import Mode, SearchWindow, _Signals, _StreamJob
from vector_embed.core.rag import Source


def source(n: int = 1, path: str = r"D:\p\a.py", line: int = 7) -> Source:
    return Source(n, path, "p", "code", "f", line, line + 3, 0, "body")


class FakeAssistant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.events: list[Event] = [
            Delta("Hello "),
            Delta("world [1]."),
            Finished([source(1), source(2, r"D:\p\b.py", 20)], "\n\nSources:\n[1] a"),
        ]
        self.maintain_result: str | None = None

    def begin_chat(self) -> None:
        self.calls.append(("begin", None))

    def end_chat(self, reason: str = "closed") -> None:
        self.calls.append(("end", reason))

    def maintain(self) -> str | None:
        self.calls.append(("maintain", None))
        return self.maintain_result

    def ask(self, question: str) -> Iterator[Event]:
        self.calls.append(("ask", question))
        yield from self.events

    def chat(self, message: str, state: ChatState) -> Iterator[Event]:
        self.calls.append(("chat", (message, state.session_id, list(state.pinned), state.scratch)))
        state.session_id = 42
        yield from self.events


def wait_for(qapp: QApplication, condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def parts(
    qapp: QApplication, tmp_path: Path
) -> tuple[SearchWindow, FakeAssistant, list[tuple[str, str, int]]]:
    opened: list[tuple[str, str, int]] = []
    launcher = Launcher(startfile=lambda p: opened.append(("open", p, 0)))
    launcher.open_at = lambda path, line=0: opened.append(("at", path, line))  # type: ignore[method-assign]
    assistant = FakeAssistant()
    window = SearchWindow(
        FakeService([result(path=r"D:\p\a.py")]),  # type: ignore[arg-type]
        launcher,
        tmp_path,
        assistant,  # type: ignore[arg-type]
    )
    return window, assistant, opened


def kinds(assistant: FakeAssistant) -> list[str]:
    return [kind for kind, _ in assistant.calls]


class TestModeSwitching:
    def test_tab_cycles_search_ask_chat(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.show()
        QTest.keyClick(window.input, Qt.Key.Key_Tab)
        assert window.mode is Mode.ASK
        QTest.keyClick(window.input, Qt.Key.Key_Tab)
        assert window.mode is Mode.CHAT
        wait_for(qapp, lambda: "begin" in kinds(assistant))
        assert "begin" in kinds(assistant)
        QTest.keyClick(window.input, Qt.Key.Key_Tab)
        assert window.mode is Mode.SEARCH
        wait_for(qapp, lambda: "end" in kinds(assistant))
        assert ("end", "left chat mode") in assistant.calls

    def test_without_an_assistant_tab_keeps_search_mode(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        window = SearchWindow(FakeService(), Launcher(), tmp_path)  # type: ignore[arg-type]
        window.show()
        QTest.keyClick(window.input, Qt.Key.Key_Tab)
        assert window.mode is Mode.SEARCH
        window.input.setText("?not a question")
        assert window.mode is Mode.SEARCH

    def test_question_mark_prefix_switches_to_ask(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.input.setText("?how do retries work")
        assert window.mode is Mode.ASK
        assert window.input.text() == "how do retries work"
        assert window.mode_label.text() == "ASK"

    def test_mode_labels_and_placeholders_follow_the_mode(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.set_mode(Mode.CHAT)
        assert window.mode_label.text() == "CHAT"
        assert "pinned" in window.input.placeholderText()
        window.set_mode(Mode.CHAT)  # same mode: nothing happens


class TestAsk:
    def test_answer_streams_and_sources_are_listed_and_clickable(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, opened = parts
        window.set_mode(Mode.ASK)
        window.input.setText("how do retries work")
        QTest.keyClick(window.input, Qt.Key.Key_Return)
        wait_for(qapp, lambda: window.list.count() == 2)
        assert ("ask", "how do retries work") in assistant.calls
        text = window.answer.toPlainText()
        assert "Hello world [1]." in text
        assert "Sources:" in text
        assert window.status.text() == "done"
        assert "[1] a.py (lines 7-10)" in window.list.item(0).text()
        window.list.setCurrentRow(1)
        QTest.keyClick(window.input, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
        assert opened == [("at", r"D:\p\b.py", 20)]
        window.list.setCurrentRow(0)
        window.list.itemActivated.emit(window.list.item(0))
        assert opened[-1] == ("at", r"D:\p\a.py", 7)

    def test_empty_question_is_ignored(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.set_mode(Mode.ASK)
        window.submit()
        assert assistant.calls == []

    def test_failures_are_shown(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        assistant.events = [Failed("on battery: plug in to chat")]
        window.set_mode(Mode.ASK)
        window.input.setText("q")
        window.submit()
        wait_for(qapp, lambda: "plug in" in window.status.text())
        assert "plug in" in window.answer.toPlainText()

    def test_stale_events_are_dropped(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.set_mode(Mode.ASK)
        window._generation = 3
        window._on_event(2, Delta("old"))
        assert window.answer.toPlainText() == ""
        window._on_event(3, Delta("new"))
        assert window.answer.toPlainText() == "new"

    def test_search_results_arriving_in_another_mode_are_ignored(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.set_mode(Mode.ASK)
        from vector_embed.app.controller import SearchOutcome

        window._on_outcome(window._generation, SearchOutcome([result()], 1.0))
        assert window.list.count() == 0


class TestChat:
    def test_messages_share_one_session_and_show_the_conversation(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.set_mode(Mode.CHAT)
        window.input.setText("what is missing?")
        window.submit()
        wait_for(qapp, lambda: window.status.text().startswith("done"))
        assert window.input.text() == ""
        assert "You: what is missing?" in window.answer.toPlainText()
        window.input.setText("rewrite my bullets")
        window.submit()
        wait_for(qapp, lambda: len([c for c in assistant.calls if c[0] == "chat"]) == 2)
        chats = [payload for kind, payload in assistant.calls if kind == "chat"]
        assert chats[0][1] is None  # type: ignore[index]
        assert chats[1][1] == 42  # type: ignore[index]

    def test_ctrl_t_pins_the_selected_result_and_opens_chat(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.show_results(
            __import__("vector_embed.app.controller", fromlist=["SearchOutcome"]).SearchOutcome(
                [result(path=r"D:\p\resume.pdf")], 1.0
            )
        )
        QTest.keyClick(window.input, Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier)
        assert window.mode is Mode.CHAT
        assert window._chat.pinned == [r"D:\p\resume.pdf"]
        assert "resume.pdf" in window.status.text()
        wait_for(qapp, lambda: "begin" in kinds(assistant))
        window.input.setText("summarise")
        window.submit()
        wait_for(qapp, lambda: "chat" in kinds(assistant))
        payload = next(p for k, p in assistant.calls if k == "chat")
        assert payload[2] == [r"D:\p\resume.pdf"]  # type: ignore[index]

    def test_ctrl_t_needs_a_result_and_search_mode(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.chat_with_selected()
        assert window.mode is Mode.SEARCH

    def test_long_paste_becomes_a_scratch_document(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.set_mode(Mode.CHAT)
        clipboard = QGuiApplication.clipboard()
        clipboard.setText("Senior engineer wanted.\nMust know Python.\n" * 3)
        QTest.keyClick(window.input, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
        assert window._chat.scratch.startswith("Senior engineer")
        assert "pasted document attached" in window.status.text()
        window.input.setText("which resume fits?")
        window.submit()
        wait_for(qapp, lambda: "chat" in kinds(assistant))
        payload = next(p for k, p in assistant.calls if k == "chat")
        assert payload[3].startswith("Senior engineer")  # type: ignore[index]

    def test_short_paste_is_left_to_the_line_edit(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.set_mode(Mode.CHAT)
        QGuiApplication.clipboard().setText("short")
        assert window.paste_scratch() is False
        window.set_mode(Mode.SEARCH)
        QGuiApplication.clipboard().setText("long\n" * 100)
        assert window.paste_scratch() is False  # only Chat mode captures pasted documents

    def test_esc_closes_the_window_and_unloads_the_model(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.show()
        window.set_mode(Mode.CHAT)
        window.dismiss()
        assert not window.isVisible()
        assert window.mode is Mode.SEARCH
        wait_for(qapp, lambda: ("end", "left chat mode") in assistant.calls)
        assert ("end", "left chat mode") in assistant.calls

    def test_esc_in_search_mode_also_releases_a_lingering_session(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.dismiss()
        wait_for(qapp, lambda: ("end", "closed") in assistant.calls)
        assert ("end", "closed") in assistant.calls

    def test_idle_maintenance_reports_unloads(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        assistant.maintain_result = "idle"
        window.maintain_model()
        wait_for(qapp, lambda: "unloaded" in window.status.text())
        assert "chat model unloaded (idle)" in window.status.text()
        assistant.maintain_result = None
        window.status.setText("unchanged")
        window.maintain_model()
        wait_for(qapp, lambda: kinds(assistant).count("maintain") == 2)
        qapp.processEvents()
        assert window.status.text() == "unchanged"

    def test_maintenance_without_an_assistant_is_a_noop(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        window = SearchWindow(FakeService(), Launcher(), tmp_path)  # type: ignore[arg-type]
        window.maintain_model()
        window.dismiss()

    def test_background_call_errors_are_reported(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts

        def boom() -> None:
            raise RuntimeError("gpu on fire")

        assistant.begin_chat = boom  # type: ignore[method-assign]
        window.set_mode(Mode.CHAT)
        wait_for(qapp, lambda: "gpu on fire" in window.status.text())
        assert "gpu on fire" in window.status.text()


class TestStreamCancellation:
    def test_cancelling_closes_the_generator_so_the_http_stream_closes(
        self, qapp: QApplication
    ) -> None:
        closed: list[int] = []

        def endless() -> Iterator[Event]:
            try:
                while True:
                    yield Delta("x")
            finally:
                closed.append(1)

        signals = _Signals()
        job = _StreamJob(1, endless(), signals)
        signals.streamed.connect(lambda _generation, _event: job.cancel())
        job.run()
        assert closed == [1]
        assert job.finished.is_set()

    def test_ending_a_chat_waits_for_the_answer_to_stop_before_unloading(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        release = threading.Event()
        closed = threading.Event()

        def slow_chat(message: str, state: ChatState) -> Iterator[Event]:
            try:
                yield Delta("first ")
                release.wait(5)
                yield Delta("second")
            finally:
                closed.set()

        assistant.chat = slow_chat  # type: ignore[method-assign]
        window.set_mode(Mode.CHAT)
        wait_for(qapp, lambda: "begin" in kinds(assistant))
        window.input.setText("hello")
        window.submit()
        wait_for(qapp, lambda: "first" in window.answer.toPlainText())
        window.set_mode(Mode.SEARCH)
        wait_for(qapp, lambda: False, timeout=0.3)
        assert "end" not in kinds(assistant)  # the model must not be unloaded under a live stream
        release.set()
        wait_for(qapp, lambda: "end" in kinds(assistant))
        assert closed.is_set()
        assert "second" not in window.answer.toPlainText()
