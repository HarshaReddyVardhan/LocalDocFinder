import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton
from tests.core.app.test_app import FakeService, result

from localdoc_finder.app.assistant import ChatState, Delta, Event, Failed, Finished
from localdoc_finder.app.controller import Launcher
from localdoc_finder.app.theme import Scheme
from localdoc_finder.app.window import EXPANDED_HEIGHT, Mode, SearchWindow, _Signals, _StreamJob
from localdoc_finder.core.rag import Source


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

    session_active = False

    def reset(self) -> None:
        self.calls.append(("reset", None))

    def revoke_consent(self) -> None:
        self.calls.append(("revoke_consent", None))

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

    def test_only_the_features_switched_on_get_a_mode(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        enabled: set[str] = {"chat"}
        window = SearchWindow(
            FakeService(),  # type: ignore[arg-type]
            Launcher(),
            tmp_path,
            FakeAssistant(),  # type: ignore[arg-type]
            features=lambda: enabled,
        )
        window.show()
        assert window.available_modes() == [Mode.SEARCH, Mode.CHAT]
        window.input.setText("?a question")  # Ask is off: the text stays a search
        assert window.mode is Mode.SEARCH
        enabled.clear()
        window.reload_context()  # Settings switched the last feature off
        assert window.available_modes() == [Mode.SEARCH]
        assert window.mode_bar.isHidden()  # nothing to switch between
        assert "Ctrl+T" not in window.hints.text()

    def test_switching_a_feature_off_leaves_its_mode(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        enabled: set[str] = {"ask"}
        window = SearchWindow(
            FakeService(),  # type: ignore[arg-type]
            Launcher(),
            tmp_path,
            FakeAssistant(),  # type: ignore[arg-type]
            features=lambda: enabled,
        )
        window.show()
        window.set_mode(Mode.ASK)
        enabled.clear()
        window.reload_context()
        assert window.mode is Mode.SEARCH

    def test_question_mark_prefix_switches_to_ask(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.input.setText("?how do retries work")
        assert window.mode is Mode.ASK
        assert window.input.text() == "how do retries work"
        assert window.mode_bar.current_title() == "Ask"

    def test_mode_labels_and_placeholders_follow_the_mode(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.set_mode(Mode.CHAT)
        assert window.mode_bar.current_title() == "Chat"
        assert "pinned" in window.input.placeholderText()
        assert "send" in window.hints.text()
        window.set_mode(Mode.CHAT)  # same mode: nothing happens

    def test_the_mode_pills_show_every_mode_and_a_click_switches(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        pills = window.mode_bar.findChildren(QPushButton, "modePill")
        assert [pill.text() for pill in pills] == ["Search", "Ask", "Chat", "Match"][: len(pills)]
        assert window.mode_bar.current_title() == "Search"  # visible in Search mode too
        next(pill for pill in pills if pill.text() == "Ask").click()
        assert window.mode is Mode.ASK
        assert window.mode_bar.current_title() == "Ask"

    def test_a_pill_for_an_unknown_mode_changes_nothing(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window._choose_mode("nope")
        assert window.mode is Mode.SEARCH
        assert window.mode_bar.current_title() == "Search"


class TestWindowChrome:
    def test_only_search_hides_when_the_focus_moves_away(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.show()
        window.isActiveWindow = lambda: False  # type: ignore[method-assign]  # a browser took it
        window.set_mode(Mode.ASK)
        window._hide_if_inactive()
        assert window.isVisible()  # reading an answer beside another app
        window.set_mode(Mode.SEARCH)
        window._hide_if_inactive()
        assert not window.isVisible()

    def test_summon_centres_the_card_until_the_user_moves_it(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.summon()
        screen = QGuiApplication.screenAt(window.geometry().center())
        assert screen is not None
        assert abs(window.geometry().center().x() - screen.availableGeometry().center().x()) <= 1
        window.dismiss()
        window._placed = True  # dragged somewhere
        window.move(window.x() + 40, window.y() + 30)
        moved = window.pos()
        window.summon()
        assert window.pos() == moved

    def test_dragging_the_card_moves_the_window(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.show()
        start = window.pos()
        grab = QPoint(window.width() // 2, 30)  # the header's empty middle
        QTest.mousePress(window, Qt.MouseButton.LeftButton, pos=grab)
        QTest.mouseMove(window, grab + QPoint(60, 40))
        QTest.mouseRelease(window, Qt.MouseButton.LeftButton, pos=grab + QPoint(60, 40))
        assert window.pos() == start + QPoint(60, 40)
        assert window._placed  # the next summon leaves it there

    def test_a_height_set_with_the_grip_is_kept_for_expanded_modes(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.show()
        window.set_mode(Mode.ASK)
        window.resize(window.width(), EXPANDED_HEIGHT + 120)  # the user drags the grip
        qapp.processEvents()
        window.set_mode(Mode.SEARCH)
        assert window.height() == window.compact_height()
        window.set_mode(Mode.CHAT)
        assert window.height() == EXPANDED_HEIGHT + 120

    def test_the_card_paints_in_both_schemes(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        for scheme in Scheme:
            window.apply_scheme(scheme)
            image = window.grab().toImage()
            centre = image.pixelColor(image.width() // 2, image.height() // 2)
            assert centre.alpha() == 255  # the card is opaque; only the shadow margin is not
            assert (centre.lightness() > 128) is (scheme is Scheme.LIGHT)
        corner = window.grab().toImage().pixelColor(0, 0)
        assert corner.alpha() < 40  # outside the rounded card: (nearly) transparent shadow


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
        from localdoc_finder.app.controller import SearchOutcome

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
            __import__("localdoc_finder.app.controller", fromlist=["SearchOutcome"]).SearchOutcome(
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
        window.show()
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

    def test_the_timer_stops_when_there_is_no_window_and_no_loaded_model(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.show()
        window._maintain.start()
        window.hide()
        window.maintain_model()
        assert not window._maintain.isActive()  # nothing left to watch
        assert "maintain" not in kinds(assistant)  # and no pointless check was run

    def test_the_timer_keeps_running_while_a_chat_model_is_loaded(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        assistant.session_active = True  # a model is still loaded: idle unload must stay possible
        window.show()
        window._maintain.start()
        window.hide()
        window.maintain_model()
        wait_for(qapp, lambda: "maintain" in kinds(assistant))
        assert window._maintain.isActive()

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


class TestReloadContext:
    def test_a_settings_change_resets_the_services_and_ends_a_chat(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.set_mode(Mode.CHAT)
        wait_for(qapp, lambda: "begin" in kinds(assistant))
        window.reload_context()
        assert window.mode is Mode.SEARCH  # the chat's model or route may have changed
        wait_for(qapp, lambda: "end" in kinds(assistant))
        assert "reset" in kinds(assistant)
        assert window._service.resets == 1  # type: ignore[attr-defined]

    def test_in_search_mode_it_only_resets(
        self, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window.reload_context()
        assert kinds(assistant) == ["reset"]


class TestStreamingRender:
    def test_a_burst_of_tokens_is_rendered_far_fewer_times_than_it_arrives(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.set_mode(Mode.ASK)
        renders: list[int] = []
        original = window.answer.setMarkdown
        window.answer.setMarkdown = lambda text: renders.append(len(text)) or original(text)  # type: ignore[method-assign]
        for i in range(200):
            window._on_event(window._generation, Delta(f"token{i} "))
        assert len(renders) < 20  # not one markdown parse per token
        wait_for(qapp, lambda: "token199" in window.answer.toPlainText())
        assert "token0 " in window.answer.toPlainText()
        assert "token199" in window.answer.toPlainText()  # the trailing flush shows everything

    def test_finishing_flushes_immediately(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.set_mode(Mode.ASK)
        window._on_event(window._generation, Delta("first "))
        window._on_event(window._generation, Delta("second"))  # held back by the throttle
        window._on_event(window._generation, Finished([], ""))
        assert "second" in window.answer.toPlainText()  # no wait for the timer


class TestDialogsAndQuitting:
    def test_the_popup_stays_while_a_dialog_is_open_and_hides_afterwards(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts
        window.show()
        window.isActiveWindow = lambda: False  # type: ignore[method-assign]  # focus went to the dialog

        def while_open() -> bool:
            window._hide_if_inactive()  # the focus-lost timer fires while the dialog is up
            return window.isVisible()

        assert window._in_dialog(while_open) is True
        window._hide_if_inactive()  # once it closed and focus is elsewhere, it hides as before
        assert not window.isVisible()

    def test_the_dialog_guard_counts_nested_dialogs_and_survives_errors(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, _, _ = parts

        def boom() -> None:
            raise RuntimeError("dialog crashed")

        with pytest.raises(RuntimeError):
            window._in_dialog(boom)
        assert window._dialogs == 0  # never stuck "open"
        window._in_dialog(lambda: window._in_dialog(lambda: None))
        assert window._dialogs == 0

    def test_shutdown_ends_the_chat_and_releases_the_embedder(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        window._maintain.start()
        window.shutdown()
        assert ("end", "quitting") in assistant.calls  # synchronous: nothing left to a pool job
        assert window._service.released == 1  # type: ignore[attr-defined]
        assert not window._maintain.isActive()

    def test_shutdown_waits_for_a_running_answer_to_stop(
        self, qapp: QApplication, parts: tuple[SearchWindow, FakeAssistant, list]
    ) -> None:
        window, assistant, _ = parts
        release = threading.Event()
        closed = threading.Event()

        def slow_ask(question: str) -> Iterator[Event]:
            try:
                yield Delta("started")
                release.wait(5)
                yield Delta("more")
            finally:
                closed.set()

        assistant.ask = slow_ask  # type: ignore[method-assign]
        window.set_mode(Mode.ASK)
        window.input.setText("q")
        window.submit()
        wait_for(qapp, lambda: "started" in window.answer.toPlainText())
        threading.Timer(0.2, release.set).start()
        window.shutdown()
        assert closed.is_set()  # the answer was stopped before the models were unloaded
        assert ("end", "quitting") in assistant.calls
