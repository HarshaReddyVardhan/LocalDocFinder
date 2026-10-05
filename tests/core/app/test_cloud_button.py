import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant

from localdoc_finder.app.assistant import ChatState, CloudPreview, Delta, Event, Failed, Finished
from localdoc_finder.app.cloud_dialog import CloudAnswer
from localdoc_finder.app.cloud_switcher import CloudChoice
from localdoc_finder.app.controller import Launcher
from localdoc_finder.app.window import Mode, SearchWindow

PREVIEW = CloudPreview(
    destination="OpenRouter / vendor/chat",
    badge="☁ Sending 2 excerpts (≈1.2k tokens) to OpenRouter / vendor/chat",
    shield="🛡 1 sensitive item will be masked: Passport (resume.pdf)",
    text="--- user ---\nPassport No: [PASSPORT REMOVED]",
)


class CloudAssistant(FakeAssistant):
    def __init__(self) -> None:
        super().__init__()
        self.available = True
        self.preview: CloudPreview | None = PREVIEW
        self.preview_error: Exception | None = None
        self.escalations = 0
        self.consent_needed = False  # the user's routing sends a normal request to the cloud
        self.remembered: list[bool] = []

    def cloud_available(self) -> bool:
        return self.available

    def cloud_destination(self) -> str | None:
        return "OpenRouter / vendor/chat" if self.available else None

    def needs_cloud_consent(self) -> bool:
        self.calls.append(("needs_consent", None))
        return self.consent_needed

    def cloud_preview_ask(self, question: str, *, escalate: bool = True) -> CloudPreview | None:
        self.calls.append(
            ("preview_ask", question) if escalate else ("preview_ask_routed", question)
        )
        if self.preview_error:
            raise self.preview_error
        return self.preview

    def cloud_preview_chat(
        self, message: str, state: ChatState, *, escalate: bool = True
    ) -> CloudPreview | None:
        self.calls.append(
            ("preview_chat", message) if escalate else ("preview_chat_routed", message)
        )
        return self.preview

    def escalated(self, events: Iterator[Event], remember: bool = False) -> Iterator[Event]:
        self.escalations += 1
        self.remembered.append(remember)
        yield from events

    def ask_escalated(self, question: str, remember: bool = False) -> Iterator[Event]:
        yield from self.escalated(self.ask(question), remember)

    def chat_escalated(
        self, message: str, state: ChatState, remember: bool = False
    ) -> Iterator[Event]:
        yield from self.escalated(self.chat(message, state), remember)

    def ask_routed(self, question: str, remember: bool = False) -> Iterator[Event]:
        self.calls.append(("ask_routed", question))
        self.remembered.append(remember)
        yield from self.events

    def chat_routed(
        self, message: str, state: ChatState, remember: bool = False
    ) -> Iterator[Event]:
        self.calls.append(("chat_routed", message))
        self.remembered.append(remember)
        yield from self.events


def wait_for(qapp: QApplication, condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def parts(
    qapp: QApplication, tmp_path: Path
) -> tuple[SearchWindow, CloudAssistant, list[CloudPreview]]:
    assistant = CloudAssistant()
    assistant.events = [Delta("Cloud answer."), Finished([], "")]
    shown: list[CloudPreview] = []
    window = SearchWindow(
        FakeService(),  # type: ignore[arg-type]
        Launcher(),
        tmp_path,
        assistant,  # type: ignore[arg-type]
    )

    def confirm(preview: CloudPreview) -> CloudAnswer:
        shown.append(preview)
        return CloudAnswer(True)

    window.cloud_confirm = confirm
    return window, assistant, shown


def ask(window: SearchWindow, text: str = "how do retries work") -> None:
    window.set_mode(Mode.ASK)
    window.input.setText(text)
    window.submit()


def test_the_button_only_appears_in_ask_and_chat_when_a_cloud_is_configured(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, _ = parts
    window.show()
    assert not window.cloud_button.isVisible()  # search mode
    window.set_mode(Mode.ASK)
    wait_for(qapp, window.cloud_button.isVisible)  # found out in the background
    assert window.cloud_button.isVisible()
    window.set_mode(Mode.CHAT)
    wait_for(qapp, window.cloud_button.isVisible)
    assert window.cloud_button.isVisible()
    window.set_mode(Mode.SEARCH)
    assert not window.cloud_button.isVisible()
    assistant.available = False
    window.set_mode(Mode.ASK)
    wait_for(qapp, lambda: False, timeout=0.3)
    assert not window.cloud_button.isVisible()


def test_a_failing_availability_check_just_hides_the_button(
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]], monkeypatch: pytest.MonkeyPatch
) -> None:
    window, assistant, _ = parts

    def boom() -> bool:
        raise RuntimeError("no index yet")

    monkeypatch.setattr(assistant, "cloud_available", boom)
    window.set_mode(Mode.ASK)
    assert not window.cloud_button.isVisible()


def test_answer_better_shows_the_preview_and_asks_the_cloud_after_consent(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, shown = parts
    ask(window)
    wait_for(qapp, lambda: window.status.text() == "done")
    window.answer_better()
    wait_for(qapp, lambda: shown)
    assert shown == [PREVIEW]
    wait_for(qapp, lambda: assistant.escalations == 1)
    assert assistant.escalations == 1
    assert ("preview_ask", "how do retries work") in assistant.calls
    assert (
        "asking OpenRouter / vendor/chat" in window.status.text() or window.status.text() == "done"
    )


def test_declining_sends_nothing(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, _ = parts
    window.cloud_confirm = lambda _preview: CloudAnswer(False)
    ask(window)
    wait_for(qapp, lambda: window.status.text() == "done")
    window.answer_better()
    wait_for(qapp, lambda: window.status.text() == "cancelled: nothing was sent")
    assert window.status.text() == "cancelled: nothing was sent"
    assert assistant.escalations == 0


def test_chat_mode_previews_the_chat_request(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, shown = parts
    window.set_mode(Mode.CHAT)
    window.input.setText("what is missing?")
    window.submit()
    wait_for(qapp, lambda: window.status.text().startswith("done"))
    window.answer_better()
    wait_for(qapp, lambda: shown)
    assert ("preview_chat", "what is missing?") in assistant.calls
    assert shown == [PREVIEW]
    wait_for(qapp, lambda: assistant.escalations == 1)


def test_missing_cloud_and_preview_errors_are_reported(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, _ = parts
    ask(window)
    wait_for(qapp, lambda: window.status.text() == "done")
    assistant.preview = None
    window.answer_better()
    wait_for(qapp, lambda: "no cloud provider is configured" in window.status.text())
    assert "no cloud provider is configured" in window.status.text()
    assistant.preview_error = RuntimeError("secret.md is private and cannot be sent")
    window.answer_better()
    wait_for(qapp, lambda: "private and cannot be sent" in window.status.text())
    assert "private and cannot be sent" in window.status.text()


def test_answer_better_needs_a_previous_question_and_the_right_mode(
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]],
) -> None:
    window, assistant, _ = parts
    window.answer_better()  # search mode, nothing asked
    window.set_mode(Mode.ASK)
    window.answer_better()  # nothing asked yet
    assert assistant.escalations == 0
    assert not any(kind.startswith("preview") for kind, _ in assistant.calls)


def test_failures_during_an_escalated_answer_are_shown(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, _ = parts
    ask(window)
    wait_for(qapp, lambda: window.status.text() == "done")
    assistant.events = [Failed("the monthly cloud budget ($5.00) is used up")]
    window.answer_better()
    wait_for(qapp, lambda: "budget" in window.status.text())
    assert "budget" in window.answer.toPlainText()


def test_the_preview_is_built_off_the_ui_thread(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    import threading

    window, assistant, _ = parts
    threads: list[str] = []
    original = assistant.cloud_preview_ask

    def spy(question: str, **kwargs: bool) -> CloudPreview | None:
        threads.append(threading.current_thread().name)
        return original(question, **kwargs)

    assistant.cloud_preview_ask = spy  # type: ignore[method-assign]
    ask(window)
    wait_for(qapp, lambda: window.status.text() == "done")
    window.answer_better()
    wait_for(qapp, lambda: threads)
    assert threads
    assert threads[0] != threading.main_thread().name


# ------------------------------------------------------------------ routed requests
def wait_ready(qapp: QApplication, window: SearchWindow) -> None:
    """Until the background probe has found the configured provider."""
    wait_for(qapp, lambda: window._cloud_ready)
    assert window._cloud_ready


def test_a_request_routed_to_the_cloud_is_previewed_and_sent_after_consent(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, shown = parts
    assistant.consent_needed = True
    window.set_mode(Mode.ASK)
    wait_ready(qapp, window)
    window.input.setText("what changed")
    window.submit()
    wait_for(qapp, lambda: ("ask_routed", "what changed") in assistant.calls)
    assert shown == [PREVIEW]
    assert ("preview_ask_routed", "what changed") in assistant.calls  # not an escalated preview
    assert assistant.escalations == 0  # the route stays a routed one
    assert not any(kind == "ask" for kind, _ in assistant.calls)  # not the plain local path
    assert assistant.remembered == [False]


def test_a_routed_chat_turn_is_previewed_too(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, shown = parts
    assistant.consent_needed = True
    window.set_mode(Mode.CHAT)
    wait_ready(qapp, window)
    window.input.setText("hello")
    window.submit()
    wait_for(qapp, lambda: any(c[0] == "chat_routed" for c in assistant.calls))
    assert ("preview_chat_routed", "hello") in assistant.calls
    assert shown == [PREVIEW]


def test_declining_a_routed_request_sends_nothing(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, _ = parts
    assistant.consent_needed = True
    window.cloud_confirm = lambda _preview: CloudAnswer(False)
    window.set_mode(Mode.ASK)
    wait_ready(qapp, window)
    window.input.setText("what changed")
    window.submit()
    wait_for(qapp, lambda: window.status.text() == "cancelled: nothing was sent")
    assert window.status.text() == "cancelled: nothing was sent"
    assert not any(kind in ("ask", "ask_routed") for kind, _ in assistant.calls)
    assert assistant.escalations == 0


def test_dont_ask_again_is_passed_on_with_the_request(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, _ = parts
    assistant.consent_needed = True
    window.cloud_confirm = lambda _preview: CloudAnswer(True, remember=True)
    window.set_mode(Mode.ASK)
    wait_ready(qapp, window)
    window.input.setText("what changed")
    window.submit()
    wait_for(qapp, lambda: assistant.remembered)
    assert assistant.remembered == [True]


def test_answer_better_passes_dont_ask_again_too(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, _ = parts
    window.cloud_confirm = lambda _preview: CloudAnswer(True, remember=True)
    ask(window)
    wait_for(qapp, lambda: window.status.text() == "done")
    window.answer_better()
    wait_for(qapp, lambda: assistant.escalations == 1)
    assert assistant.remembered == [True]


def test_no_dialog_when_the_request_needs_no_consent(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, shown = parts
    window.set_mode(Mode.ASK)
    wait_ready(qapp, window)
    window.input.setText("what changed")
    window.submit()
    wait_for(qapp, lambda: window.status.text() == "done")
    assert shown == []
    assert ("needs_consent", None) in assistant.calls
    assert ("ask", "what changed") in assistant.calls


def test_a_newer_question_cancels_the_pending_routed_preview(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, assistant, shown = parts
    assistant.consent_needed = True
    window.set_mode(Mode.ASK)
    wait_ready(qapp, window)
    window._generation += 1  # the user asked something else meanwhile
    window._on_consent_checked(window._generation - 1, True)
    wait_for(qapp, lambda: False, timeout=0.2)
    assert shown == []


# ------------------------------------------------------------------ the model switcher
class FakeSwitcher:
    def __init__(self, choices: list[CloudChoice]) -> None:
        self._choices = choices
        self.chosen: list[CloudChoice] = []
        self.error: Exception | None = None

    def choices(self) -> list[CloudChoice]:
        if self.error:
            raise self.error
        return self._choices

    def choose(self, choice: CloudChoice) -> None:
        if self.error:
            raise self.error
        self.chosen.append(choice)


A = CloudChoice("openrouter", "OpenRouter", "vendor/chat", current=True)
B = CloudChoice("gemini", "Google Gemini", "gemini-2.5-flash")


@pytest.fixture
def switcher(
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]],
) -> FakeSwitcher:
    fake = FakeSwitcher([A, B])
    parts[0].cloud_switcher = fake  # type: ignore[assignment]
    return fake


def test_the_model_button_shows_with_the_cloud_button_and_names_the_destination(
    qapp: QApplication,
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]],
    switcher: FakeSwitcher,
) -> None:
    window, _, _ = parts
    window.show()
    assert not window.model_button.isVisible()  # search mode
    window.set_mode(Mode.ASK)
    wait_for(qapp, window.model_button.isVisible)
    assert window.model_button.isVisible()
    assert "OpenRouter / vendor/chat" in window.cloud_button.toolTip()
    window.set_mode(Mode.SEARCH)
    assert not window.model_button.isVisible()


def test_without_a_switcher_there_is_no_model_button(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    window, _, _ = parts
    window.show()
    window.set_mode(Mode.ASK)
    wait_for(qapp, window.cloud_button.isVisible)
    assert window.cloud_button.isVisible()
    assert not window.model_button.isVisible()
    window.show_model_menu()  # harmless


def test_picking_a_model_switches_to_it(
    qapp: QApplication,
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]],
    switcher: FakeSwitcher,
) -> None:
    window, _, _ = parts
    offered: list[list[CloudChoice]] = []

    def pick(choices: list[CloudChoice]) -> CloudChoice:
        offered.append(choices)
        return B

    window.choose_model = pick
    window.show_model_menu()
    assert offered == [[A, B]]
    assert switcher.chosen == [B]
    assert window.status.text() == "cloud model: Google Gemini / gemini-2.5-flash"


def test_closing_the_menu_changes_nothing(
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]], switcher: FakeSwitcher
) -> None:
    window, _, _ = parts
    window.choose_model = lambda _choices: None
    window.show_model_menu()
    assert switcher.chosen == []


def test_manage_opens_settings_on_the_cloud_tab(
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]], switcher: FakeSwitcher
) -> None:
    from localdoc_finder.app.window import MANAGE_CLOUD

    window, _, _ = parts
    requested: list[int] = []
    window.cloud_settings_requested.connect(lambda: requested.append(1))
    window.choose_model = lambda _choices: MANAGE_CLOUD
    window.show_model_menu()
    assert requested == [1]
    assert switcher.chosen == []


def test_settings_errors_are_shown_not_raised(
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]], switcher: FakeSwitcher
) -> None:
    from localdoc_finder.core.settings import SettingsError

    window, _, _ = parts
    switcher.error = SettingsError("settings.toml is broken")
    window.show_model_menu()
    assert "settings.toml is broken" in window.status.text()
    switcher.error = None
    window.choose_model = lambda _choices: B
    switcher.error = SettingsError("no provider named 'gemini'")
    window.choose_model = lambda _choices: B
    switcher._choices = [A, B]
    window.show_model_menu()
    assert "⚠" in window.status.text()


def test_the_real_menu_lists_the_choices_checked_and_manage(
    qapp: QApplication, parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]]
) -> None:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMenu

    window, _, _ = parts
    seen: list[tuple[str, bool]] = []

    def press(*keys: Qt.Key) -> None:
        menu = QApplication.activePopupWidget()
        assert isinstance(menu, QMenu)
        seen.extend((a.text(), a.isChecked()) for a in menu.actions() if not a.isSeparator())
        for key in keys:
            QTest.keyClick(menu, key)

    QTimer.singleShot(200, lambda: press(Qt.Key.Key_Down, Qt.Key.Key_Down, Qt.Key.Key_Return))
    assert window._model_menu([A, B]) == B
    assert seen == [
        ("OpenRouter / vendor/chat", True),
        ("Google Gemini / gemini-2.5-flash", False),
        ("Manage…", False),
    ]
    QTimer.singleShot(200, lambda: press(Qt.Key.Key_Escape))
    assert window._model_menu([A, B]) is None
    QTimer.singleShot(
        200, lambda: press(Qt.Key.Key_Down, Qt.Key.Key_Down, Qt.Key.Key_Down, Qt.Key.Key_Return)
    )
    assert window._model_menu([A, B]) == "manage-cloud"
