import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant

from vector_embed.app.assistant import ChatState, CloudPreview, Delta, Event, Failed, Finished
from vector_embed.app.controller import Launcher
from vector_embed.app.window import Mode, SearchWindow

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

    def cloud_available(self) -> bool:
        return self.available

    def cloud_preview_ask(self, question: str) -> CloudPreview | None:
        self.calls.append(("preview_ask", question))
        if self.preview_error:
            raise self.preview_error
        return self.preview

    def cloud_preview_chat(self, message: str, state: ChatState) -> CloudPreview | None:
        self.calls.append(("preview_chat", message))
        return self.preview

    def escalated(self, events: Iterator[Event]) -> Iterator[Event]:
        self.escalations += 1
        yield from events


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

    def confirm(preview: CloudPreview) -> bool:
        shown.append(preview)
        return True

    window.cloud_confirm = confirm
    return window, assistant, shown


def ask(window: SearchWindow, text: str = "how do retries work") -> None:
    window.set_mode(Mode.ASK)
    window.input.setText(text)
    window.submit()


def test_the_button_only_appears_in_ask_and_chat_when_a_cloud_is_configured(
    parts: tuple[SearchWindow, CloudAssistant, list[CloudPreview]],
) -> None:
    window, assistant, _ = parts
    window.show()
    assert not window.cloud_button.isVisible()  # search mode
    window.set_mode(Mode.ASK)
    assert window.cloud_button.isVisible()
    window.set_mode(Mode.CHAT)
    assert window.cloud_button.isVisible()
    window.set_mode(Mode.SEARCH)
    assert not window.cloud_button.isVisible()
    assistant.available = False
    window.set_mode(Mode.ASK)
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
    window.cloud_confirm = lambda _preview: False
    ask(window)
    wait_for(qapp, lambda: window.status.text() == "done")
    window.answer_better()
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
    assert "no cloud provider is configured" in window.status.text()
    assistant.preview_error = RuntimeError("secret.md is private and cannot be sent")
    window.answer_better()
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
