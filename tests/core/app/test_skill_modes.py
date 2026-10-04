"""Extensibility: a skill added as one registered class gets a window mode with no UI code."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import Field
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant, wait_for

from vector_embed.app.assistant import AssistantService, Delta, Event, Failed, Finished
from vector_embed.app.controller import Launcher
from vector_embed.app.window import Mode, SearchWindow, SkillMode
from vector_embed.core.skills.base import (
    SKILLS,
    UI_PANEL,
    Skill,
    SkillContext,
    SkillInput,
    panel_skills,
)


class ShoutInput(SkillInput):
    text: str = Field(min_length=2, description="Text to shout")


class ShoutSkill(Skill):
    """A stand-in for a future feature: streams its answer."""

    name = "shout"
    title = "Shout"
    description = "Repeat the text in capitals."
    Input = ShoutInput
    ui_hint = UI_PANEL
    cli_positional = "text"

    def run(self, params: SkillInput) -> str:
        assert isinstance(params, ShoutInput)
        return params.text.upper()

    def stream(self, params: SkillInput) -> Iterator[str]:
        assert isinstance(params, ShoutInput)
        yield from params.text.upper().split(" ")


class CountSkill(ShoutSkill):
    """A panel skill that does not stream: its rendered result is the answer."""

    name = "count"
    title = "Count"

    def stream(self, params: SkillInput) -> None:
        return None

    def render(self, output: object) -> str:
        return f"{len(str(output))} characters"


@pytest.fixture
def extra_skills() -> Iterator[None]:
    for skill in (ShoutSkill, CountSkill):
        SKILLS.add(skill.name, skill)
    try:
        yield
    finally:
        for skill in (ShoutSkill, CountSkill):
            SKILLS.remove(skill.name)


class SkillAssistant(FakeAssistant):
    def run_skill(self, name: str, text: str) -> Iterator[Event]:
        self.calls.append((name, text))
        yield Delta(f"{name}: {text.upper()}")
        yield Finished()


def drain(events: Iterator[Event]) -> list[Event]:
    return list(events)


def test_registered_panel_skills_are_found(extra_skills: None) -> None:
    names = [skill.name for skill in panel_skills()]
    assert {"ask", "shout", "count"} <= set(names)
    assert "search" not in names  # a list skill has its own UI


def test_the_service_runs_streaming_and_plain_skills(
    extra_skills: None, skill_ctx: SkillContext
) -> None:
    service = AssistantService(lambda: skill_ctx)
    assert drain(service.run_skill("shout", "hi there")) == [
        Delta("HI"),
        Delta("THERE"),
        Finished(),
    ]
    assert drain(service.run_skill("count", "hello")) == [Delta("5 characters"), Finished()]
    (failed,) = drain(service.run_skill("shout", "x"))  # fails the skill's own input rules
    assert isinstance(failed, Failed) and "shout" in failed.message
    (unknown,) = drain(service.run_skill("nope", "x"))
    assert isinstance(unknown, Failed)


def test_a_new_skill_becomes_a_window_mode(
    qapp: QApplication, tmp_path: Path, extra_skills: None
) -> None:
    assistant = SkillAssistant()
    window = SearchWindow(
        FakeService(),  # type: ignore[arg-type]
        Launcher(),
        tmp_path,
        assistant,  # type: ignore[arg-type]
    )
    window.show()
    shout = next(m for m in window.available_modes() if isinstance(m, SkillMode))
    assert shout.value == "shout"
    for _ in range(3):  # Search -> Ask -> Chat -> Shout
        QTest.keyClick(window.input, Qt.Key.Key_Tab)
    assert window.mode == shout
    assert window.mode_bar.current_title() == "Shout"
    assert "capitals" in window.input.placeholderText()

    window.input.setText("make it loud")
    QTest.keyClick(window.input, Qt.Key.Key_Return)
    wait_for(qapp, lambda: "MAKE IT LOUD" in window.answer.toPlainText())
    assert ("shout", "make it loud") in assistant.calls
    assert "shout: MAKE IT LOUD" in window.answer.toPlainText()

    window.set_mode(Mode.SEARCH)
    assert not window.answer.isVisible()


def test_without_an_assistant_no_skill_modes_appear(
    qapp: QApplication, tmp_path: Path, extra_skills: None
) -> None:
    window = SearchWindow(FakeService(), Launcher(), tmp_path)  # type: ignore[arg-type]
    assert window.available_modes() == [Mode.SEARCH]
