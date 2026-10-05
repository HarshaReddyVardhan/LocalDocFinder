from collections.abc import Callable

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox

from localdoc_finder.app.assistant import CloudPreview
from localdoc_finder.app.cloud_dialog import REMEMBER_TEXT, CloudAnswer, confirm_cloud_dialog

PREVIEW = CloudPreview("OpenRouter / m", "badge", "shield", "the exact text")


def press(monkeypatch: pytest.MonkeyPatch, button: str, *, tick: bool) -> list[QMessageBox]:
    """Make the dialog behave as if the user ticked the box (or not) and pressed ``button``."""
    boxes: list[QMessageBox] = []

    def fake_exec(box: QMessageBox) -> int:
        boxes.append(box)
        checkbox = box.checkBox()
        assert isinstance(checkbox, QCheckBox)
        checkbox.setChecked(tick)
        target = next(b for b in box.buttons() if b.text().replace("&", "") == button)
        target.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    return boxes


@pytest.fixture
def run(qapp: QApplication) -> Callable[[], CloudAnswer]:
    return lambda: confirm_cloud_dialog(PREVIEW)


def test_send_without_the_box_ticked_is_for_this_request_only(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, run: Callable[[], CloudAnswer]
) -> None:
    boxes = press(monkeypatch, "Send", tick=False)
    assert run() == CloudAnswer(send=True, remember=False)
    box = boxes[0]
    assert box.detailedText() == "the exact text"
    checkbox = box.checkBox()
    assert checkbox is not None
    assert checkbox.text() == REMEMBER_TEXT


def test_send_with_the_box_ticked_remembers(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, run: Callable[[], CloudAnswer]
) -> None:
    press(monkeypatch, "Send", tick=True)
    assert run() == CloudAnswer(send=True, remember=True)


def test_cancelling_remembers_nothing_even_with_the_box_ticked(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, run: Callable[[], CloudAnswer]
) -> None:
    press(monkeypatch, "Cancel", tick=True)
    assert run() == CloudAnswer(send=False, remember=False)
