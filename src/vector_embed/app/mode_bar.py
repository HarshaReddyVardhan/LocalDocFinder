"""The row of mode pills above the search bar: shows where you are, and a click switches."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget


class ModeBar(QWidget):
    """One checkable pill per mode; exactly one is lit. Pills never take the keyboard focus."""

    chosen = Signal(str)  # the clicked mode's value

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._pills: dict[str, QPushButton] = {}
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

    def set_modes(self, modes: list[tuple[str, str]]) -> None:
        """``(value, title)`` pairs, in Tab order; the same list again keeps the pills."""
        if [(value, pill.text()) for value, pill in self._pills.items()] == modes:
            return
        for pill in self._pills.values():
            self._group.removeButton(pill)
            pill.hide()  # deleteLater alone leaves it drawn until the event loop runs
            pill.setParent(None)
            pill.deleteLater()
        self._pills = {}
        for value, title in modes:
            pill = QPushButton(title)
            pill.setObjectName("modePill")
            pill.setCheckable(True)
            pill.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # typing stays in the search bar
            pill.setCursor(Qt.CursorShape.PointingHandCursor)
            pill.clicked.connect(lambda _checked=False, v=value: self.chosen.emit(v))
            self._group.addButton(pill)
            self._layout.addWidget(pill)
            self._pills[value] = pill

    def set_current(self, value: str) -> None:
        if (pill := self._pills.get(value)) is not None:
            pill.setChecked(True)

    def current_title(self) -> str:
        checked = self._group.checkedButton()
        return checked.text() if checked is not None else ""
