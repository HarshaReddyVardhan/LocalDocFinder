"""A model picker for a provider that may list hundreds of models (OpenRouter does).

An editable combo box with a contains-anywhere completer, a star to keep favourites at the top,
and, when the model has no published price, boxes to enter it so the budget can still be enforced.
"""

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from localdoc_finder.app.settings_controller import CloudModel

STAR_ON, STAR_OFF = "★", "☆"
MAX_PRICE_USD = 10000.0


class ModelPicker(QWidget):
    """One role's model choice. Emits ``chosen`` with the model id (empty = cleared)."""

    chosen = Signal(str)
    star_toggled = Signal(str)  # model id
    price_entered = Signal(str, float, float)  # model id, input $/1M, output $/1M

    def __init__(self, title: str, placeholder: str = "") -> None:
        super().__init__()
        self._committed = ""
        self._favorites: set[str] = set()
        self.title = QLabel(title)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo.setMinimumContentsLength(28)
        if (completer := self.combo.completer()) is not None:  # typing narrows a long list
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        line = self.combo.lineEdit()
        if line is not None:
            line.setPlaceholderText(placeholder)
        self.star = QPushButton(STAR_OFF)
        self.star.setToolTip("Keep this model at the top of the list")
        self.star.setFixedWidth(32)

        self.price_in = _price_box("in")
        self.price_out = _price_box("out")
        self.save_price = QPushButton("Save price")
        self.price_row = QWidget()
        row = QHBoxLayout(self.price_row)
        row.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("No price listed. $ per 1M tokens:")
        row.addWidget(hint)
        row.addWidget(self.price_in)
        row.addWidget(self.price_out)
        row.addWidget(self.save_price)
        row.addStretch(1)
        self.price_row.setVisible(False)

        pick = QHBoxLayout()
        pick.addWidget(self.combo, 1)
        pick.addWidget(self.star)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title)
        layout.addLayout(pick)
        layout.addWidget(self.price_row)

        self.combo.activated.connect(lambda _index: self._commit())
        self.combo.editTextChanged.connect(lambda _text: self._show_star())
        if line is not None:
            line.editingFinished.connect(self._commit)
        self.star.clicked.connect(self._toggle_star)
        self.save_price.clicked.connect(self._save_price)

    # ------------------------------------------------------------------ data in
    def set_models(self, models: Sequence[CloudModel], favorites: Sequence[str]) -> None:
        """Replace the list; favourites first, then the provider's own (name) order."""
        self._favorites = set(favorites)
        ordered = sorted(models, key=lambda m: m.id not in self._favorites)  # stable
        shown = self.current_id()  # keep what is in the box, picked or half-typed
        self.combo.blockSignals(True)
        self.combo.clear()
        for model in ordered:
            prefix = f"{STAR_ON} " if model.id in self._favorites else ""
            self.combo.addItem(prefix + model.label, model.id)
        self.combo.blockSignals(False)
        self.combo.setEditText(shown)
        self._show_star()

    def set_current(self, model: str) -> None:
        """Show ``model`` without announcing it as a change."""
        self._committed = model
        self.combo.setEditText(model)
        self._show_star()

    def set_favorites(self, favorites: Sequence[str]) -> None:
        self._favorites = set(favorites)
        self._show_star()

    def show_price_entry(self, visible: bool) -> None:
        self.price_row.setVisible(visible and bool(self.current_id()))

    # ------------------------------------------------------------------ data out
    def current_id(self) -> str:
        """The id in the box: a picked item's id, or what the user typed."""
        text = self.combo.currentText().strip()
        index = self.combo.findText(text)
        return str(self.combo.itemData(index)) if index >= 0 else text

    # ------------------------------------------------------------------ internals
    def _commit(self) -> None:
        model = self.current_id()
        self.combo.setEditText(model)  # a picked row shows its id, not the whole description
        self._show_star()
        if model != self._committed:
            self._committed = model
            self.chosen.emit(model)

    def _show_star(self) -> None:
        model = self.current_id()
        self.star.setText(STAR_ON if model in self._favorites else STAR_OFF)
        self.star.setEnabled(bool(model))

    def _toggle_star(self) -> None:
        if model := self.current_id():
            self.star_toggled.emit(model)

    def _save_price(self) -> None:
        self.price_entered.emit(self.current_id(), self.price_in.value(), self.price_out.value())


def _price_box(prefix: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setPrefix(f"{prefix} $")
    box.setDecimals(3)
    box.setRange(0.0, MAX_PRICE_USD)
    return box
