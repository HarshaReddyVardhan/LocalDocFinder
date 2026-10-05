"""The strip above the result list: filter by file type, sort by relevance, date or name.

It exists only while there are results, so the search bar stays bare until something is found.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from localdoc_finder.core.result_view import SortOrder, ext_counts
from localdoc_finder.core.skills.search import SearchResult

ALL_TYPES = "All types"
NO_EXTENSION = "(no extension)"


class RefineBar(QWidget):
    """Two drop-downs; ``changed`` fires when the user picks a type or a sort order."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.type_box = self._combo("Show only one file type")
        self.sort_box = self._combo("Sort the results")
        for order in SortOrder:
            self.sort_box.addItem(order.value, order)
        label = QLabel("Sort")
        label.setObjectName("refineLabel")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.type_box)
        row.addStretch(1)
        row.addWidget(label)
        row.addWidget(self.sort_box)

    def _combo(self, tip: str) -> QComboBox:
        box = QComboBox()
        box.setToolTip(tip)
        box.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # typing stays in the search bar
        box.currentIndexChanged.connect(lambda _index: self.changed.emit())
        return box

    @property
    def ext(self) -> str | None:
        """The chosen extension (lower case, with its dot; '' for none), or ``None`` for all."""
        value = self.type_box.currentData()
        return value if isinstance(value, str) else None

    @property
    def order(self) -> SortOrder:
        value = self.sort_box.currentData()
        return value if isinstance(value, SortOrder) else SortOrder.RELEVANCE

    def set_results(self, results: list[SearchResult]) -> None:
        """List the types in ``results``; the chosen one stays if it is still there.

        Silent (no ``changed``): the caller is already redrawing for the new results.
        """
        keep = self.ext
        self.type_box.blockSignals(True)
        self.type_box.clear()
        self.type_box.addItem(ALL_TYPES, None)
        for ext, count in ext_counts(results):
            self.type_box.addItem(f"{ext or NO_EXTENSION}  ({count})", ext)
        index = self.type_box.findData(keep) if keep is not None else 0
        self.type_box.setCurrentIndex(max(index, 0))
        self.type_box.blockSignals(False)
        # One type alone has nothing to filter: hide the box rather than offer a dead choice.
        self.type_box.setVisible(self.type_box.count() > 2)
