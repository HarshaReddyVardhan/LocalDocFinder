"""When each feature uses the cloud, and a button to withdraw "don't ask again"."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QPushButton, QWidget

from localdoc_finder.core.models.catalog import ROLE_CHAT, ROLE_MATCH_SCORER

POLICY_CHOICES = (
    ("local", "Only when I press Answer better"),
    ("auto", "When the local model can't"),
    ("cloud", "Always use the cloud"),
)


class RoutingBox(QWidget):
    """One policy per feature. Emits ``routing_changed(role, policy)`` when the user picks one."""

    routing_changed = Signal(str, str)
    forget_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.chat = _policy_combo()
        self.match = _policy_combo()
        self.forget = QPushButton("Forget \u201cdon\u2019t ask again\u201d")
        self.forget.setToolTip("The next cloud request asks for your consent again")
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Ask && Chat uses the cloud", self.chat)
        form.addRow("Match uses the cloud", self.match)
        form.addRow(self.forget)
        self.chat.activated.connect(lambda _i: self._picked(ROLE_CHAT, self.chat))
        self.match.activated.connect(lambda _i: self._picked(ROLE_MATCH_SCORER, self.match))
        self.forget.clicked.connect(self.forget_clicked)

    def set_policies(self, chat: str, match: str) -> None:
        self.chat.setCurrentIndex(max(0, self.chat.findData(chat)))
        self.match.setCurrentIndex(max(0, self.match.findData(match)))

    def _picked(self, role: str, combo: QComboBox) -> None:
        self.routing_changed.emit(role, str(combo.currentData()))


def _policy_combo() -> QComboBox:
    combo = QComboBox()
    for policy, label in POLICY_CHOICES:
        combo.addItem(label, policy)
    return combo
