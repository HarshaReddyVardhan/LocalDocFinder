"""The "What's new" dialog: the release notes of a downloaded update, before it is applied."""

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTextBrowser, QVBoxLayout, QWidget

NO_NOTES = "This release has no notes."


class WhatsNewDialog(QDialog):
    """Accepted = restart now; rejected = later (the tray keeps "Restart to update")."""

    def __init__(self, version: str, notes: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"What's new in {version}")
        self.setMinimumSize(460, 360)
        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        self.notes.setMarkdown(notes.strip() or NO_NOTES)
        buttons = QDialogButtonBox()
        self.restart = buttons.addButton("Restart now", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Later", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Version {version} has been downloaded."))
        layout.addWidget(self.notes)
        layout.addWidget(buttons)


def show_whats_new(version: str, notes: str) -> bool:
    """Show the notes; True when the user chose to restart into the new version."""
    return WhatsNewDialog(version, notes).exec() == QDialog.DialogCode.Accepted
