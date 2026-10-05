"""The "what will be sent" dialog shown before a cloud request."""

from dataclasses import dataclass

from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox

from localdoc_finder.app.assistant import CloudPreview

REMEMBER_TEXT = "Don't ask again until LocalDoc Finder restarts"


@dataclass(frozen=True)
class CloudAnswer:
    """What the user decided: ``send`` this request, and ``remember`` that for the session."""

    send: bool
    remember: bool = False


def confirm_cloud_dialog(preview: CloudPreview) -> CloudAnswer:
    """Show the privacy badge and let the user inspect the exact text before anything is sent."""
    box = QMessageBox(QApplication.activeWindow())  # owned by the popup, so it stays on top of it
    box.setWindowTitle("Answer better with the cloud?")
    note = f"\n{preview.shield}" if preview.shield else ""
    box.setText(preview.badge + note)
    box.setInformativeText("Nothing is sent until you press Send.")
    box.setDetailedText(preview.text)  # the "View what will be sent" panel
    remember = QCheckBox(REMEMBER_TEXT)
    box.setCheckBox(remember)
    send = box.addButton("Send", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.exec()
    sent = box.clickedButton() is send
    return CloudAnswer(sent, remember=sent and remember.isChecked())  # cancelling remembers nothing
