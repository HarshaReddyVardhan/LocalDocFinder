"""The "what will be sent" dialog shown before any cloud request."""

from PySide6.QtWidgets import QApplication, QMessageBox

from localdoc_finder.app.assistant import CloudPreview


def confirm_cloud_dialog(preview: CloudPreview) -> bool:
    """Show the privacy badge and let the user inspect the exact text before anything is sent."""
    box = QMessageBox(QApplication.activeWindow())  # owned by the popup, so it stays on top of it
    box.setWindowTitle("Answer better with the cloud?")
    note = f"\n{preview.shield}" if preview.shield else ""
    box.setText(preview.badge + note)
    box.setInformativeText("Nothing is sent until you press Send.")
    box.setDetailedText(preview.text)  # the "View what will be sent" panel
    send = box.addButton("Send", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.exec()
    return box.clickedButton() is send
