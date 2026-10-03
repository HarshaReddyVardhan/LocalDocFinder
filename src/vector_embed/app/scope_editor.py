"""What to index: the whole PC or chosen folders/drives, and which file types.

One widget shared by the setup wizard and the Settings window. It only collects the choice;
``SettingsController.set_scope`` validates and saves it.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from vector_embed.core.drives import fixed_drives
from vector_embed.core.protection import SystemProtection
from vector_embed.core.settings import ScopeSettings

ADD_FOLDER_TITLE = "Add a folder to index"
COVERAGE_ENTIRE = "entire_pc"
COVERAGE_CHOSEN = "chosen"
TYPES_DOCUMENTS = "documents"
TYPES_EVERYTHING = "everything"
MIN_COMBO_CHARS = 20
ALWAYS_SKIPPED = (
    "Never indexed, whatever you choose: Windows and program folders, other users' profiles, "
    "the recycle bin, build and cache folders, and files that look like passwords or keys."
)


@dataclass(frozen=True)
class ScopeChoice:
    coverage: str
    roots: list[str]
    file_types: str


def pick_folder() -> str | None:
    path = QFileDialog.getExistingDirectory(None, ADD_FOLDER_TITLE)
    return path or None


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


class ScopeEditor(QWidget):
    changed = Signal()

    def __init__(
        self,
        choose_folder: Callable[[], str | None] = pick_folder,
        list_drives: Callable[[], list[str]] = fixed_drives,
        protection: SystemProtection | None = None,
    ) -> None:
        super().__init__()
        self._choose_folder = choose_folder
        self._list_drives = list_drives
        self._protection = protection or SystemProtection()
        self.entire = QRadioButton("Documents from this whole PC (recommended)")
        self.chosen = QRadioButton("Only the folders and drives listed below")
        self.drives_note = _label()
        self.list_note = _label()
        self.folders = QListWidget()
        self.add_folder = QPushButton("Add folder…")
        self.add_drive = QPushButton("Add drive")
        self.remove_folder = QPushButton("Remove selected")
        self.types = QComboBox()
        # Without this the combo is as wide as its longest line and stops the window shrinking.
        self.types.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.types.setMinimumContentsLength(MIN_COMBO_CHARS)
        self.types.addItem(
            "Documents only: PDF, Word, PowerPoint, text, Markdown (recommended)", TYPES_DOCUMENTS
        )
        self.types.addItem("Documents, code, data files and images", TYPES_EVERYTHING)
        self.warning = _label()
        self._drive_menu = QMenu(self)
        self.add_drive.setMenu(self._drive_menu)
        self._drive_menu.aboutToShow.connect(self._fill_drive_menu)

        buttons = QHBoxLayout()
        for button in (self.add_folder, self.add_drive, self.remove_folder):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.entire)
        layout.addWidget(self.drives_note)
        layout.addWidget(self.chosen)
        layout.addWidget(self.list_note)
        layout.addWidget(self.folders, 1)
        layout.addLayout(buttons)
        layout.addWidget(_label("File types"))
        layout.addWidget(self.types)
        layout.addWidget(self.warning)
        layout.addWidget(_label(ALWAYS_SKIPPED))

        self.entire.setChecked(True)
        self._refresh_drives_note()
        self._sync_note()
        self.entire.toggled.connect(self._on_edited)
        self.types.currentIndexChanged.connect(self._on_edited)
        self.add_folder.clicked.connect(self._pick_folder)
        self.remove_folder.clicked.connect(self._remove_selected)

    # ------------------------------------------------------------------ values
    def load(self, scope: ScopeSettings) -> None:
        """Show the saved settings."""
        self.blockSignals(True)
        self.entire.setChecked(scope.coverage == COVERAGE_ENTIRE)
        self.chosen.setChecked(scope.coverage == COVERAGE_CHOSEN)
        self.folders.clear()
        self.folders.addItems(list(scope.roots))
        self.types.setCurrentIndex(max(self.types.findData(scope.file_types), 0))
        self.blockSignals(False)
        self._on_edited()

    def roots(self) -> list[str]:
        return [self.folders.item(i).text() for i in range(self.folders.count())]

    def choice(self) -> ScopeChoice:
        coverage = COVERAGE_ENTIRE if self.entire.isChecked() else COVERAGE_CHOSEN
        return ScopeChoice(coverage, self.roots(), self.types.currentData())

    def is_valid(self) -> bool:
        """A chosen scope needs at least one folder."""
        return self.entire.isChecked() or bool(self.roots())

    # ------------------------------------------------------------------ editing
    def add_root(self, path: str) -> bool:
        """Add a folder or drive; refuses system folders, says why, and ignores duplicates."""
        reason = self._protection.reason(path)
        if reason:
            self.warning.setText(f"{path} is {reason}; it is never indexed, so it was not added.")
            return False
        if path not in self.roots():
            self.folders.addItem(path)
        self._on_edited()
        return True

    def _pick_folder(self) -> None:
        folder = self._choose_folder()
        if folder:
            self.add_root(folder)

    def _fill_drive_menu(self) -> None:
        self._drive_menu.clear()
        for drive in self._list_drives():
            self._drive_menu.addAction(drive, self._adder(drive))

    def _adder(self, drive: str) -> Callable[[], bool]:
        return lambda: self.add_root(drive)

    def _remove_selected(self) -> None:
        row = self.folders.currentRow()
        if row >= 0:
            self.folders.takeItem(row)
            self._on_edited()

    # ------------------------------------------------------------------ state
    def _refresh_drives_note(self) -> None:
        drives = self._list_drives()
        names = [
            f"{drive} (Windows drive: only your own folders)"
            if self._protection.is_system_drive_root(drive)
            else drive
            for drive in drives
        ]
        self.drives_note.setText("Drives: " + (", ".join(names) or "none found"))

    def _sync_note(self) -> None:
        on = self.chosen.isChecked()
        self.list_note.setText(
            "These are the only places indexed."
            if on
            else "Folders and drives added here are indexed on top of the whole PC, for example "
            "an external drive, a network share or a folder that is skipped by default."
        )

    def _update_warning(self) -> None:
        system_roots = [r for r in self.roots() if self._protection.is_system_drive_root(r)]
        if system_roots:
            self.warning.setText(
                f"{system_roots[0]} is your Windows drive: only your own folders on it are "
                "indexed, not Windows, programs or other users."
            )
        elif self.chosen.isChecked() and not self.roots():
            self.warning.setText("Add at least one folder or drive.")
        else:
            self.warning.setText("")

    def _on_edited(self) -> None:
        self._sync_note()
        self._update_warning()
        self.changed.emit()
