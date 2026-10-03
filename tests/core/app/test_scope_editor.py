from pathlib import Path

from PySide6.QtWidgets import QApplication
from tests.core.setup.fakes import Harness

from vector_embed.app.scope_editor import ScopeEditor
from vector_embed.core.protection import SystemProtection
from vector_embed.core.settings import ScopeSettings

ENV = {"SystemRoot": r"C:\Windows", "SystemDrive": "C:"}
PROTECTION = SystemProtection(ENV, Path("C:/Users/me"))
DRIVES = ["C:\\", "D:\\"]


def editor() -> ScopeEditor:
    return ScopeEditor(
        choose_folder=lambda: None, list_drives=lambda: DRIVES, protection=PROTECTION
    )


def test_the_default_is_documents_from_the_whole_pc(qapp: QApplication) -> None:
    box = editor()
    choice = box.choice()
    assert (choice.coverage, choice.roots, choice.file_types) == ("entire_pc", [], "documents")
    assert box.is_valid()
    assert "recommended" in box.entire.text()
    assert "recommended" in box.types.itemText(0)


def test_it_lists_the_drives_and_marks_the_windows_drive(qapp: QApplication) -> None:
    box = editor()
    text = box.drives_note.text()
    assert "D:\\" in text
    assert r"C:\ (Windows drive" in text


def test_adding_a_folder_switches_to_a_chosen_scope(qapp: QApplication) -> None:
    box = editor()
    assert box.add_root(r"D:\Work")
    assert box.add_root(r"D:\Work")  # a duplicate is ignored
    assert box.choice().coverage == "chosen"
    assert box.roots() == [r"D:\Work"]


def test_a_chosen_scope_needs_a_folder(qapp: QApplication) -> None:
    box = editor()
    box.chosen.setChecked(True)
    assert not box.is_valid()
    assert "at least one" in box.warning.text()


def test_system_folders_are_refused_with_a_reason(qapp: QApplication) -> None:
    box = editor()
    assert not box.add_root(r"C:\Windows\System32")
    assert box.roots() == []
    assert "never indexed" in box.warning.text()


def test_choosing_the_windows_drive_warns_that_only_own_folders_count(qapp: QApplication) -> None:
    box = editor()
    box.add_root("C:\\")
    assert "Windows drive" in box.warning.text()
    assert box.roots() == ["C:\\"]


def test_the_drive_menu_adds_a_drive(qapp: QApplication) -> None:
    box = editor()
    box._fill_drive_menu()
    actions = {a.text(): a for a in box.add_drive.menu().actions()}
    assert set(actions) == {"C:\\", "D:\\"}
    actions["D:\\"].trigger()
    assert box.roots() == ["D:\\"]


def test_load_shows_saved_settings_and_remove_works(qapp: QApplication) -> None:
    box = editor()
    box.load(ScopeSettings(coverage="chosen", roots=("D:\a", "E:\b"), file_types="everything"))
    assert box.chosen.isChecked()
    assert box.types.currentData() == "everything"
    box.folders.setCurrentRow(0)
    box.remove_folder.click()
    assert box.roots() == ["E:\b"]


def test_the_wizard_saves_the_choice_when_leaving_the_page(
    qapp: QApplication, tmp_path: Path
) -> None:
    from tests.core.app.test_setup_wizard import wizard_for

    harness = Harness(tmp_path)
    wizard = wizard_for(harness)
    page = wizard.scope
    page.initializePage()
    page.editor.add_root(r"D:\Papers")
    assert page.validatePage()
    assert harness.saved()["scope"]["roots"] == [r"D:\Papers"]
    assert harness.saved()["scope"]["coverage"] == "chosen"
