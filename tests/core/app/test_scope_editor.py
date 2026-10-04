from pathlib import Path

from PySide6.QtWidgets import QApplication
from tests.core.setup.fakes import Harness

from vector_embed.app.scope_editor import ScopeEditor
from vector_embed.core.file_kinds import FileKind
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


def test_a_folder_added_on_the_whole_pc_is_an_extra(qapp: QApplication) -> None:
    box = editor()
    assert box.add_root(r"E:\External")
    assert box.add_root(r"E:\External")  # a duplicate is ignored
    choice = box.choice()
    assert (choice.coverage, choice.roots) == ("entire_pc", [r"E:\External"])
    assert "on top of the whole PC" in box.list_note.text()
    box.chosen.setChecked(True)
    assert "only places" in box.list_note.text()


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
    box.chosen.setChecked(True)
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


def test_a_preset_names_what_it_includes_and_hides_the_checklist(qapp: QApplication) -> None:
    box = editor()
    assert box.types_summary.text() == "Includes: Documents, Text and notes"
    assert box.kinds_panel.isHidden()
    box.types.setCurrentIndex(box.types.findData("everything"))
    assert "Source code" in box.types_summary.text()
    assert box.choice().kinds == ()


def test_custom_shows_every_kind_ticked_from_the_last_preset(qapp: QApplication) -> None:
    box = editor()
    box.types.setCurrentIndex(box.types.findData("custom"))
    assert not box.kinds_panel.isHidden()
    assert box.types_summary.isHidden()
    assert set(box.kind_boxes) == set(FileKind)
    assert box.choice().kinds == ("documents", "notes")
    box.kind_boxes[FileKind.CODE].setChecked(True)
    box.kind_boxes[FileKind.NOTES].setChecked(False)
    assert box.choice().kinds == ("documents", "code")


def test_custom_needs_at_least_one_kind(qapp: QApplication) -> None:
    box = editor()
    box.types.setCurrentIndex(box.types.findData("custom"))
    for check in box.kind_boxes.values():
        check.setChecked(False)
    assert not box.is_valid()
    assert "at least one kind" in box.warning.text()


def test_load_ticks_the_saved_custom_kinds(qapp: QApplication) -> None:
    box = editor()
    box.load(ScopeSettings(file_types="custom", custom_kinds=frozenset({FileKind.IMAGES})))
    assert box.types.currentData() == "custom"
    assert box.choice().kinds == ("images",)


def test_the_wizard_saves_custom_kinds(qapp: QApplication, tmp_path: Path) -> None:
    from tests.core.app.test_setup_wizard import wizard_for

    harness = Harness(tmp_path)
    wizard = wizard_for(harness)
    page = wizard.scope
    page.initializePage()
    page.editor.types.setCurrentIndex(page.editor.types.findData("custom"))
    page.editor.kind_boxes[FileKind.CODE].setChecked(True)
    assert page.validatePage()
    assert harness.saved()["scope"]["file_types"] == "custom"
    assert harness.saved()["scope"]["custom_kinds"] == ["code", "documents", "notes"]


def test_the_wizard_saves_the_choice_when_leaving_the_page(
    qapp: QApplication, tmp_path: Path
) -> None:
    from tests.core.app.test_setup_wizard import wizard_for

    harness = Harness(tmp_path)
    wizard = wizard_for(harness)
    page = wizard.scope
    page.initializePage()
    page.editor.add_root(r"D:\Papers")
    page.editor.chosen.setChecked(True)
    assert page.validatePage()
    assert harness.saved()["scope"]["roots"] == [r"D:\Papers"]
    assert harness.saved()["scope"]["coverage"] == "chosen"
