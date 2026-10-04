from pathlib import Path

import pytest

from localdoc_finder.core.file_kinds import FileKind
from localdoc_finder.core.protection import SystemProtection
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.scope_roots import resolve_roots
from localdoc_finder.core.settings import ScopeSettings

ENV = {
    "SystemRoot": r"C:\Windows",
    "ProgramFiles": r"C:\Program Files",
    "ProgramFiles(x86)": r"C:\Program Files (x86)",
    "ProgramData": r"C:\ProgramData",
    "SystemDrive": "C:",
}
HOME = Path(r"C:\Users\me")


@pytest.fixture
def protection() -> SystemProtection:
    return SystemProtection(ENV, HOME)


# ------------------------------------------------------------------ protection
@pytest.mark.parametrize(
    "path",
    [
        r"C:\Windows",
        r"C:\Windows\System32\drivers\etc\hosts",
        r"c:\program files\App\readme.txt",
        r"C:\Program Files (x86)\Thing",
        r"C:\ProgramData\x",
        r"C:\Users\someone_else\Documents\a.pdf",
        r"D:\$RECYCLE.BIN\S-1\a.docx",
        r"E:\System Volume Information\x",
        r"D:\Windows.old\Users\me\a.txt",
    ],
)
def test_system_places_are_protected(protection: SystemProtection, path: str) -> None:
    assert protection.is_protected(path)
    assert protection.reason(path)


@pytest.mark.parametrize(
    "path",
    [r"C:\Users\me\Documents\a.pdf", r"C:\Users\me", r"D:\Work", r"C:\Work\notes.md", "D:\\"],
)
def test_own_places_are_not_protected(protection: SystemProtection, path: str) -> None:
    assert not protection.is_protected(path)


def test_the_system_drive_root_is_recognised(protection: SystemProtection) -> None:
    assert protection.is_system_drive_root("C:\\")
    assert protection.is_system_drive_root("c:")
    assert not protection.is_system_drive_root("D:\\")
    assert not protection.is_system_drive_root(r"C:\Users")


def test_the_scope_never_enters_or_indexes_protected_paths(protection: SystemProtection) -> None:
    scope = ScopePolicy(ScopeSettings(file_types="everything"), protection=protection)
    assert not scope.should_descend(r"C:\Windows\System32")
    assert not scope.should_descend(r"C:\Users\someone_else")
    assert not scope.is_valid_file(r"C:\Program Files\App\README.md")


def test_choosing_the_windows_folder_itself_still_indexes_nothing(
    protection: SystemProtection,
) -> None:
    # The blocked-name rule only applies below a chosen root, so a root of C:\Windows needs
    # the protection to stop it.
    settings = ScopeSettings(coverage="chosen", roots=(r"C:\Windows",), file_types="everything")
    scope = ScopePolicy(settings, protection=protection)
    assert not scope.is_valid_file(r"C:\Windows\notes.txt")
    assert not scope.should_descend(r"C:\Windows\Temp")


# ------------------------------------------------------------------ roots
def dirs(*names: str) -> object:
    return lambda _root: [rf"C:\{name}" for name in names]


def resolve(
    settings: ScopeSettings, protection: SystemProtection, **kwargs: object
) -> tuple[str, ...]:
    home = HOME
    return resolve_roots(
        settings,
        drives=lambda: ["C:\\", "D:\\"],
        protection=protection,
        home=home,
        list_dirs=kwargs.get("list_dirs", lambda _r: []),  # type: ignore[arg-type]
    )


def test_the_whole_pc_is_every_fixed_drive_with_the_windows_drive_made_safe(
    protection: SystemProtection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from localdoc_finder.core import scope_roots

    monkeypatch.setattr(scope_roots, "is_reparse_point", lambda _path: False)  # fake folders
    home = tmp_path / "me"
    home.mkdir()
    roots = resolve_roots(
        ScopeSettings(),
        drives=lambda: ["C:\\", "D:\\"],
        protection=protection,
        home=home,
        list_dirs=lambda _r: [
            r"C:\Windows",
            r"C:\Program Files",
            r"C:\Users",
            r"C:\$Recycle.Bin",
            r"C:\node_modules",
            r"C:\Work",
            r"C:\Docs",
        ],
    )
    assert roots == (str(home), r"C:\Work", r"C:\Docs", "D:\\")


def test_a_chosen_drive_and_folder_are_used_as_chosen(protection: SystemProtection) -> None:
    settings = ScopeSettings(coverage="chosen", roots=("D:\\", r"E:\Papers"))
    assert resolve(settings, protection) == ("D:\\", r"E:\Papers")


def test_chosen_system_folders_are_dropped(protection: SystemProtection) -> None:
    settings = ScopeSettings(coverage="chosen", roots=(r"C:\Windows", r"D:\Work"))
    assert resolve(settings, protection) == (r"D:\Work",)


def test_extra_folders_are_added_on_top_of_the_whole_pc(protection: SystemProtection) -> None:
    settings = ScopeSettings(coverage="entire_pc", roots=(r"E:\External", r"C:\Windows"))
    # the drives (C: has no home or folders in this fake setup) plus the extra; a protected
    # extra is still dropped
    assert resolve(settings, protection) == ("D:\\", r"E:\External")


def test_nested_and_duplicate_roots_are_scanned_once(protection: SystemProtection) -> None:
    settings = ScopeSettings(coverage="chosen", roots=(r"D:\Work", "D:\\", r"d:\work"))
    assert resolve(settings, protection) == ("D:\\",)


# ------------------------------------------------------------------ settings
def test_documents_are_the_default_file_types() -> None:
    scope = ScopePolicy(ScopeSettings(), protection=SystemProtection(ENV, HOME))
    assert ScopeSettings().coverage == "entire_pc"
    assert ScopeSettings().file_types == "documents"
    assert not scope.is_valid_file(r"D:\Work\main.py")  # code is not a document
    assert not scope.is_valid_file(r"D:\Work\photo.png")


def test_documents_only_accepts_the_document_types(tmp_path: Path) -> None:
    settings = ScopeSettings(
        coverage="chosen", roots=(str(tmp_path),), blocked_dirs=frozenset({"windows"})
    )
    scope = ScopePolicy(settings, protection=SystemProtection(ENV, Path.home()))
    for name in ("a.pdf", "b.docx", "c.pptx", "d.txt", "e.md", "f.rtf"):
        (tmp_path / name).write_text("x")
    for name in ("g.py", "h.json", "i.png", "Dockerfile"):
        (tmp_path / name).write_text("x")
    kept = sorted(f.name for f in tmp_path.iterdir() if scope.is_valid_file(f))
    assert kept == ["a.pdf", "b.docx", "c.pptx", "d.txt", "e.md", "f.rtf"]


def test_everything_adds_code_and_images(tmp_path: Path) -> None:
    settings = ScopeSettings(
        coverage="chosen",
        roots=(str(tmp_path),),
        file_types="everything",
        blocked_dirs=frozenset({"windows"}),
    )
    scope = ScopePolicy(settings, protection=SystemProtection(ENV, Path.home()))
    (tmp_path / "g.py").write_text("x = 1")
    (tmp_path / "i.png").write_bytes(b"x")
    assert scope.is_valid_file(tmp_path / "g.py")
    assert scope.is_valid_file(tmp_path / "i.png")


def test_custom_indexes_only_the_chosen_kinds(tmp_path: Path) -> None:
    settings = ScopeSettings(
        coverage="chosen",
        roots=(str(tmp_path),),
        file_types="custom",
        custom_kinds=frozenset({FileKind.IMAGES, FileKind.CODE}),
        blocked_dirs=frozenset({"windows"}),
    )
    scope = ScopePolicy(settings, protection=SystemProtection(ENV, Path.home()))
    for name in ("a.pdf", "b.md", "c.py", "d.jpg", "e.json", "Dockerfile"):
        (tmp_path / name).write_text("x")
    kept = sorted(f.name for f in tmp_path.iterdir() if scope.is_valid_file(f))
    assert kept == ["Dockerfile", "c.py", "d.jpg"]


def test_custom_without_kinds_is_invalid() -> None:
    with pytest.raises(ValueError, match="no kinds"):
        ScopeSettings(file_types="custom", custom_kinds=frozenset())


def test_presets_name_their_kinds() -> None:
    assert ScopeSettings().enabled_kinds() == {FileKind.DOCUMENTS, FileKind.NOTES}
    assert ScopeSettings(file_types="everything").enabled_kinds() == set(FileKind)


def test_every_kind_extension_has_an_extractor_route() -> None:
    cfg = ScopeSettings()
    offered = set().union(*cfg.kind_exts.values())
    assert offered == cfg.text_exts | cfg.doc_exts | cfg.image_exts
    assert set(cfg.kind_exts) == set(FileKind)


def test_file_kind_classifies_by_extension_and_name() -> None:
    scope = ScopePolicy(ScopeSettings(), protection=SystemProtection(ENV, HOME))
    assert scope.file_kind(r"D:\a\Scan.TIFF") is FileKind.IMAGES
    assert scope.file_kind(r"D:\a\Makefile") is FileKind.CODE
    assert scope.file_kind(r"D:\a\sheet.csv") is FileKind.DATA
    assert scope.file_kind(r"D:\a\movie.mp4") is None


def test_listed_folders_without_a_coverage_mean_a_choice() -> None:
    assert ScopeSettings.model_validate({"roots": [r"D:\a"]}).coverage == "chosen"


def test_a_chosen_scope_without_folders_is_invalid() -> None:
    with pytest.raises(ValueError, match="no folders"):
        ScopeSettings(coverage="chosen")


def test_fixed_drives_lists_only_internal_disks(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import namedtuple

    from localdoc_finder.core import drives

    part = namedtuple("part", "mountpoint opts")
    monkeypatch.setattr(
        drives.psutil,
        "disk_partitions",
        lambda all: [
            part("D:\\", "rw,fixed"),
            part("C:\\", "rw,fixed"),
            part("E:\\", "rw,removable"),
            part("F:\\", "ro,cdrom"),
        ],
    )
    assert drives.fixed_drives() == ["C:\\", "D:\\"]


def test_fixed_drives_survives_a_listing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from localdoc_finder.core import drives

    def broken(all: bool) -> list[object]:  # psutil's parameter name
        raise OSError("no")

    monkeypatch.setattr(drives.psutil, "disk_partitions", broken)
    assert drives.fixed_drives() == []
