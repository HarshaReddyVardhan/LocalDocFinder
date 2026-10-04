from pathlib import Path

import pytest

from localdoc_finder.core import data_migration
from localdoc_finder.core.data_migration import migrate_legacy_data, rename_data_dir
from localdoc_finder.core.settings import (
    DATA_DIR_NAME,
    OLD_NAME_INSTALL_DIR_NAME,
    default_data_dir,
    legacy_data_dir,
)


def make_legacy(root: Path) -> Path:
    legacy = root / "VectorEmbed"
    (legacy / "current").mkdir(parents=True)
    (legacy / "current" / "VectorEmbed.exe").write_text("exe")
    (legacy / "packages").mkdir()
    (legacy / "Update.exe").write_text("u")
    (legacy / "state.sqlite").write_text("db")
    (legacy / "lance").mkdir()
    (legacy / "lance" / "t.lance").write_text("x")
    (legacy / "settings.toml").write_text("")
    return legacy


def test_data_and_install_folders_differ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_data_dir() == tmp_path / DATA_DIR_NAME
    assert legacy_data_dir() == tmp_path / OLD_NAME_INSTALL_DIR_NAME
    assert default_data_dir() != legacy_data_dir()


def test_moves_data_and_leaves_the_install(tmp_path: Path) -> None:
    legacy = make_legacy(tmp_path)
    new = tmp_path / "LocalDocFinderData"
    assert migrate_legacy_data(legacy, new)
    assert (new / "state.sqlite").read_text() == "db"
    assert (new / "lance" / "t.lance").exists()
    assert (new / "settings.toml").exists()
    assert not (new / "current").exists()
    assert (legacy / "current" / "VectorEmbed.exe").exists()
    assert (legacy / "Update.exe").exists()
    assert not (legacy / "state.sqlite").exists()


def test_skips_when_new_folder_has_content(tmp_path: Path) -> None:
    legacy = make_legacy(tmp_path)
    new = tmp_path / "LocalDocFinderData"
    new.mkdir()
    (new / "state.sqlite").write_text("new")
    assert not migrate_legacy_data(legacy, new)
    assert (legacy / "state.sqlite").exists()


def test_skips_when_no_data_markers(tmp_path: Path) -> None:
    legacy = tmp_path / "VectorEmbed"
    (legacy / "current").mkdir(parents=True)
    assert not migrate_legacy_data(legacy, tmp_path / "LocalDocFinderData")
    assert not migrate_legacy_data(tmp_path / "missing", tmp_path / "LocalDocFinderData")


def test_failure_is_logged_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = make_legacy(tmp_path)

    def boom(src: str, dst: str) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(data_migration.shutil, "move", boom)
    assert not migrate_legacy_data(legacy, tmp_path / "LocalDocFinderData")
    assert (legacy / "state.sqlite").exists()


def test_the_old_named_data_folder_is_renamed_whole(tmp_path: Path) -> None:
    old = tmp_path / "VectorEmbedData"
    (old / "lance").mkdir(parents=True)
    (old / "state.sqlite").write_text("db")
    new = tmp_path / "LocalDocFinderData"
    assert rename_data_dir(old, new)
    assert (new / "state.sqlite").read_text() == "db"
    assert not old.exists()
    assert not rename_data_dir(old, new)  # nothing left to move


def test_a_folder_in_use_is_left_and_still_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    old = tmp_path / "VectorEmbedData"
    old.mkdir()

    def in_use(self: Path, target: Path) -> None:
        raise PermissionError("an old copy has a file open")

    monkeypatch.setattr(Path, "rename", in_use)
    assert not rename_data_dir(old, tmp_path / DATA_DIR_NAME)
    assert default_data_dir() == old  # the index keeps working until the rename succeeds
    (tmp_path / DATA_DIR_NAME).mkdir()
    assert default_data_dir() == tmp_path / DATA_DIR_NAME
