from pathlib import Path

import pytest

from vector_embed.core import data_migration
from vector_embed.core.data_migration import migrate_legacy_data
from vector_embed.core.settings import (
    APP_DIR_NAME,
    DATA_DIR_NAME,
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
    assert legacy_data_dir() == tmp_path / APP_DIR_NAME
    assert default_data_dir() != legacy_data_dir()


def test_moves_data_and_leaves_the_install(tmp_path: Path) -> None:
    legacy = make_legacy(tmp_path)
    new = tmp_path / "VectorEmbedData"
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
    new = tmp_path / "VectorEmbedData"
    new.mkdir()
    (new / "state.sqlite").write_text("new")
    assert not migrate_legacy_data(legacy, new)
    assert (legacy / "state.sqlite").exists()


def test_skips_when_no_data_markers(tmp_path: Path) -> None:
    legacy = tmp_path / "VectorEmbed"
    (legacy / "current").mkdir(parents=True)
    assert not migrate_legacy_data(legacy, tmp_path / "VectorEmbedData")
    assert not migrate_legacy_data(tmp_path / "missing", tmp_path / "VectorEmbedData")


def test_failure_is_logged_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = make_legacy(tmp_path)

    def boom(src: str, dst: str) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(data_migration.shutil, "move", boom)
    assert not migrate_legacy_data(legacy, tmp_path / "VectorEmbedData")
    assert (legacy / "state.sqlite").exists()
