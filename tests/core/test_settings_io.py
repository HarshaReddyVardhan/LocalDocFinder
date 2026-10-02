import tomllib
from pathlib import Path

import pytest

from vector_embed.core import settings as s
from vector_embed.core.settings_io import set_setting


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def read(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_creates_the_file_and_nested_tables(tmp_path: Path) -> None:
    path = tmp_path / "data" / "settings.toml"
    set_setting(path, ["models", "overrides", "chat"], "llama3.2")
    assert read(path) == {
        "schema_version": s.SCHEMA_VERSION,
        "models": {"overrides": {"chat": "llama3.2"}},
    }
    assert s.load_settings(path).models.overrides == {"chat": "llama3.2"}


def test_other_user_settings_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('log_level = "DEBUG"\n[power]\nac_settle_seconds = 5\n', encoding="utf-8")
    set_setting(path, ["embedding", "model"], "bge-m3")
    data = read(path)
    assert data["log_level"] == "DEBUG"
    assert data["power"] == {"ac_settle_seconds": 5}
    assert data["embedding"] == {"model": "bge-m3"}


def test_none_removes_a_key(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    set_setting(path, ["models", "overrides", "chat"], "llama3.2")
    set_setting(path, ["models", "overrides", "chat"], None)
    assert read(path)["models"] == {"overrides": {}}
    set_setting(path, ["models", "overrides", "never-set"], None)  # removing nothing is fine


def test_a_non_table_in_the_way_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("models = 5\n", encoding="utf-8")
    set_setting(path, ["models", "overrides", "chat"], "x")
    assert read(path)["models"] == {"overrides": {"chat": "x"}}


def test_invalid_values_are_refused_and_the_file_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    set_setting(path, ["power", "ac_settle_seconds"], 7)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(s.SettingsError, match="refusing to write"):
        set_setting(path, ["idle", "cpu_percent"], 500)
    assert path.read_text(encoding="utf-8") == before
    assert not path.with_suffix(".toml.tmp").exists()


def test_unreadable_toml_is_a_settings_error(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("[power\n", encoding="utf-8")
    with pytest.raises(s.SettingsError, match="cannot read"):
        set_setting(path, ["log_level"], "DEBUG")
