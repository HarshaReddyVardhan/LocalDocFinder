import tomllib
from pathlib import Path

import pytest

from localdoc_finder.core import settings as s
from localdoc_finder.core.settings_io import set_setting


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


class TestSafeWrites:
    def test_concurrent_writers_do_not_lose_each_others_changes(self, tmp_path: Path) -> None:
        import threading

        path = tmp_path / "settings.toml"
        errors: list[BaseException] = []

        def write(key: str, value: int) -> None:
            try:
                set_setting(path, ["power", key], value)
            except BaseException as exc:
                errors.append(exc)

        # settings changed at the same moment, as the tray, the CLI and the wizard might do
        threads = [
            threading.Thread(target=write, args=("ac_settle_seconds", 11)),
            threading.Thread(target=set_setting, args=(path, ["idle", "cpu_percent"], 30)),
            threading.Thread(target=set_setting, args=(path, ["idle", "cpu_seconds"], 45)),
            threading.Thread(target=set_setting, args=(path, ["log_level"], "WARNING")),
            threading.Thread(target=set_setting, args=(path, ["search", "hotkey"], "ctrl+alt+f9")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        saved = read(path)
        assert saved["power"] == {"ac_settle_seconds": 11}
        assert saved["idle"] == {"cpu_percent": 30, "cpu_seconds": 45}
        assert saved["log_level"] == "WARNING"
        assert saved["search"] == {"hotkey": "ctrl+alt+f9"}

    def test_a_held_lock_times_out_with_a_clear_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import localdoc_finder.core.settings_io as io_module

        monkeypatch.setattr(io_module, "LOCK_TIMEOUT_SECONDS", 0.2)
        path = tmp_path / "settings.toml"
        # the same process asking again through a second handle behaves like another process
        with (
            io_module._locked(path),
            pytest.raises(s.SettingsError, match="another program"),
            io_module._locked(path),
        ):
            pass

    def test_the_lock_is_released_after_a_failed_change(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.toml"
        with pytest.raises(s.SettingsError):
            set_setting(path, ["idle", "cpu_percent"], 500)
        set_setting(path, ["idle", "cpu_percent"], 20)  # would hang or fail if still locked
        assert read(path)["idle"] == {"cpu_percent": 20}

    def test_no_temporary_files_are_left_behind(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.toml"
        set_setting(path, ["log_level"], "DEBUG")
        with pytest.raises(s.SettingsError):
            set_setting(path, ["idle", "cpu_percent"], 500)
        leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_a_briefly_locked_target_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "settings.toml"
        original = Path.replace
        attempts: list[int] = []

        def flaky(self: Path, target: Path) -> Path:
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError("the file is in use")  # an antivirus scan, say
            return original(self, target)

        monkeypatch.setattr(Path, "replace", flaky)
        set_setting(path, ["log_level"], "DEBUG")
        assert len(attempts) == 3
        assert read(path)["log_level"] == "DEBUG"

    def test_a_permanently_locked_target_gives_up_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "settings.toml"

        def refuse(self: Path, target: Path) -> Path:
            raise PermissionError("held open by another program")

        monkeypatch.setattr(Path, "replace", refuse)
        with pytest.raises(PermissionError):
            set_setting(path, ["log_level"], "DEBUG")
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []

    def test_what_is_written_is_the_upgraded_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def rename_key(data: dict[str, object]) -> dict[str, object]:
            data = dict(data)
            if "old_level" in data:
                data["log_level"] = data.pop("old_level")
            return data

        monkeypatch.setattr(s, "MIGRATIONS", {1: rename_key})
        monkeypatch.setattr(s, "SCHEMA_VERSION", 2)
        path = tmp_path / "settings.toml"
        path.write_text('schema_version = 1\nold_level = "DEBUG"\n', encoding="utf-8")
        set_setting(path, ["search", "hotkey"], "ctrl+alt+f9")
        saved = read(path)
        assert saved["schema_version"] == 2
        assert saved["log_level"] == "DEBUG"  # the migrated key, not the stale old one
        assert "old_level" not in saved
        assert saved["search"] == {"hotkey": "ctrl+alt+f9"}


class TestSettingsFileLocation:
    def test_a_loaded_configuration_remembers_its_file(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.toml"
        path.write_text('log_level = "DEBUG"\n', encoding="utf-8")
        assert s.load_settings(path).settings_path() == path

    def test_a_data_dir_entry_moves_the_index_but_not_the_settings_file(
        self, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "bigdisk" / "ldf"
        path = tmp_path / "settings.toml"
        path.write_text(f'[storage]\ndata_dir = "{elsewhere.as_posix()}"\n', encoding="utf-8")
        loaded = s.load_settings(path)
        assert loaded.storage.data_dir == elsewhere
        assert loaded.settings_path() == path  # saves go where they will be read from
        assert loaded.settings_path() != elsewhere / "settings.toml"

    def test_changes_made_through_the_controller_are_read_back(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "ve-data"
        path = tmp_path / "settings.toml"
        path.write_text(f'[storage]\ndata_dir = "{elsewhere.as_posix()}"\n', encoding="utf-8")
        loaded = s.load_settings(path)
        set_setting(loaded.settings_path(), ["log_level"], "WARNING")
        assert s.load_settings(path).log_level == "WARNING"  # it used to land beside the index

    def test_an_unloaded_configuration_falls_back_to_its_data_folder(self, tmp_path: Path) -> None:
        settings = s.Settings(storage=s.StorageSettings(data_dir=tmp_path / "d"))
        assert settings.settings_path() == tmp_path / "d" / "settings.toml"

    def test_the_file_cannot_name_itself_in_the_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.toml"
        path.write_text('settings_file = "C:/elsewhere.toml"\n', encoding="utf-8")
        assert s.load_settings(path).settings_path() == path
