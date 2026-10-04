import os
from pathlib import Path

import pytest

from localdoc_finder.core import settings as s


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No stray .env in the cwd, no LDF_* leakage from the developer's shell."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    for key in [k for k in os.environ if k.startswith("LDF_")]:
        monkeypatch.delenv(key)


def write_toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = s.load_settings(tmp_path / "nope.toml")
    assert cfg.schema_version == s.SCHEMA_VERSION
    assert cfg.power.require_ac_power is True
    assert cfg.power.chat_on_battery is False
    assert cfg.embedding.model == "qwen3-embedding:0.6b"


def test_data_dir_defaults_under_localappdata(tmp_path: Path) -> None:
    cfg = s.load_settings(tmp_path / "nope.toml")
    assert cfg.storage.data_dir == tmp_path / "appdata" / s.DATA_DIR_NAME


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    path = write_toml(tmp_path, '[power]\nac_settle_seconds = 5\n[embedding]\nmodel = "bge-m3"\n')
    cfg = s.load_settings(path)
    assert cfg.power.ac_settle_seconds == 5
    assert cfg.embedding.model == "bge-m3"
    assert cfg.power.poll_seconds == 30  # untouched fields keep defaults


def test_env_beats_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_toml(tmp_path, "[power]\nrequire_ac_power = true\n")
    monkeypatch.setenv("LDF_POWER__REQUIRE_AC_POWER", "false")
    monkeypatch.setenv("LDF_LOG_LEVEL", "DEBUG")
    cfg = s.load_settings(path)
    assert cfg.power.require_ac_power is False
    assert cfg.log_level == "DEBUG"


def test_dotenv_file_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "LDF_OLLAMA_HOST=http://example:1\nUNRELATED=1\n", encoding="utf-8"
    )
    assert s.load_settings(tmp_path / "nope.toml").ollama_host == "http://example:1"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "[power]\nrequire_ac = false\n")  # typo
    with pytest.raises(s.SettingsError, match="invalid settings"):
        s.load_settings(path)


def test_out_of_range_value_is_rejected(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "[idle]\ncpu_percent = 250\n")
    with pytest.raises(s.SettingsError):
        s.load_settings(path)


def test_malformed_toml_is_a_settings_error(tmp_path: Path) -> None:
    with pytest.raises(s.SettingsError, match="cannot read"):
        s.load_settings(write_toml(tmp_path, "[power\n"))


def test_settings_are_immutable(tmp_path: Path) -> None:
    cfg = s.load_settings(tmp_path / "nope.toml")
    with pytest.raises(ValueError, match="frozen"):
        cfg.log_level = "DEBUG"  # type: ignore[misc]  # asserting immutability


def test_profile_for_known_and_unknown_model(tmp_path: Path) -> None:
    emb = s.load_settings(tmp_path / "nope.toml").embedding
    assert emb.profile_for("nomic-embed-text").query == "search_query: "
    assert emb.profile_for("qwen3-embedding:0.6b").query.startswith("Instruct:")
    assert emb.profile_for("some-new-model") == s.EmbeddingProfile()
    assert emb.profile_for().query.startswith("Instruct:")  # the configured model


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("qwen3-embedding:0.6b", "qwen3-embedding"),
        ("qwen3-embedding:4b", "qwen3-embedding"),
        ("qwen3-embedding:8b", "qwen3-embedding"),
        ("nomic-embed-text:latest", "nomic-embed-text"),
        ("bge-m3:latest", "bge-m3"),
        ("BGE-M3", "bge-m3"),
        ("embeddinggemma:300m", "embeddinggemma"),
    ],
)
def test_profiles_resolve_by_family_whatever_the_tag(model: str, family: str) -> None:
    emb = s.EmbeddingSettings()
    assert emb.profile_for(model) == s.DEFAULT_EMBEDDING_PROFILES[family]


def test_every_default_profile_has_a_similarity_floor() -> None:
    assert all(p.min_similarity > 0 for p in s.DEFAULT_EMBEDDING_PROFILES.values())


def test_embeddinggemma_gets_its_task_prefixes() -> None:
    profile = s.EmbeddingSettings().profile_for("embeddinggemma")
    assert profile.query == "task: search result | query: "
    assert profile.document == "title: none | text: "


def test_a_tagged_profile_beats_its_family_and_settings_merge_over_defaults() -> None:
    emb = s.EmbeddingSettings(
        profiles={"qwen3-embedding:8b": s.EmbeddingProfile(min_similarity=0.42)}
    )
    assert emb.profile_for("qwen3-embedding:8b").min_similarity == 0.42
    assert emb.profile_for("qwen3-embedding:4b").min_similarity == 0.50
    assert emb.profile_for("bge-m3").min_similarity == 0.53  # the defaults are kept


def test_migrate_applies_chain_up_to_current(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(s, "SCHEMA_VERSION", 3)
    calls: list[int] = []

    def step(version: int) -> s.Migration:
        def run(raw: s.RawSettings) -> s.RawSettings:
            calls.append(version)
            return raw

        return run

    out = s.migrate({"schema_version": 1}, migrations={1: step(1), 2: step(2)})
    assert calls == [1, 2]
    assert out["schema_version"] == 3


def test_v4_rank_fusion_settings_are_dropped() -> None:
    raw = {
        "schema_version": 4,
        "search": {"rrf_k": 60, "filename_boost": 2.0, "filename_exact_boost": 6.0, "results": 9},
    }
    out = s.migrate(raw)
    assert out["search"] == {"results": 9}
    assert s.SearchSettings(**out["search"]).results == 9
    untouched = s.migrate({"schema_version": 4, "search": {"results": 9}})
    assert untouched["search"] == {"results": 9}


def test_v3_prefixes_become_profiles() -> None:
    raw = {
        "schema_version": 3,
        "embedding": {"model": "m", "prefixes": {"m": {"query": "Q: ", "document": "D: "}}},
    }
    out = s.migrate(raw)
    assert out["embedding"] == {
        "model": "m",
        "profiles": {"m": {"query": "Q: ", "document": "D: "}},
    }
    assert s.migrate({"schema_version": 3})["schema_version"] == s.SCHEMA_VERSION
    assert s.EmbeddingSettings(**out["embedding"]).profile_for().query == "Q: "


def test_v2_settings_lose_the_old_document_list() -> None:
    raw = {"schema_version": 2, "scope": {"file_types": "documents", "document_exts": [".pdf"]}}
    out = s.migrate(raw)
    assert out["scope"] == {"file_types": "documents"}
    assert s.migrate({"schema_version": 2})["schema_version"] == s.SCHEMA_VERSION


def test_migrate_missing_step_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(s, "SCHEMA_VERSION", 2)
    with pytest.raises(s.SettingsError, match="no migration"):
        s.migrate({"schema_version": 1}, migrations={})


def test_newer_schema_is_refused() -> None:
    with pytest.raises(s.SettingsError, match="newer"):
        s.migrate({"schema_version": s.SCHEMA_VERSION + 1})


@pytest.mark.parametrize("bad", [0, -1, "1", True])
def test_invalid_schema_version_is_refused(bad: object) -> None:
    with pytest.raises(s.SettingsError, match="invalid schema_version"):
        s.migrate({"schema_version": bad})


def test_missing_schema_version_means_current() -> None:
    assert s.migrate({"log_level": "INFO"}) == {"log_level": "INFO"}


def test_default_settings_path_honours_data_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LDF_STORAGE__DATA_DIR", str(tmp_path / "custom"))
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / s.SETTINGS_FILENAME).write_text("[power]\npoll_seconds = 9\n")
    assert s.load_settings().power.poll_seconds == 9


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(s.SettingsError, match="unknown settings keys"):
        s.load_settings(write_toml(tmp_path, 'log_levle = "DEBUG"\n'))


def test_a_dotenv_in_the_working_folder_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "some-repo"
    repo.mkdir()
    (repo / ".env").write_text("LDF_OLLAMA_HOST=http://evil.example:1\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = s.load_settings(data_dir / "settings.toml")
    assert cfg.ollama_host == "http://127.0.0.1:11434"


def test_the_dotenv_beside_settings_toml_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / ".env").write_text("LDF_OLLAMA_HOST=http://example:2\n", encoding="utf-8")
    assert s.load_settings(data_dir / "settings.toml").ollama_host == "http://example:2"
