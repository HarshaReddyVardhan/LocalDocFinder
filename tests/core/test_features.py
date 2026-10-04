from pathlib import Path

from localdoc_finder.core.features import FEATURE_TITLES, FEATURES, enabled_features, is_enabled
from localdoc_finder.core.settings import FeatureSettings, Settings, load_settings
from localdoc_finder.core.settings_schema import editable_options


def test_every_feature_has_a_title() -> None:
    assert set(FEATURES) == {"ask", "chat", "match"}
    assert set(FEATURE_TITLES) == set(FEATURES)


def test_a_fresh_install_is_search_only() -> None:
    assert enabled_features(Settings()) == frozenset()


def test_enabled_features_lists_the_ones_switched_on() -> None:
    settings = Settings(features=FeatureSettings(chat=True, match=True))
    assert enabled_features(settings) == {"chat", "match"}


def test_search_and_other_skills_are_always_on() -> None:
    assert is_enabled("search", frozenset())
    assert is_enabled("summarize", frozenset())
    assert not is_enabled("ask", frozenset())
    assert is_enabled("ask", {"ask"})


def test_settings_from_before_optional_features_keep_them_on(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('schema_version = 1\n[search]\nhotkey = "ctrl+alt+k"\n', encoding="utf-8")
    assert enabled_features(load_settings(path)) == set(FEATURES)


def test_the_upgrade_keeps_a_feature_already_switched_off(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("schema_version = 1\n[features]\nmatch = false\n", encoding="utf-8")
    assert enabled_features(load_settings(path)) == {"ask", "chat"}


def test_a_new_settings_file_is_search_only(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[search]\nhotkey = "ctrl+alt+k"\n', encoding="utf-8")  # no version: current
    assert enabled_features(load_settings(path)) == frozenset()


def test_features_have_their_own_tab_not_the_advanced_one() -> None:
    assert not any(option.section == "features" for option in editable_options())
