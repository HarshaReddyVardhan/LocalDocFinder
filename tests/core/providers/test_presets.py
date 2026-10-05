import pytest
from pydantic import ValidationError

from localdoc_finder.core.providers.presets import CUSTOM, PRESETS, preset_for
from localdoc_finder.core.settings import CloudProviderSettings

EXPECTED = {
    "openrouter",
    "openai",
    "gemini",
    "anthropic",
    "groq",
    "mistral",
    "deepseek",
    "custom",
}


def test_every_planned_service_has_a_preset() -> None:
    assert set(PRESETS) == EXPECTED
    for key, preset in PRESETS.items():
        assert preset.key == key
        assert preset.label


@pytest.mark.parametrize("key", sorted(EXPECTED - {CUSTOM}))
def test_real_services_have_an_https_address_and_a_key_page(key: str) -> None:
    preset = PRESETS[key]
    assert preset.base_url.startswith("https://")
    assert preset.key_url.startswith("https://")


def test_custom_has_neither_address_nor_key_page() -> None:
    assert PRESETS[CUSTOM].base_url == ""
    assert PRESETS[CUSTOM].key_url == ""


def test_exclusions_hide_models_that_cannot_chat() -> None:
    openai = PRESETS["openai"].exclude
    assert openai is not None
    assert openai.search("text-embedding-3-small")
    assert openai.search("gpt-4o-mini-tts")
    assert openai.search("Whisper-1")
    assert not openai.search("gpt-4o-mini")
    gemini = PRESETS["gemini"].exclude
    assert gemini is not None
    assert gemini.search("gemini-embedding-001")
    assert not gemini.search("gemini-2.5-flash")


def test_service_quirks() -> None:
    assert PRESETS["gemini"].strip_prefix == "models/"
    assert PRESETS["anthropic"].json_mode == "prompt"
    assert PRESETS["openai"].json_mode == "schema"
    assert PRESETS["openrouter"].headers == {"X-Title": "LocalDoc Finder"}


def test_unknown_keys_behave_as_custom() -> None:
    assert preset_for("nope") is PRESETS[CUSTOM]
    assert preset_for("groq") is PRESETS["groq"]


def test_settings_accept_known_presets_and_reject_unknown_ones() -> None:
    assert CloudProviderSettings(base_url="u").preset == "custom"
    assert CloudProviderSettings(base_url="u", preset="gemini").preset == "gemini"
    with pytest.raises(ValidationError, match="unknown preset"):
        CloudProviderSettings(base_url="u", preset="skynet")


def test_favorites_default_to_empty_and_are_not_shared() -> None:
    first = CloudProviderSettings(base_url="u")
    second = CloudProviderSettings(base_url="u")
    assert first.favorites == [] and first.favorites is not second.favorites
