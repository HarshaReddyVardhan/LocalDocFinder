"""Provider presets: what differs between OpenAI-compatible services, as plain data.

Every preset runs through ``OpenAICompatibleProvider``; adding a service is adding an entry here,
not code. ``custom`` is the escape hatch for any other endpoint (LM Studio, vLLM, a proxy).
"""

import re
from dataclasses import dataclass, field
from typing import Literal

JsonMode = Literal["schema", "prompt"]
CUSTOM = "custom"


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    base_url: str  # empty for ``custom``: the user enters it
    key_url: str  # where to create an API key; empty when there is no such page
    # Model ids matching this cannot chat (embeddings, speech, images ...) and are hidden.
    exclude: re.Pattern[str] | None = None
    strip_prefix: str = ""  # Gemini lists ids as ``models/gemini-...``
    # "schema": send response_format. "prompt": spell the schema out in the prompt instead, for
    # services whose compatibility layer ignores response_format (Anthropic).
    json_mode: JsonMode = "schema"
    headers: dict[str, str] = field(default_factory=dict)


def _pattern(words: str) -> re.Pattern[str]:
    return re.compile(words, re.IGNORECASE)


PRESETS: dict[str, ProviderPreset] = {
    preset.key: preset
    for preset in (
        ProviderPreset(
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/keys",
            headers={"X-Title": "LocalDoc Finder"},
        ),
        ProviderPreset(
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            "https://platform.openai.com/api-keys",
            exclude=_pattern(
                "embedding|tts|whisper|dall-e|moderation|realtime|audio|transcribe|image|search"
            ),
        ),
        ProviderPreset(
            "gemini",
            "Google Gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "https://aistudio.google.com/apikey",
            exclude=_pattern("embedding|imagen|veo|aqa|tts"),
            strip_prefix="models/",
        ),
        ProviderPreset(
            "anthropic",
            "Anthropic (Claude)",
            "https://api.anthropic.com/v1/",
            "https://console.anthropic.com/settings/keys",
            json_mode="prompt",
        ),
        ProviderPreset(
            "groq",
            "Groq",
            "https://api.groq.com/openai/v1",
            "https://console.groq.com/keys",
            exclude=_pattern("whisper|guard|tts"),
        ),
        ProviderPreset(
            "mistral",
            "Mistral",
            "https://api.mistral.ai/v1",
            "https://console.mistral.ai/api-keys",
            exclude=_pattern("embed|moderation|ocr"),
        ),
        ProviderPreset(
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com/v1",
            "https://platform.deepseek.com/api_keys",
        ),
        ProviderPreset(CUSTOM, "Custom (OpenAI-compatible)", "", ""),
    )
}


def preset_for(key: str) -> ProviderPreset:
    """The preset for ``key``; unknown keys behave as ``custom``."""
    return PRESETS.get(key, PRESETS[CUSTOM])
