"""Which optional features (Ask, Chat, Match) the user turned on.

Search is always available and needs only the embedder. The others need a chat model, so setup
downloads one only when a feature that uses it is wanted, and the front-ends hide the rest.
"""

from collections.abc import Collection

from vector_embed.core.settings import FeatureSettings, Settings

FEATURES: tuple[str, ...] = tuple(FeatureSettings.model_fields)  # ask, chat, match
FEATURE_TITLES = {
    "ask": "Ask: answers from your files, with citations",
    "chat": "Chat: talk with your files and follow up",
    "match": "Match: rank documents against a job description or similar",
}


def enabled_features(settings: Settings) -> frozenset[str]:
    """Names of the features switched on in ``settings``."""
    return frozenset(name for name in FEATURES if getattr(settings.features, name))


def is_enabled(name: str, enabled: Collection[str]) -> bool:
    """Search and any skill that is not an optional feature are always on."""
    return name not in FEATURES or name in enabled
