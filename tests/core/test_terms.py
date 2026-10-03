from pathlib import Path

from vector_embed.core import terms
from vector_embed.core.store.sqlite import StateDb


def test_terms_are_accepted_per_version(tmp_path: Path) -> None:
    with StateDb(tmp_path) as state:
        assert not terms.terms_accepted(state)
        terms.accept_terms(state)
        assert terms.terms_accepted(state)
        state.set_meta(terms.TERMS_ACCEPTED_KEY, "0")  # an older version: ask again
        assert not terms.terms_accepted(state)


def test_the_text_covers_the_essentials() -> None:
    text = terms.TERMS_TEXT.lower()
    for topic in ("warranty", "liability", "ollama", "cloud", "files"):
        assert topic in text
