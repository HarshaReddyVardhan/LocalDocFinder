"""The source contract and its registry."""

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Protocol

from vector_embed.core.projects import Projects
from vector_embed.core.registry import Registry, discover_modules
from vector_embed.core.scope import ScopePolicy


class ContentSource(Protocol):
    """Something that yields files to index and can say whether a known file still belongs."""

    name: str

    @property
    def roots(self) -> tuple[Path, ...]:
        """Where the source's files live; a manifest entry under a root belongs to it."""
        ...

    def iter_files(self, roots: Iterable[str | Path] | None = None) -> Iterator[Path]:
        """Every file to index, under ``roots`` (default: all of the source's roots)."""
        ...

    def still_valid(self, path: str) -> bool:
        """Whether an indexed ``path`` should stay indexed (exists and is still in scope)."""
        ...


SourceFactory = Callable[[Projects, ScopePolicy], ContentSource]
SOURCES: Registry[SourceFactory] = Registry("source")


def load_sources() -> None:
    """Import every module of this package so its ``@SOURCES.register`` decorators run."""
    discover_modules("vector_embed.core.sources")


def content_sources(projects: Projects, scope: ScopePolicy) -> list[ContentSource]:
    """One instance of every registered source, sharing the indexing scope."""
    load_sources()
    return [factory(projects, scope) for factory in SOURCES]
