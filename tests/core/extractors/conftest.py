from collections.abc import Callable
from pathlib import Path

import pytest

from vector_embed.core.extractors.base import ExtractContext, ExtractorSet
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import ChunkingSettings, ScopeSettings

Writer = Callable[[str, str | bytes], Path]


@pytest.fixture
def ctx(scope_settings: ScopeSettings) -> ExtractContext:
    return ExtractContext(ScopePolicy(scope_settings), scope_settings)


@pytest.fixture
def extractors(ctx: ExtractContext) -> ExtractorSet:
    return ExtractorSet(ctx)


@pytest.fixture
def with_chunking(ctx: ExtractContext) -> Callable[[ChunkingSettings], ExtractorSet]:
    """Build an ExtractorSet using custom chunk-size settings."""

    def build(chunking: ChunkingSettings) -> ExtractorSet:
        return ExtractorSet(
            ExtractContext(ctx.scope, ctx.scope_settings, chunking=chunking, images=ctx.images)
        )

    return build


@pytest.fixture
def write(tmp_path: Path) -> Writer:
    def _write(name: str, content: str | bytes) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def no_merge(with_chunking: Callable[[ChunkingSettings], ExtractorSet]) -> ExtractorSet:
    """Extractors that never merge small neighbouring chunks, so each unit stays visible."""
    return with_chunking(ChunkingSettings(min_chunk_chars=0))
