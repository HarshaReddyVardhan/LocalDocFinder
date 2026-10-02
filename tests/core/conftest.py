from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.core.fakes import DIM, FakeEmbedder

from vector_embed.core.doctypes.base import DocTypeClassifierSet
from vector_embed.core.extractors.base import ExtractContext, ExtractorSet
from vector_embed.core.indexer import Indexer
from vector_embed.core.projects import Projects
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import ScopeSettings, Settings, StorageSettings
from vector_embed.core.store.lance import LanceStore
from vector_embed.core.store.sqlite import StateDb


@pytest.fixture
def scope_settings() -> ScopeSettings:
    """Defaults, minus ``appdata``: pytest's tmp dirs live under AppData on Windows."""
    default = ScopeSettings()
    return ScopeSettings(blocked_dirs=default.blocked_dirs - {"appdata"})


@pytest.fixture
def policy(scope_settings: ScopeSettings) -> ScopePolicy:
    return ScopePolicy(scope_settings)


@pytest.fixture
def make(tmp_path: Path) -> Callable[..., str]:
    """Create a non-empty text file under tmp_path and return its path."""

    def _make(rel: str, content: str = "x = 1\n") -> str:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    return _make


@dataclass
class Env:
    """A complete indexing stack on a temp directory with a fake embedder."""

    root: Path
    data_dir: Path
    settings: Settings
    state: StateDb
    store: LanceStore
    embedder: FakeEmbedder
    scope: ScopePolicy
    projects: Projects
    extractors: ExtractorSet
    classifier: DocTypeClassifierSet
    indexer: Indexer


@pytest.fixture
def env(tmp_path: Path, scope_settings: ScopeSettings) -> Iterator[Env]:
    root = tmp_path / "work"
    root.mkdir()
    data_dir = tmp_path / "data"
    settings = Settings(scope=scope_settings, storage=StorageSettings(data_dir=data_dir))
    scope = ScopePolicy(scope_settings, blocked_roots=[data_dir])
    state = StateDb(data_dir)
    store = LanceStore(data_dir, state, "fake-model", dim=DIM)
    embedder = FakeEmbedder()
    projects = Projects(scope, scope_settings, roots=[root])
    extractors = ExtractorSet(ExtractContext(scope, scope_settings, chunking=settings.chunking))
    classifier = DocTypeClassifierSet(settings.doctypes, scope_settings)
    indexer = Indexer(
        settings,
        state,
        store,
        embedder,
        extractors=extractors,
        projects=projects,
        scope=scope,
        classifier=classifier,
    )
    yield Env(
        root, data_dir, settings, state, store, embedder, scope, projects, extractors,
        classifier, indexer,
    )  # fmt: skip
    state.close()
