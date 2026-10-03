from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
from tests.core.conftest import Env

from vector_embed.core.projects import Projects
from vector_embed.core.reconcile import reconcile
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.sources.base import SOURCES, content_sources
from vector_embed.core.sources.filesystem import FilesystemSource


def write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_filesystem_is_a_registered_source(env: Env) -> None:
    sources = content_sources(env.projects, env.scope)
    assert "filesystem" in SOURCES
    assert any(isinstance(s, FilesystemSource) for s in sources)


class _ExtraSource:
    """A second source over its own folder, as a browser-history or email source would be."""

    name = "extra"
    root: Path

    def __init__(self, _projects: Projects, _scope: ScopePolicy) -> None:
        pass

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.root,)

    def iter_files(self, roots: Iterable[str | Path] | None = None) -> Iterator[Path]:
        yield from sorted(self.root.glob("*.txt"))

    def still_valid(self, path: str) -> bool:
        return Path(path).exists()


@pytest.fixture
def extra(tmp_path: Path) -> Iterator[type[_ExtraSource]]:
    _ExtraSource.root = tmp_path / "extra"
    _ExtraSource.root.mkdir()
    SOURCES.add("extra", _ExtraSource)
    try:
        yield _ExtraSource
    finally:
        SOURCES.remove("extra")


def test_reconcile_scans_every_registered_source(env: Env, extra: type[_ExtraSource]) -> None:
    write(env.root / "a.txt", "alpha " * 10)
    outside = write(extra.root / "b.txt", "bravo " * 10)
    result = reconcile(env.state, env.projects, env.scope)
    queued = {Path(i.path).name for i in env.state.claim(10, ignore_debounce=True)}
    assert queued == {"a.txt", "b.txt"}
    assert result.queued == 2

    env.state.manifest_set(outside, 1, 1, "h")
    Path(outside).unlink()
    gone = reconcile(env.state, env.projects, env.scope)
    assert gone.deleted == 1  # the extra source decided its own file is gone
