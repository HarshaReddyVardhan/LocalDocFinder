from collections.abc import Callable
from pathlib import Path

import pytest

from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import ScopeSettings


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
