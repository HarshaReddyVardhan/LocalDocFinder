import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tmp_path():
    """Work dir outside %TEMP% (which sits under the blocked AppData directory)."""
    root = Path(__file__).resolve().parent / "_work"
    root.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=root))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def make(tmp_path):
    """Create a non-empty text file under tmp_path and return its path."""
    def _make(rel: str, content: str = "x = 1\n") -> str:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return str(p)
    return _make
