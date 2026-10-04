"""The always-on watcher must stay small: importing it may not load the ML or UI libraries."""

import subprocess
import sys

import pytest

HEAVY = (
    "lancedb",
    "pyarrow",
    "openai",
    "ollama",
    "PIL",
    "numpy",
    "imagehash",
    "httpx",
    "PySide6",
    "pymupdf",
    "docx",
    "pptx",
    "tree_sitter_language_pack",
    "winrt",
)

PROBE = """
import json, sys
import {module}
print(json.dumps(sorted(m for m in {heavy!r} if m in sys.modules)))
"""


def loaded_after_importing(module: str) -> list[str]:
    code = PROBE.format(module=module, heavy=HEAVY)
    result = subprocess.run(  # our own interpreter, fixed code
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    import json

    loaded: list[str] = json.loads(result.stdout)
    return loaded


def test_the_watcher_does_not_load_heavy_libraries() -> None:
    assert loaded_after_importing("localdoc_finder.watcher") == []


@pytest.mark.parametrize("module", ["localdoc_finder.core.wiring", "localdoc_finder.core.idle"])
def test_the_light_helpers_stay_light(module: str) -> None:
    assert loaded_after_importing(module) == []
