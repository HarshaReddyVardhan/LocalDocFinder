from pathlib import Path

import localdoc_finder

PACKAGE_ROOT = Path(localdoc_finder.__file__).parent


def test_every_package_directory_has_an_init_file() -> None:
    """A namespace package is skipped by PyInstaller's submodule collection (frozen builds)."""
    missing = [
        str(folder.relative_to(PACKAGE_ROOT))
        for folder in PACKAGE_ROOT.rglob("*")
        if folder.is_dir()
        and folder.name != "__pycache__"
        and any(folder.glob("*.py"))
        and not (folder / "__init__.py").exists()
    ]
    assert missing == []
