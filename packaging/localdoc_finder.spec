# PyInstaller spec. One folder, dist/LocalDocFinder/, holding two programs that share one runtime:
#   LocalDocFinder.exe  windowed: the tray app, the watcher and the worker (python -m localdoc_finder ...)
#   ldf.exe           console:  the command line (ldf search, ldf doctor, ldf setup, ...)
# Build with scripts/build.ps1; run `pyinstaller packaging/localdoc_finder.spec` from the repo root.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821  (SPECPATH is provided by PyInstaller)
PACKAGING = ROOT / "packaging"
ICON = PACKAGING / "localdoc_finder.ico"

# Packages with native libraries or data files that the import scanner cannot see.
COLLECT_ALL = [
    "tree_sitter_language_pack",
    "lancedb",
    "pyarrow",
    "pymupdf",
]

hiddenimports = collect_submodules("localdoc_finder")  # entry points are imported by name
hiddenimports += collect_submodules("winrt")  # Windows OCR projections
hiddenimports += ["keyring.backends.Windows"]  # Credential Manager backend, loaded by entry point
datas = collect_data_files("localdoc_finder")  # models_catalog.toml and the packaged update source
datas += copy_metadata("localdoc-finder")  # importlib.metadata.version() for the About tab
binaries = []
for package in COLLECT_ALL:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Development tooling must not ship.
EXCLUDES = [
    "pytest",
    "pytest_cov",
    "pytest_mock",
    "mypy",
    "ruff",
    "pre_commit",
    "pip_audit",
    "tkinter",
    "IPython",
]


def analysis(script: str) -> Analysis:  # noqa: F821
    return Analysis(  # noqa: F821
        [str(PACKAGING / script)],
        pathex=[str(ROOT / "src")],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        excludes=EXCLUDES,
        noarchive=False,
    )


app = analysis("entry_app.py")
cli = analysis("entry_ldf.py")
MERGE((app, "LocalDocFinder", "LocalDocFinder"), (cli, "ldf", "ldf"))  # noqa: F821  one shared runtime

icon = str(ICON) if ICON.is_file() else None
app_exe = EXE(  # noqa: F821
    PYZ(app.pure),  # noqa: F821
    app.scripts,
    [],
    exclude_binaries=True,
    name="LocalDocFinder",
    console=False,
    icon=icon,
)
cli_exe = EXE(  # noqa: F821
    PYZ(cli.pure),  # noqa: F821
    cli.scripts,
    [],
    exclude_binaries=True,
    name="ldf",
    console=True,
    icon=icon,
)
COLLECT(  # noqa: F821
    app_exe,
    app.binaries,
    app.datas,
    cli_exe,
    cli.binaries,
    cli.datas,
    name="LocalDocFinder",
)
