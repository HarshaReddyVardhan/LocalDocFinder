# PyInstaller spec. One folder, dist/VectorEmbed/, holding two programs that share one runtime:
#   VectorEmbed.exe  windowed: the tray app, the watcher and the worker (python -m vector_embed ...)
#   ve.exe           console:  the command line (ve search, ve doctor, ve setup, ...)
# Build with scripts/build.ps1; run `pyinstaller packaging/vector_embed.spec` from the repo root.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821  (SPECPATH is provided by PyInstaller)
PACKAGING = ROOT / "packaging"
ICON = PACKAGING / "vector_embed.ico"

# Packages with native libraries or data files that the import scanner cannot see.
COLLECT_ALL = [
    "tree_sitter_language_pack",
    "lancedb",
    "pyarrow",
    "pymupdf",
]

hiddenimports = collect_submodules("vector_embed")  # entry points are imported by name
hiddenimports += collect_submodules("winrt")  # Windows OCR projections
hiddenimports += ["keyring.backends.Windows"]  # Credential Manager backend, loaded by entry point
datas = collect_data_files("vector_embed")  # models_catalog.toml and the packaged update source
datas += copy_metadata("vector-embed")  # importlib.metadata.version() for the About tab
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
cli = analysis("entry_ve.py")
MERGE((app, "VectorEmbed", "VectorEmbed"), (cli, "ve", "ve"))  # noqa: F821  one shared runtime

icon = str(ICON) if ICON.is_file() else None
app_exe = EXE(  # noqa: F821
    PYZ(app.pure),  # noqa: F821
    app.scripts,
    [],
    exclude_binaries=True,
    name="VectorEmbed",
    console=False,
    icon=icon,
)
cli_exe = EXE(  # noqa: F821
    PYZ(cli.pure),  # noqa: F821
    cli.scripts,
    [],
    exclude_binaries=True,
    name="ve",
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
    name="VectorEmbed",
)
