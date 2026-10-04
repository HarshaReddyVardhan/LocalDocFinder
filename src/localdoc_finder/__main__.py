"""``python -m localdoc_finder <app|watcher|worker|setup> [args]``: every entry point.

Entry points are imported lazily so the watcher never loads Qt, and the app never loads more
than it needs. With no entry named, the search app starts.
"""

import importlib
import sys
from collections.abc import Callable, Sequence

from localdoc_finder.core.process import is_frozen

DEFAULT_ENTRY = "app"
# entry -> (module with a ``main(argv)`` function, arguments always passed to it)
_ENTRIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "app": ("localdoc_finder.app.main", ()),
    "watcher": ("localdoc_finder.watcher", ()),
    "worker": ("localdoc_finder.worker", ()),
    "setup": ("localdoc_finder.app.main", ("--setup",)),  # the windowed exe has no console
}


def _run_velopack_hooks() -> None:
    from localdoc_finder.core.lifecycle import run_startup_hooks
    from localdoc_finder.core.settings import SettingsError, load_settings

    def autostart_enabled() -> bool:
        try:
            return load_settings().app.start_with_windows
        except SettingsError:
            return True

    run_startup_hooks(enabled=autostart_enabled)


def resolve(argv: Sequence[str]) -> tuple[Callable[[Sequence[str]], int], list[str]]:
    """The entry point's ``main`` and the arguments to give it."""
    tokens = list(argv)
    entry = tokens[0] if tokens and tokens[0] in _ENTRIES else DEFAULT_ENTRY
    rest = tokens[1:] if tokens and tokens[0] in _ENTRIES else tokens
    module_name, fixed = _ENTRIES[entry]
    main: Callable[[Sequence[str]], int] = importlib.import_module(module_name).main
    return main, [*fixed, *rest]


def main(argv: Sequence[str] | None = None) -> int:
    if is_frozen():
        _run_velopack_hooks()  # first, before anything else: Velopack calls us with install flags
    handler, args = resolve(sys.argv[1:] if argv is None else argv)
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
