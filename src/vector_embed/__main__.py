"""``python -m vector_embed <app|watcher|worker|setup> [args]``; also the frozen exe's dispatcher.

Entry points are imported lazily so the watcher never loads Qt, and the app never loads more
than it needs. With no entry named, the search app starts.
"""

import importlib
import sys
from collections.abc import Callable, Sequence

DEFAULT_ENTRY = "app"
# entry -> (module with a ``main(argv)`` function, arguments always passed to it)
_ENTRIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "app": ("vector_embed.app.main", ()),
    "watcher": ("vector_embed.watcher", ()),
    "worker": ("vector_embed.worker", ()),
    "setup": ("vector_embed.app.main", ("--setup",)),  # the windowed exe has no console
}


def resolve(argv: Sequence[str]) -> tuple[Callable[[Sequence[str]], int], list[str]]:
    """The entry point's ``main`` and the arguments to give it."""
    tokens = list(argv)
    entry = tokens[0] if tokens and tokens[0] in _ENTRIES else DEFAULT_ENTRY
    rest = tokens[1:] if tokens and tokens[0] in _ENTRIES else tokens
    module_name, fixed = _ENTRIES[entry]
    main: Callable[[Sequence[str]], int] = importlib.import_module(module_name).main
    return main, [*fixed, *rest]


def main(argv: Sequence[str] | None = None) -> int:
    handler, args = resolve(sys.argv[1:] if argv is None else argv)
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
