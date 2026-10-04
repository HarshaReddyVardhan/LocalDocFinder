"""Generic name -> implementation registry behind every ``@register`` extension point.

Extractors, doctypes, providers, skills and sources each own one ``Registry``. Adding a new
one is a single file with a decorator; no core code is edited.
"""

import importlib
import pkgutil
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Unknown name, or a duplicate registration."""


class Registry(Generic[T]):  # plain Generic keeps 3.11 compatibility
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, *, replace: bool = False) -> Callable[[T], T]:
        """Decorator: ``@registry.register("name")`` on a class or factory."""

        def decorator(item: T) -> T:
            self.add(name, item, replace=replace)
            return item

        return decorator

    def add(self, name: str, item: T, *, replace: bool = False) -> None:
        if name in self._items and not replace:
            raise RegistryError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "none"
            raise RegistryError(f"unknown {self.kind} {name!r} (known: {known})") from None

    def remove(self, name: str) -> None:
        self._items.pop(name, None)

    def names(self) -> list[str]:
        return list(self._items)

    def items(self) -> list[tuple[str, T]]:
        return list(self._items.items())

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


def discover_modules(package: str) -> list[str]:
    """Import every module of ``package`` so its ``@register`` decorators run.

    Dropping a new file into an extension package is all it takes to add a feature.
    Returns the imported module names.
    """
    pkg = importlib.import_module(package)
    imported: list[str] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        name = f"{package}.{info.name}"
        importlib.import_module(name)
        imported.append(name)
    return imported
