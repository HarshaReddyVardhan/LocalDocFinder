"""Event bus: lets skills and extensions react to what the core does, without editing it.

Four events, as in the plan: a file was indexed, a document was classified, a query ran, and an
answer was given (e.g. "auto-match new JDs in Downloads" subscribes to ``DocumentClassified``).

Handlers are registered per process with ``@on(EventType)`` (or ``BUS.subscribe``) and run
synchronously where the event is emitted. A failing handler is logged and never breaks the
indexer or the skill that emitted the event. Indexing events fire in the worker process.
"""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileIndexed:
    """``on_file_indexed``: a file's chunks are now in the index."""

    path: str
    chunks: int


@dataclass(frozen=True)
class DocumentClassified:
    """``on_document_classified``: a document was (re)indexed and tagged with a type."""

    path: str
    doc_type: str
    title: str


@dataclass(frozen=True)
class QueryRan:
    """``on_query``: a search ran (``results`` = files returned)."""

    query: str
    results: int


@dataclass(frozen=True)
class Answered:
    """``on_answer``: Ask or Chat finished a reply."""

    skill: str
    question: str
    answer: str
    sources: tuple[str, ...] = ()


Event = FileIndexed | DocumentClassified | QueryRan | Answered
E = TypeVar("E", FileIndexed, DocumentClassified, QueryRan, Answered)


class HookBus:
    """Subscribers per event type. Thread-safe: the app emits from worker threads."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable[[object], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> Callable[[], None]:
        """Call ``handler`` for every ``event_type`` event; returns a function that unsubscribes."""
        untyped: Callable[[object], None] = handler  # type: ignore[assignment]  # keyed by type
        with self._lock:
            self._handlers.setdefault(event_type, []).append(untyped)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(event_type, [])
                if untyped in handlers:
                    handlers.remove(untyped)

        return unsubscribe

    def emit(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # a hook must never break the core path that emitted it
                logger.exception("hook failed", extra={"event": type(event).__name__})

    @contextmanager
    def subscribed(self, event_type: type[E], handler: Callable[[E], None]) -> Iterator[None]:
        """Subscribe for the duration of a block (tests, short-lived listeners)."""
        unsubscribe = self.subscribe(event_type, handler)
        try:
            yield
        finally:
            unsubscribe()


BUS = HookBus()  # the process-wide bus every emitter uses


def on(event_type: type[E]) -> Callable[[Callable[[E], None]], Callable[[E], None]]:
    """Decorator: ``@on(DocumentClassified)`` subscribes a function to the process-wide bus."""

    def decorator(handler: Callable[[E], None]) -> Callable[[E], None]:
        BUS.subscribe(event_type, handler)
        return handler

    return decorator


def emit(event: Event) -> None:
    BUS.emit(event)
