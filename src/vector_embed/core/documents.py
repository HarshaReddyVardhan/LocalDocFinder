"""Load whole documents for Chat and Match: from the index when stored there, else re-extract."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vector_embed.core.extractors.base import ExtractError, ExtractorSet
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.store.lance import DOCUMENTS, LanceStore, sql_quote

logger = logging.getLogger(__name__)


class DocumentError(RuntimeError):
    """A document cannot be loaded (secret, missing or unreadable)."""


@dataclass(frozen=True)
class LoadedDocument:
    path: str
    title: str
    text: str
    doc_type: str
    from_index: bool


class DocumentLoader:
    """Whole-document text. Secrets are refused here so no caller can send them to a model."""

    def __init__(
        self,
        store: LanceStore,
        scope: ScopePolicy,
        extractors: Callable[[], ExtractorSet],
    ) -> None:
        self._store = store
        self._scope = scope
        self._extractors = extractors

    def load(self, path: str | Path) -> LoadedDocument:
        target = Path(path)
        if self._scope.is_secret(target):
            raise DocumentError(f"{target.name} looks like a secret and is never loaded")
        if not target.is_file():
            raise DocumentError(f"{target} does not exist")
        key = str(target)
        rows = self._store.scan(
            DOCUMENTS,
            ["title", "doc_type", "full_text"],
            f"path = {sql_quote(key)}",
            limit=1,
        )
        if rows and rows[0]["full_text"]:
            row = rows[0]
            return LoadedDocument(key, row["title"], row["full_text"], row["doc_type"], True)
        return self._extract(target)

    def _extract(self, target: Path) -> LoadedDocument:
        try:
            chunks = self._extractors().extract(target)
        except ExtractError as exc:
            raise DocumentError(f"cannot read {target.name}: {exc}") from exc
        text = "\n".join(c.text for c in chunks if c.kind != "outline")
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return LoadedDocument(str(target), first[:120] or target.name, text, "", False)
