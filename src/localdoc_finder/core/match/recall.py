"""Step 1 of Match: recall candidate documents for a pasted text, always locally.

Only the local embedding model sees the text here. Candidates come from the ``documents``
table (hybrid vector + keyword), grouped by version so the newest version of each document is
shown by default; the user then ticks which ones to score.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from localdoc_finder.core.documents import DocumentError, DocumentLoader
from localdoc_finder.core.retrieval import hybrid_candidates
from localdoc_finder.core.settings import MatchSettings
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.store.lance import DOCUMENTS, sql_quote
from localdoc_finder.core.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_COLUMNS = ["path", "title", "doc_type", "full_text", "version_group", "modified_at", "doc_vector"]
_FANOUT = 3  # recall this many times k, then trim after grouping versions
_WORD = re.compile(r"[A-Za-z0-9+#]{3,}")
_ADDED_FILE_CHARS = 2000


@dataclass
class MatchCandidate:
    path: str
    title: str
    doc_type: str
    modified_at: int
    similarity: float
    tokens: int
    version_group: str = ""
    versions: int = 1  # how many versions of this document exist
    is_latest: bool = True
    selected: bool = False
    locked: bool = False  # never sent to a cloud model; scored locally even in cloud mode

    @property
    def name(self) -> str:
        return Path(self.path).name


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / norm if norm else 0.0


def keyword_query(text: str, limit: int) -> str:
    """Distinct words of ``text`` in order of appearance, capped (job descriptions are long)."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(text):
        seen.setdefault(word.lower(), None)
        if len(seen) >= limit:
            break
    return " ".join(seen)


class Recall:
    """Recall, select and extend the candidate list."""

    def __init__(
        self,
        ctx: SkillContext,
        loader: DocumentLoader,
        is_locked: Callable[[str], bool] = lambda _path: False,
    ) -> None:
        self._ctx = ctx
        self._loader = loader
        self._is_locked = is_locked
        self._settings: MatchSettings = ctx.settings.match

    def recall(
        self, text: str, doc_type: str | None = None, include_old_versions: bool = False
    ) -> list[MatchCandidate]:
        """Candidates for ``text``, best first. ``doc_type=None`` searches every document."""
        ctx, cfg = self._ctx, self._settings
        doc_type = cfg.default_doc_type if doc_type is None else doc_type
        vector = ctx.embedder.embed([text[:8000]], kind="query", cpu=True)[0]
        found = hybrid_candidates(
            ctx.store,
            ctx.embedder,
            ctx.power,
            ctx.settings.search,
            table=DOCUMENTS,
            text=keyword_query(text, cfg.max_fts_terms),
            columns=_COLUMNS,
            where=f"doc_type = {sql_quote(doc_type)}" if doc_type else "",
            limit=cfg.recall_k * _FANOUT,
            unique_key=("path", "path"),
            query_vector=vector,
        )
        candidates = [self._candidate(c.row, vector) for c in found]
        grouped = self._apply_versions(candidates, include_old_versions)
        grouped.sort(key=lambda c: c.similarity, reverse=True)
        top = grouped[: cfg.recall_k]
        for candidate in top:
            candidate.selected = (
                candidate.is_latest and candidate.similarity >= cfg.similarity_threshold
            )
        return top

    def _candidate(self, row: dict[str, object], vector: np.ndarray) -> MatchCandidate:
        path = str(row["path"])
        text = str(row["full_text"] or "")
        tokens = estimate_tokens(text) if text else _size_tokens(path)
        return MatchCandidate(
            path=path,
            title=str(row["title"]),
            doc_type=str(row["doc_type"]),
            modified_at=int(row["modified_at"]),  # type: ignore[call-overload]
            similarity=_cosine(vector, np.asarray(row["doc_vector"], dtype=np.float32)),
            tokens=tokens,
            version_group=str(row["version_group"] or ""),
            locked=self._is_locked(path),
        )

    def _apply_versions(
        self, candidates: list[MatchCandidate], include_old: bool
    ) -> list[MatchCandidate]:
        newest: dict[str, MatchCandidate] = {}
        for candidate in candidates:
            group = candidate.version_group
            if group and (group not in newest or candidate.modified_at > newest[group].modified_at):
                newest[group] = candidate
        counts = {group: self._group_size(group) for group in newest}
        kept: list[MatchCandidate] = []
        for candidate in candidates:
            group = candidate.version_group
            if group:
                candidate.versions = counts.get(group, 1)
                candidate.is_latest = newest[group] is candidate
                if not candidate.is_latest and not include_old:
                    continue
            kept.append(candidate)
        return kept

    def _group_size(self, group: str) -> int:
        rows = self._ctx.store.scan(
            DOCUMENTS, ["path"], f"version_group = {sql_quote(group)}", limit=1000
        )
        return len(rows)

    # ------------------------------------------------------------------ user edits
    def add_file(self, path: str, jd_vector_text: str) -> MatchCandidate:
        """Include a file recall missed (picked by the user); it starts ticked."""
        document = self._loader.load(path)
        vector = self._ctx.embedder.embed([jd_vector_text[:8000]], kind="query", cpu=True)[0]
        head = document.text[:_ADDED_FILE_CHARS]
        doc_vector = self._ctx.embedder.embed([head or document.title], kind="doc", cpu=True)[0]
        return MatchCandidate(
            path=document.path,
            title=document.title,
            doc_type=document.doc_type,
            modified_at=int(Path(path).stat().st_mtime),
            similarity=_cosine(vector, doc_vector),
            tokens=estimate_tokens(document.text),
            selected=True,
            locked=self._is_locked(document.path),
        )


def _size_tokens(path: str) -> int:
    try:
        return Path(path).stat().st_size // 4
    except OSError:
        return 0


def select_all(candidates: list[MatchCandidate]) -> None:
    for candidate in candidates:
        candidate.selected = True


def select_none(candidates: list[MatchCandidate]) -> None:
    for candidate in candidates:
        candidate.selected = False


def select_top(candidates: list[MatchCandidate], n: int) -> None:
    """Tick the ``n`` most similar latest versions, untick the rest."""
    select_none(candidates)
    latest = sorted(
        (c for c in candidates if c.is_latest), key=lambda c: c.similarity, reverse=True
    )
    for candidate in latest[:n]:
        candidate.selected = True


def selected(candidates: list[MatchCandidate]) -> list[MatchCandidate]:
    return [c for c in candidates if c.selected]


__all__ = [
    "DocumentError",
    "MatchCandidate",
    "Recall",
    "keyword_query",
    "select_all",
    "select_none",
    "select_top",
    "selected",
]
