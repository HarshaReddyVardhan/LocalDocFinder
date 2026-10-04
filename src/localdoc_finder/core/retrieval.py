"""Hybrid candidate retrieval shared by Search, Ask, Chat and Match: vector + BM25.

The legs are fused by relevance, not by rank: a chunk scores ``vector_weight`` x its similarity
above the embedder's floor (scaled to 0..1) plus ``keyword_weight`` x its BM25 score relative to
the query's best. Rank fusion always put *something* first, however far away; with scores, a page
of bad matches stays low.
"""

import logging
import re
from dataclasses import dataclass

import numpy as np

from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.settings import SearchSettings
from localdoc_finder.core.skills.base import QueryEmbedder
from localdoc_finder.core.store.lance import CHUNKS, LanceStore, Row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """A chunk found by either leg, with what each leg said about it.

    ``similarity`` is the cosine similarity of the vector leg and ``bm25`` the keyword leg's raw
    score; ``None`` (and a rank of ``None``) means that leg did not return the chunk.
    """

    row: Row
    score: float
    similarity: float | None = None
    bm25: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None


@dataclass
class _Hits:
    row: Row
    similarity: float | None = None
    bm25: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None


def fts_terms(text: str) -> str:
    return " ".join(re.findall(r"\w+", text))


def hybrid_candidates(
    store: LanceStore,
    embedder: QueryEmbedder,
    power: PowerGate,
    cfg: SearchSettings,
    *,
    table: str,
    text: str,
    columns: list[str],
    where: str = "",
    limit: int | None = None,
    force_cpu: bool = False,
    unique_key: tuple[str, str] = ("path", "chunk_hash"),
    query_vector: np.ndarray | None = None,
    min_similarity: float = 0.0,
) -> list[Candidate]:
    """Vector and keyword hits for ``text`` in ``table``, best first.

    ``min_similarity`` is the embedder's floor (``SkillContext.similarity_floor``); similarity
    at or below it adds nothing.

    ``force_cpu`` embeds the query on the CPU (used while a chat model owns the GPU). A caller
    that already embedded the query passes ``query_vector``. If the model server is unreachable
    only the keyword leg contributes.
    """
    n = limit or cfg.candidates
    hits: dict[tuple[str, str], _Hits] = {}

    def hit(row: Row) -> _Hits:
        key = (row[unique_key[0]], row[unique_key[1]])
        return hits.setdefault(key, _Hits(row))

    try:
        if query_vector is not None:
            vector = query_vector
        else:
            cpu = force_cpu or power.search_on_cpu()
            vector = embedder.embed([text], kind="query", cpu=cpu)[0]
        floor = cfg.min_content_chars if table == CHUNKS else 0
        rows = store.vector_search(table, vector, columns, where, n, min_content_chars=floor)
        for rank, row in enumerate(rows, 1):
            found = hit(row)
            found.similarity, found.vector_rank = 1.0 - float(row["_distance"]), rank
    except ProviderError:
        logger.info("retrieval: embedding unavailable, using keyword search only")
    terms = fts_terms(text)
    if terms:
        for rank, row in enumerate(store.fts_search(table, terms, columns, where, n), 1):
            found = hit(row)
            found.bm25, found.keyword_rank = float(row["_score"]), rank
    top_bm25 = max((found.bm25 or 0.0 for found in hits.values()), default=0.0)
    candidates = [_fuse(found, cfg, min_similarity, top_bm25) for found in hits.values()]
    # Ties (all below the floor, no keyword) keep the nearest first.
    return sorted(candidates, key=lambda c: (c.score, c.similarity or -1.0), reverse=True)


def scaled_similarity(similarity: float | None, floor: float) -> float:
    """``similarity`` mapped from ``floor``..1 onto 0..1; at or under the floor it is 0."""
    if similarity is None or floor >= 1.0:
        return 0.0
    return min(1.0, max(0.0, (similarity - floor) / (1.0 - floor)))


def _fuse(found: _Hits, cfg: SearchSettings, floor: float, top_bm25: float) -> Candidate:
    keyword = (found.bm25 or 0.0) / top_bm25 if top_bm25 > 0 else 0.0
    score = cfg.vector_weight * scaled_similarity(found.similarity, floor)
    score += cfg.keyword_weight * keyword
    return Candidate(
        found.row, score, found.similarity, found.bm25, found.vector_rank, found.keyword_rank
    )
