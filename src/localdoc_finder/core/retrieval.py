"""Hybrid candidate retrieval shared by Search, Ask and Match: vector + BM25 fused with RRF."""

import logging
import re
from dataclasses import dataclass

import numpy as np

from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.settings import SearchSettings
from localdoc_finder.core.skills.base import QueryEmbedder
from localdoc_finder.core.store.lance import LanceStore, Row

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
) -> list[Candidate]:
    """Vector and keyword hits for ``text`` in ``table``, fused with reciprocal-rank fusion.

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
        for rank, row in enumerate(store.vector_search(table, vector, columns, where, n), 1):
            found = hit(row)
            found.similarity, found.vector_rank = 1.0 - float(row["_distance"]), rank
    except ProviderError:
        logger.info("retrieval: embedding unavailable, using keyword search only")
    terms = fts_terms(text)
    if terms:
        for rank, row in enumerate(store.fts_search(table, terms, columns, where, n), 1):
            found = hit(row)
            found.bm25, found.keyword_rank = float(row["_score"]), rank
    candidates = [_fuse(found, cfg) for found in hits.values()]
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def _fuse(found: _Hits, cfg: SearchSettings) -> Candidate:
    """Reciprocal-rank fusion: each leg adds ``1 / (k + rank)``."""
    score = sum(
        1.0 / (cfg.rrf_k + rank) for rank in (found.vector_rank, found.keyword_rank) if rank
    )
    return Candidate(
        found.row, score, found.similarity, found.bm25, found.vector_rank, found.keyword_rank
    )
