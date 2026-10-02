"""Hybrid candidate retrieval shared by Search, Ask and Match: vector + BM25 fused with RRF."""

import logging
import re
from dataclasses import dataclass

import numpy as np

from vector_embed.core.power import PowerGate
from vector_embed.core.providers.base import ProviderError
from vector_embed.core.settings import SearchSettings
from vector_embed.core.skills.base import QueryEmbedder
from vector_embed.core.store.lance import LanceStore, Row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    row: Row
    score: float


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
    by_key: dict[tuple[str, str], Row] = {}
    scores: dict[tuple[str, str], float] = {}

    def add(rows: list[Row]) -> None:
        for rank, row in enumerate(rows):
            key = (row[unique_key[0]], row[unique_key[1]])
            by_key.setdefault(key, row)
            scores[key] = scores.get(key, 0.0) + 1.0 / (cfg.rrf_k + rank + 1)

    try:
        if query_vector is not None:
            vector = query_vector
        else:
            cpu = force_cpu or power.search_on_cpu()
            vector = embedder.embed([text], kind="query", cpu=cpu)[0]
        add(store.vector_search(table, vector, columns, where, n))
    except ProviderError:
        logger.info("retrieval: embedding unavailable, using keyword search only")
    terms = fts_terms(text)
    if terms:
        add(store.fts_search(table, terms, columns, where, n))
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [Candidate(by_key[key], score) for key, score in ranked]
