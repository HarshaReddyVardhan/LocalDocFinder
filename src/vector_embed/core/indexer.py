"""Incremental indexing: extract -> chunk-hash diff -> embed only new text -> LanceDB.

Embedding is the expensive step, so rows are keyed by a hash of each chunk's text: an unchanged
chunk (or identical code copied to another project) reuses its stored vector. Each file also
gets one row in the ``documents`` table (classified type, text, mean vector) for matching.
"""

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import xxhash

from vector_embed.core.doctypes.base import DocInfo, DocTypeClassifierSet
from vector_embed.core.doctypes.versions import VersionCandidate, group_versions
from vector_embed.core.extractors.base import Chunk, ExtractError, ExtractorSet
from vector_embed.core.projects import Projects
from vector_embed.core.providers.base import EmbedKind
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import Settings
from vector_embed.core.store.lance import DOCUMENTS, LanceStore, Row
from vector_embed.core.store.sqlite import QueueItem, StateDb

logger = logging.getLogger(__name__)

_HASH_BLOCK = 1 << 20
_TITLE_CHARS = 120
_DOC_SCAN_LIMIT = 100_000


class BatchEmbedder(Protocol):
    def embed(
        self,
        texts: list[str],
        kind: EmbedKind = "doc",
        cpu: bool = False,
        stop_check: Callable[[], bool] | None = None,
    ) -> np.ndarray: ...


def file_hash(path: str | Path) -> str:
    digest = xxhash.xxh3_128()
    with Path(path).open("rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class IndexStats:
    files: int = 0
    skipped: int = 0
    deleted: int = 0
    errors: int = 0
    chunks: int = 0
    embedded: int = 0
    reused: int = 0


@dataclass
class _Prepared:
    path: str
    seq: int
    mtime_ns: int
    mtime: int
    size: int
    content_hash: str
    project: str
    metas: list[dict[str, Any]] = field(default_factory=list)  # chunk rows minus the vector
    doc_text: str = ""


class Indexer:
    """Turns queue items into LanceDB rows. Raises ``Interrupted`` if ``stop_check`` fires."""

    def __init__(
        self,
        settings: Settings,
        state: StateDb,
        store: LanceStore,
        embedder: BatchEmbedder,
        *,
        extractors: ExtractorSet,
        projects: Projects,
        scope: ScopePolicy,
        classifier: DocTypeClassifierSet,
        stop_check: Callable[[], bool] = lambda: False,
    ) -> None:
        self.settings = settings
        self.state = state
        self.store = store
        self.embedder = embedder
        self.extractors = extractors
        self.projects = projects
        self.scope = scope
        self.classifier = classifier
        self.stop_check = stop_check
        self.stats = IndexStats()

    # ------------------------------------------------------------------ row building
    def _display(self, path: str) -> tuple[str, str]:
        """``(project name, path relative to the project root, or the file name)``."""
        root = self.projects.project_root(path)
        if root is None:
            return "", Path(path).name
        return root.name, os.path.relpath(path, root).replace("\\", "/")

    def _metas(self, prepared: _Prepared, chunks: list[Chunk]) -> list[dict[str, Any]]:
        path = prepared.path
        project, rel = self._display(path)
        source = self.scope.ai_note_source(path)
        ext = Path(path).suffix.lower()
        cfg = self.settings.chunking
        metas: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in chunks:
            symbol = chunk.symbol
            # Embedded text omits project and drive so identical code copied between projects
            # hashes the same and is embedded once.
            embed_text = f"{rel} > {symbol}\n{chunk.text}" if symbol else f"{rel}\n{chunk.text}"
            digest = xxhash.xxh3_128_hexdigest(
                f"{chunk.kind}\0{embed_text}".encode(errors="replace")
            )
            if digest in seen:
                continue
            seen.add(digest)
            head = f"{project} > {path}" + (f" > {symbol}" if symbol else "")
            metas.append(
                {
                    "embed_text": embed_text,
                    "text": (head + "\n" + chunk.text)[: cfg.stored_text_chars],
                    "path": path,
                    "project": project,
                    "kind": "ai-note" if source and chunk.kind == "doc" else chunk.kind,
                    "source": str(source) if source else "",
                    "ext": ext,
                    "symbol": symbol,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "page": chunk.page,
                    "chunk_hash": digest,
                    "model_id": self.store.model_id,
                    "mtime": prepared.mtime,
                }
            )
        return metas

    # ------------------------------------------------------------------ per file
    def prepare(self, path: str, seq: int = 0, force: bool = False) -> _Prepared | None:
        """A prepared file, or ``None`` when nothing needs to change (or it was recorded as bad)."""
        info = Path(path).stat()
        old = self.state.manifest_get(path)
        if old and not force and (old.mtime_ns, old.size) == (info.st_mtime_ns, info.st_size):
            self.stats.skipped += 1
            return None
        digest = file_hash(path)
        if old and not force and old.content_hash == digest:  # touched but unchanged
            self.state.manifest_set(path, info.st_mtime_ns, info.st_size, digest)
            self.stats.skipped += 1
            return None
        prepared = _Prepared(
            path=path,
            seq=seq,
            mtime_ns=info.st_mtime_ns,
            mtime=int(info.st_mtime),
            size=info.st_size,
            content_hash=digest,
            project=self._display(path)[0],
        )
        try:
            chunks = self.extractors.extract(path)
        except ExtractError as exc:
            logger.info("skip %s: %s", path, exc)
            self.store.delete_paths([path])
            self.state.manifest_set(path, info.st_mtime_ns, info.st_size, digest)
            self.stats.errors += 1
            return None
        prepared.metas = self._metas(prepared, chunks)
        prepared.doc_text = "\n".join(c.text for c in chunks if c.kind != "outline")
        return prepared

    # ------------------------------------------------------------------ batches
    def process(self, items: Sequence[QueueItem], force: bool = False) -> list[tuple[str, int]]:
        """Process claimed queue items; returns the finished ``(path, seq)`` pairs."""
        prepared, deletes, finished = self._prepare_all(items, force)
        vectors = self._vectors_for([m for p in prepared for m in p.metas])
        rows, doc_rows = self._build_rows(prepared, vectors)
        self.stats.chunks += len(rows)

        touched = [p.path for p in prepared] + [path for path, _ in deletes]
        self.store.replace_rows(touched, rows)
        self.store.replace_documents(touched, doc_rows)
        for item in prepared:
            self.state.manifest_set(item.path, item.mtime_ns, item.size, item.content_hash)
            finished.append((item.path, item.seq))
            self.stats.files += 1
        for path, seq in deletes:
            self.state.manifest_delete(path)
            finished.append((path, seq))
            self.stats.deleted += 1
        return finished

    def _prepare_all(
        self, items: Sequence[QueueItem], force: bool
    ) -> tuple[list[_Prepared], list[tuple[str, int]], list[tuple[str, int]]]:
        prepared: list[_Prepared] = []
        deletes: list[tuple[str, int]] = []
        finished: list[tuple[str, int]] = []
        for item in items:
            if self.stop_check():
                break  # remaining items stay queued
            path = item.path
            try:
                if (
                    item.op == "delete"
                    or not Path(path).exists()
                    or not self.scope.is_valid_file(path, is_ignored=self.projects.is_ignored)
                ):
                    deletes.append((path, item.seq))
                    continue
                result = self.prepare(path, item.seq, force)
                if result is None:
                    finished.append((path, item.seq))
                else:
                    prepared.append(result)
            except (PermissionError, FileNotFoundError):
                deletes.append((path, item.seq))
            except Exception:  # one bad file must not stop the batch; it is retried later
                logger.warning("failed %s", path, exc_info=True)
                self.stats.errors += 1
                self.state.fail(path)
        return prepared, deletes, finished

    def _vectors_for(self, metas: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """Stored vectors for known chunk hashes; embeds only text never embedded before."""
        vectors = self.store.vectors_for_hashes(m["chunk_hash"] for m in metas)
        todo: dict[str, str] = {}
        for meta in metas:
            if meta["chunk_hash"] not in vectors and meta["chunk_hash"] not in todo:
                todo[meta["chunk_hash"]] = meta["embed_text"]
        self.stats.reused += len(metas) - len(todo)
        if todo:
            keys = list(todo)
            embedded = self.embedder.embed(
                [todo[k] for k in keys], kind="doc", stop_check=self.stop_check
            )
            for key, vector in zip(keys, embedded, strict=True):
                vectors[key] = vector
            self.stats.embedded += len(keys)
        return vectors

    def _build_rows(
        self, prepared: list[_Prepared], vectors: dict[str, np.ndarray]
    ) -> tuple[list[Row], list[Row]]:
        rows: list[Row] = []
        doc_rows: list[Row] = []
        for item in prepared:
            file_vectors: list[np.ndarray] = []
            for meta in item.metas:
                row = {k: v for k, v in meta.items() if k != "embed_text"}
                row["vector"] = vectors[meta["chunk_hash"]]
                rows.append(row)
                file_vectors.append(vectors[meta["chunk_hash"]])
            if file_vectors:
                doc_rows.append(self._document_row(item, file_vectors))
        return rows, doc_rows

    def _document_row(self, item: _Prepared, file_vectors: list[np.ndarray]) -> Row:
        mean = np.mean(file_vectors, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        doc_vector = mean / norm if norm else mean
        text = item.doc_text
        path = Path(item.path)
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        title = first_line[:_TITLE_CHARS] or path.name
        kind = self.classifier.classify(DocInfo(path, title, text, doc_vector))
        keep = len(text) <= self.settings.doctypes.full_text_max_chars
        return {
            "doc_vector": doc_vector,
            "path": item.path,
            "project": item.project,
            "doc_type": kind.doc_type,
            "title": title,
            "full_text": text if keep else "",
            "version_group": "",
            "modified_at": item.mtime,
            "model_id": self.store.model_id,
        }

    def index_paths(
        self, paths: Sequence[str], force: bool = False, batch: int | None = None
    ) -> None:
        """Index files directly, bypassing the queue (used by tests and the eval harness)."""
        size = batch or self.settings.chunking.worker_batch_files
        for start in range(0, len(paths), size):
            items = [QueueItem(p, "upsert", 0) for p in paths[start : start + size]]
            self.process(items, force=force)

    # ------------------------------------------------------------------ version groups
    def assign_version_groups(self) -> int:
        """Group near-identical versions of resumes, JDs, ... and store the group ids."""
        cfg = self.settings.doctypes
        rows = self.store.scan(
            DOCUMENTS,
            ["path", "doc_type", "full_text", "modified_at"],
            limit=_DOC_SCAN_LIMIT,
        )
        candidates = [
            VersionCandidate(r["path"], r["doc_type"], r["full_text"], r["modified_at"])
            for r in rows
            if r["doc_type"] in cfg.versioned_types and r["full_text"]
        ]
        groups = group_versions(candidates, cfg.versioned_types, cfg.version_similarity)
        if self.store.documents is not None:
            self.store.set_version_groups(groups, [c.path for c in candidates])
        return len(groups)
