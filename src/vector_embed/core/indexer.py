"""Incremental indexing: extract -> chunk-hash diff -> embed only new text -> LanceDB.

Embedding is the expensive step, so rows are keyed by a hash of each chunk's text: an unchanged
chunk (or identical code copied to another project) reuses its stored vector. Each file also
gets one row in the ``documents`` table (classified type, text, mean vector) for matching.
"""

import logging
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import xxhash

from vector_embed.core import hooks
from vector_embed.core.doctypes.base import DocInfo, DocTypeClassifierSet
from vector_embed.core.doctypes.versions import VersionCandidate, group_versions
from vector_embed.core.extractors.base import Chunk, ExtractError, ExtractorSet
from vector_embed.core.extractors.image import remove_thumbnail
from vector_embed.core.privacy.mask import strip_secret_tokens
from vector_embed.core.projects import Projects
from vector_embed.core.providers.base import EmbedKind
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import Settings
from vector_embed.core.store.lance import DOCUMENTS, LanceStore, Row, sql_quote
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
        self.versioned_changed = False  # a resume/JD/... was (re)indexed since the last grouping

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
            # The hash covers what the chunk *says*, not where its file lives: renaming or moving
            # a file keeps every vector (the path stays in the row metadata).
            digest = xxhash.xxh3_128_hexdigest(
                f"{chunk.kind}\0{symbol}\0{chunk.text}".encode(errors="replace")
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
        for chunk in chunks:  # a credential pasted into a note must not reach the index
            chunk.text = strip_secret_tokens(chunk.text)
        prepared.metas = self._metas(prepared, chunks)
        prepared.doc_text = "\n".join(c.text for c in chunks if c.kind != "outline")
        return prepared

    # ------------------------------------------------------------------ batches
    def process(self, items: Sequence[QueueItem], force: bool = False) -> list[tuple[str, int]]:
        """Process claimed queue items; returns the finished ``(path, seq)`` pairs."""
        try:
            prepared, deletes, finished = self._prepare_all(items, force)
        finally:
            self.extractors.release_models()  # the vision model leaves before the embedder loads
        vectors = self._vectors_for([m for p in prepared for m in p.metas])
        rows, doc_rows = self._build_rows(prepared, vectors)
        self.stats.chunks += len(rows)

        touched = [p.path for p in prepared] + [path for path, _ in deletes]
        self.store.replace_rows(touched, rows)
        self.store.replace_documents(touched, doc_rows)
        for item in prepared:
            self._drop_thumbnail(item.path)  # the old version's thumbnail is now orphaned
            self.state.manifest_set(item.path, item.mtime_ns, item.size, item.content_hash)
            finished.append((item.path, item.seq))
            self.stats.files += 1
        for path, seq in deletes:
            self._drop_thumbnail(path)
            self.state.manifest_delete(path)
            finished.append((path, seq))
            self.stats.deleted += 1
        self._announce(prepared, rows, doc_rows)
        return finished

    @staticmethod
    def _announce(prepared: list[_Prepared], rows: list[Row], doc_rows: list[Row]) -> None:
        """Tell hook subscribers what is now in the index (after it was written)."""
        chunks = Counter(str(row["path"]) for row in rows)
        for item in prepared:
            hooks.emit(hooks.FileIndexed(item.path, chunks[item.path]))
        for doc in doc_rows:
            hooks.emit(
                hooks.DocumentClassified(str(doc["path"]), str(doc["doc_type"]), str(doc["title"]))
            )

    def _drop_thumbnail(self, path: str) -> None:
        """Remove the stored thumbnail of ``path``'s last indexed version, if it was an image."""
        ctx = self.extractors.ctx
        if Path(path).suffix.lower() not in ctx.scope_settings.image_exts:
            return
        entry = self.state.manifest_get(path)
        if entry is not None:
            remove_thumbnail(ctx.thumbs_dir, path, entry.mtime_ns, entry.size)

    def _prepare_all(
        self, items: Sequence[QueueItem], force: bool
    ) -> tuple[list[_Prepared], list[tuple[str, int]], list[tuple[str, int]]]:
        prepared: list[_Prepared] = []
        deletes: list[tuple[str, int]] = []
        finished: list[tuple[str, int]] = []
        for item in items:
            if self.stop_check():
                break  # remaining items stay queued
            path = self.state.canonical_path(item.path)  # one spelling per file in the index
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
            except FileNotFoundError:
                deletes.append((path, item.seq))
            except PermissionError:
                # A sharing violation (another program has the file open) is not a deletion:
                # keep the index rows and try again later.
                logger.info("locked or unreadable for now: %s", path)
                self.stats.errors += 1
                self.state.fail(path)
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
        if kind.doc_type in self.settings.doctypes.versioned_types:
            self.versioned_changed = True  # version groups must be recomputed after this run
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
        """Group near-identical versions of resumes, JDs, ... and store the group ids.

        Only documents of the versioned types are read, and only when this run indexed one:
        grouping everything after every run was the dominant cost of an otherwise tiny update.
        """
        if not self.versioned_changed:
            return 0
        cfg = self.settings.doctypes
        listed = ",".join(sql_quote(t) for t in sorted(cfg.versioned_types))
        rows = self.store.scan(
            DOCUMENTS,
            ["path", "doc_type", "full_text", "modified_at"],
            f"doc_type IN ({listed}) AND full_text != ''",
            limit=_DOC_SCAN_LIMIT,
        )
        candidates = [
            VersionCandidate(r["path"], r["doc_type"], r["full_text"], r["modified_at"])
            for r in rows
        ]
        groups = group_versions(candidates, cfg.versioned_types, cfg.version_similarity)
        if self.store.documents is not None:
            self.store.set_version_groups(groups, [c.path for c in candidates])
        self.versioned_changed = False
        return len(groups)
