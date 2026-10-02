"""Incremental indexing worker. Spawned by watcher.py when idle + on AC; exits when the queue is drained.

    python worker.py --now [--path D:\\Projects\\x] [--reconcile] [--allow-battery]

Power: checked before every file and between embedding batches. If the laptop is unplugged the
worker commits what is already done, unloads the model (keep_alive=0) and exits.
"""
import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import xxhash

import indexer_config as cfg
import power
from embedder import EmbedError, Embedder, Interrupted
from extract import Chunk, ExtractError, extract
from projects import Projects
from store import Store, single_instance

log = logging.getLogger("worker")


def file_hash(path: str) -> str:
    h = xxhash.xxh3_128()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Prepared:
    path: str
    seq: int
    mtime_ns: int
    size: int
    content_hash: str
    metas: List[dict] = field(default_factory=list)   # row dicts minus vector


class Pipeline:
    """extract -> chunk hash diff -> embed (only new text) -> LanceDB, per batch of files."""

    def __init__(self, store: Store, embedder: Embedder, projects: Projects = None,
                 stop_check: Callable[[], bool] = None):
        self.store = store
        self.embedder = embedder
        self.projects = projects or Projects()
        self.stop_check = stop_check or (lambda: False)
        self.stats = {"files": 0, "skipped": 0, "deleted": 0, "errors": 0, "chunks": 0, "embedded": 0, "reused": 0}

    # -------------------------------------------------------------- row building
    def _display(self, path: str) -> Tuple[str, str, str]:
        """(project, path relative to project root (or file name), full path)."""
        root = self.projects.project_root(path)
        if root:
            return os.path.basename(root), os.path.relpath(path, root).replace("\\", "/"), path
        return "", os.path.basename(path), path

    def _metas(self, path: str, st: os.stat_result, chunks: List[Chunk]) -> List[dict]:
        project, rel, full = self._display(path)
        ext = Path(path).suffix.lower()
        source = cfg_ai_source(path)
        metas, seen = [], set()
        for c in chunks:
            sym = c.symbol
            # Embedded text deliberately omits project/drive so identical code copied between
            # projects hashes the same and is embedded only once.
            embed_text = f"{rel} › {sym}\n{c.text}" if sym else f"{rel}\n{c.text}"
            digest = xxhash.xxh3_128_hexdigest(f"{c.kind}\0{embed_text}".encode("utf-8", "replace"))
            if digest in seen:
                continue
            seen.add(digest)
            head = f"{project} › {full}" + (f" › {sym}" if sym else "")
            metas.append({
                "embed_text": embed_text,
                "text": (head + "\n" + c.text)[: cfg.STORED_TEXT_CHARS],
                "path": path, "project": project,
                "kind": "ai-note" if source and c.kind == "doc" else c.kind,
                "source": source or "", "ext": ext, "symbol": sym,
                "start_line": c.start_line, "end_line": c.end_line, "page": c.page,
                "chunk_hash": digest, "model_id": self.store.model_id,
                "mtime": int(st.st_mtime),
            })
        return metas

    # ---------------------------------------------------------------- per file
    def prepare(self, path: str, seq: int = 0, force: bool = False) -> Optional[Prepared]:
        """Return a Prepared file, or None if nothing needs to change (or it was handled as an error)."""
        st = os.stat(path)
        old = self.store.manifest_get(path)
        if old and not force and old[0] == st.st_mtime_ns and old[1] == st.st_size:
            self.stats["skipped"] += 1
            return None
        digest = file_hash(path)
        if old and not force and old[2] == digest:  # touched but unchanged: nothing to re-embed
            self.store.manifest_set(path, st.st_mtime_ns, st.st_size, digest)
            self.stats["skipped"] += 1
            return None
        try:
            chunks = extract(path)
        except ExtractError as e:
            log.info("skip %s: %s", path, e)
            self.store.delete_paths([path])
            self.store.manifest_set(path, st.st_mtime_ns, st.st_size, digest)
            self.stats["errors"] += 1
            return None
        return Prepared(path, seq, st.st_mtime_ns, st.st_size, digest, self._metas(path, st, chunks))

    # ----------------------------------------------------------------- batches
    def process(self, items: List[Tuple[str, str, int]], force: bool = False) -> List[Tuple[str, int]]:
        """Process claimed queue items. Returns the (path, seq) pairs that are finished.

        Raises Interrupted (nothing committed) if stop_check fires while embedding.
        """
        prepared: List[Prepared] = []
        deletes: List[Tuple[str, int]] = []
        finished: List[Tuple[str, int]] = []
        for path, op, seq in items:
            if self.stop_check():
                break  # leave remaining items queued
            try:
                if op == "delete" or not os.path.exists(path) or \
                        not cfg_valid(path, self.projects):
                    deletes.append((path, seq))
                    continue
                p = self.prepare(path, seq, force)
                if p is None:
                    finished.append((path, seq))
                else:
                    prepared.append(p)
            except (PermissionError, FileNotFoundError):
                deletes.append((path, seq))
            except Exception as e:
                log.warning("failed %s: %s", path, e)
                self.stats["errors"] += 1
                self.store.fail(path)

        # Embed only chunk texts we have never embedded (by hash) with this model.
        all_metas = [m for p in prepared for m in p.metas]
        vectors: Dict[str, np.ndarray] = self.store.vectors_for_hashes(m["chunk_hash"] for m in all_metas)
        todo: Dict[str, str] = {}
        for m in all_metas:
            if m["chunk_hash"] not in vectors and m["chunk_hash"] not in todo:
                todo[m["chunk_hash"]] = m["embed_text"]
        self.stats["reused"] += len(all_metas) - len(todo)
        if todo:
            keys = list(todo)
            vecs = self.embedder.embed_documents([todo[k] for k in keys], stop_check=self.stop_check)
            for k, v in zip(keys, vecs):
                vectors[k] = v
            self.stats["embedded"] += len(keys)

        rows = []
        for m in all_metas:
            row = {k: v for k, v in m.items() if k != "embed_text"}
            row["vector"] = vectors[m["chunk_hash"]]
            rows.append(row)
        self.stats["chunks"] += len(rows)

        self.store.replace_rows([p.path for p in prepared] + [d[0] for d in deletes], rows)
        for p in prepared:
            self.store.manifest_set(p.path, p.mtime_ns, p.size, p.content_hash)
            finished.append((p.path, p.seq))
            self.stats["files"] += 1
        for path, seq in deletes:
            self.store.manifest_delete(path)
            finished.append((path, seq))
            self.stats["deleted"] += 1
        return finished

    def index_paths(self, paths: List[str], force: bool = False, batch: int = None) -> None:
        """Index files directly, bypassing the queue (used by the eval harness)."""
        batch = batch or cfg.WORKER_BATCH_FILES
        for i in range(0, len(paths), batch):
            self.process([(p, "upsert", 0) for p in paths[i:i + batch]], force=force)


def cfg_valid(path: str, projects: Projects) -> bool:
    return cfg.is_valid_file(path, is_ignored=projects.is_ignored)


def cfg_ai_source(path: str) -> Optional[str]:
    return cfg.ai_note_source(path)


# ------------------------------------------------------------------- scanning
def reconcile(store: Store, projects: Projects, roots: List[str] = None,
              stop_check: Callable[[], bool] = None) -> Dict[str, int]:
    """Compare disk with the manifest and queue the differences. Catches events watchdog missed."""
    roots = [os.path.normpath(r) for r in (roots or cfg.WATCH_ROOTS)]
    manifest = store.manifest_all()
    seen = set()
    items = []
    for path in projects.iter_files(roots):
        if stop_check and stop_check():
            log.info("reconcile interrupted")
            return {"queued": 0, "deleted": 0, "interrupted": 1}
        seen.add(path)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if manifest.get(path) != (st.st_mtime_ns, st.st_size):
            items.append((path, "upsert", st.st_mtime))
    n_up = store.enqueue_many(items)

    prefixes = tuple(os.path.normcase(r).rstrip("\\/") + os.sep for r in roots)
    gone = []
    for p in manifest:
        if p in seen or not os.path.normcase(p).startswith(prefixes):
            continue
        if not os.path.exists(p) or not cfg_valid(p, projects):
            gone.append((p, "delete", 1e18))
    n_del = store.enqueue_many(gone)
    log.info("reconcile: %d changed/new, %d gone", n_up, n_del)
    return {"queued": n_up, "deleted": n_del}


def open_store(embedder: Embedder, data_dir: Path) -> Store:
    """Open the store, probing the embedding dimension only if the model changed."""
    peek = Store(data_dir, embedder.model, dim=None, read_only=True)
    same = peek.get_meta("model_id") == embedder.model and peek.get_meta("dim")
    dim = int(peek.get_meta("dim")) if same else embedder.dim
    peek.close()
    store = Store(data_dir, embedder.model, dim)
    return store


def run(args) -> int:
    data_dir = Path(args.data_dir or cfg.DATA_DIR)
    ok, why = power.worker_may_continue(args.allow_battery, respect_activity=False)
    if not ok:
        log.info("not starting: %s", why)
        return 0

    embedder = Embedder(args.model)
    projects = Projects()
    stopped = {"why": ""}

    def stop_check() -> bool:
        if stopped["why"]:
            return True
        ok, why = power.worker_may_continue(args.allow_battery, respect_activity=not args.now)
        if not ok:
            stopped["why"] = why
        return not ok

    store = None
    t0 = time.time()
    try:
        store = open_store(embedder, data_dir)
        if store.check_model():
            log.info("model changed: index wiped, full re-index queued")
        roots = args.path or None
        if args.path or args.reconcile:
            res = reconcile(store, projects, roots, stop_check)
            if not args.path and not res.get("interrupted"):
                store.set_meta("last_reconcile", str(time.time()))
        pipe = Pipeline(store, embedder, projects, stop_check)
        while not stop_check():
            items = store.claim(cfg.WORKER_BATCH_FILES, ignore_debounce=args.now)
            if not items:
                break
            try:
                finished = pipe.process(items)
            except Interrupted:
                break
            except EmbedError as e:
                log.error("embedding failed, is Ollama running? %s", e)
                for p, _, _ in items:
                    store.fail(p, delay=600)
                return 2
            store.done(finished)
            log.info("progress: %s | %d queued", pipe.stats, store.queue_size())
            if args.limit and pipe.stats["files"] >= args.limit:
                break
        if stopped["why"]:
            log.info("stopping early (%s); progress committed, queue kept", stopped["why"])
        store.maintain()
        log.info("done in %.1fs: %s (max chunk %d chars, %d prompt tokens)", time.time() - t0, pipe.stats,
                 embedder.max_chars_seen, embedder.tokens_seen)
        return 0
    finally:
        embedder.unload()  # keep_alive=0 -> VRAM back to 0
        if store:
            store.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--now", action="store_true", help="skip the debounce and the 'user is active' yield")
    ap.add_argument("--path", action="append", help="scan + index this directory (repeatable)")
    ap.add_argument("--reconcile", action="store_true", help="scan all watch roots for missed changes first")
    ap.add_argument("--allow-battery", action="store_true", help="index even when unplugged")
    ap.add_argument("--limit", type=int, help="stop after N files (testing)")
    ap.add_argument("--model", help=f"embedding model (default {cfg.EMBED_MODEL})")
    ap.add_argument("--data-dir", help="store location (default %%LOCALAPPDATA%%\\VectorEmbed)")
    args = ap.parse_args(argv)

    log_dir = Path(args.data_dir or cfg.DATA_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_dir / "worker.log", encoding="utf-8")])
    with single_instance("worker", Path(args.data_dir or cfg.DATA_DIR)) as got:
        if not got:
            log.info("another worker is running")
            return 0
        return run(args)


if __name__ == "__main__":
    sys.exit(main())
