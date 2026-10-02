"""Ollama embedding wrapper: batching, num_ctx, per-model prefixes, keep_alive, retry."""
import time
from typing import List, Optional

import numpy as np
import ollama

import indexer_config as cfg

_MAX_BATCH_CHARS = 60_000  # keeps one request's total prompt size sane


class EmbedError(RuntimeError):
    pass


class Interrupted(Exception):
    """Raised between embedding batches when the caller's stop_check says to stop."""


class Embedder:
    def __init__(self, model: str = None, out_dim: int = None, client: ollama.Client = None):
        self.model = model or cfg.EMBED_MODEL
        self.client = client or ollama.Client()
        self.prefixes = cfg.MODEL_PREFIXES.get(self.model, {"query": "", "document": ""})
        self._max_dim = out_dim or cfg.EMBED_DIM
        self._dim: Optional[int] = None
        self.max_chars_seen = 0      # for the "no truncation" check
        self.tokens_seen = 0         # prompt tokens as reported by Ollama

    # ------------------------------------------------------------------ core
    def _call(self, texts: List[str], keep_alive, cpu: bool = False) -> np.ndarray:
        options = {"num_ctx": cfg.NUM_CTX}
        if cpu:
            options["num_gpu"] = 0
        last: Optional[Exception] = None
        for attempt in range(4):
            try:
                r = self.client.embed(model=self.model, input=texts, options=options,
                                      keep_alive=keep_alive)
                self.tokens_seen += int(getattr(r, "prompt_eval_count", 0) or 0)
                arr = np.asarray(r["embeddings"], dtype=np.float32)
                if arr.shape[0] != len(texts):
                    raise EmbedError(f"expected {len(texts)} vectors, got {arr.shape[0]}")
                return self._finish(arr)
            except Exception as e:  # network blips, model still loading, OOM retry
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise EmbedError(f"ollama embed failed after retries: {last}")

    def _finish(self, arr: np.ndarray) -> np.ndarray:
        """Matryoshka-truncate if the model is wider than the configured dim, then L2-normalise."""
        if arr.shape[1] > self._max_dim:
            arr = arr[:, : self._max_dim]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    @property
    def dim(self) -> int:
        """Output dimension (probes the model once)."""
        if self._dim is None:
            self._dim = int(self._call(["dimension probe"], keep_alive=cfg.SEARCH_MODEL_KEEP_ALIVE).shape[1])
        return self._dim

    # ------------------------------------------------------------- documents
    def embed_documents(self, texts: List[str], keep_alive="5m", stop_check=None) -> np.ndarray:
        """Embed chunk texts (document prefix applied), in size-bounded batches."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.prefixes["document"]
        prepared = [prefix + t for t in texts]
        self.max_chars_seen = max(self.max_chars_seen, max(len(t) for t in prepared))
        out: List[np.ndarray] = []
        batch: List[str] = []
        size = 0
        for t in prepared:
            if batch and (len(batch) >= cfg.EMBED_BATCH_SIZE or size + len(t) > _MAX_BATCH_CHARS):
                if stop_check and stop_check():
                    raise Interrupted()
                out.append(self._call(batch, keep_alive))
                batch, size = [], 0
            batch.append(t)
            size += len(t)
        if batch:
            if stop_check and stop_check():
                raise Interrupted()
            out.append(self._call(batch, keep_alive))
        return np.vstack(out)

    # ---------------------------------------------------------------- queries
    def embed_query(self, query: str, cpu: bool = False) -> np.ndarray:
        return self._call([self.prefixes["query"] + query], cfg.SEARCH_MODEL_KEEP_ALIVE, cpu=cpu)[0]

    def warm(self, cpu: bool = False) -> None:
        """Load the model into memory (empty-ish embed) so the first real query is fast."""
        try:
            self._call(["warmup"], cfg.SEARCH_MODEL_KEEP_ALIVE, cpu=cpu)
        except EmbedError:
            pass

    def unload(self) -> None:
        """Release VRAM now (keep_alive=0)."""
        try:
            self.client.embed(model=self.model, input=["x"], options={"num_ctx": cfg.NUM_CTX},
                              keep_alive=0)
        except Exception:
            pass
