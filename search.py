"""Hybrid search: vector + BM25 (LanceDB FTS) fused with RRF, plus query filters.

Filters:  type:img|code|doc|plan|memory|note|pdf|docx  ext:py  proj:foo  in:D:\\x  after:2026-01  before:2026-06
Example:  payment retry proj:billing type:code after:2026-01
"""
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import indexer_config as cfg
import power
from embedder import EmbedError, Embedder
from store import Store, _q

_FILTER_RE = re.compile(r'\b(type|ext|proj|project|in|after|before):("[^"]*"|\S+)', re.I)
_TYPE_SQL = {
    "img": "kind = 'image'", "image": "kind = 'image'", "images": "kind = 'image'",
    "code": "kind IN ('code','outline')",
    "doc": "kind = 'doc'", "docs": "kind = 'doc'",
    "plan": "source = 'claude-plan'", "plans": "source = 'claude-plan'",
    "memory": "source = 'claude-memory'", "mem": "source = 'claude-memory'",
    "rules": "source = 'agent-rules'",
    "note": "kind = 'ai-note'", "ai": "kind = 'ai-note'",
}


class SearchDisabled(Exception):
    """Search is turned off while on battery (SEARCH_ON_BATTERY = False)."""


@dataclass
class Result:
    path: str
    project: str
    kind: str
    source: str
    symbol: str
    start_line: int
    end_line: int
    page: int
    snippet: str
    score: float
    ext: str = ""
    mtime: int = 0
    extra_hits: int = 0
    text: str = ""

    @property
    def location(self) -> str:
        if self.page:
            return f"page {self.page}"
        if self.start_line:
            return f"line {self.start_line}"
        return ""


def _parse_date(s: str, end: bool = False) -> Optional[int]:
    for fmt, step in (("%Y-%m-%d", "d"), ("%Y-%m", "m"), ("%Y", "y")):
        try:
            d = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if end:  # 'before:2026-06' means before the START of that period, so no adjustment
            pass
        return int(d.timestamp())
    return None


def parse_query(q: str) -> Tuple[str, str]:
    """Split a raw query into (free text, SQL where clause or '')."""
    clauses: List[str] = []

    def take(m: "re.Match") -> str:
        key, val = m.group(1).lower(), m.group(2).strip('"')
        if key == "type":
            ors = []
            for t in val.lower().split(","):
                if t in _TYPE_SQL:
                    ors.append(_TYPE_SQL[t])
                elif t:
                    ors.append(f"ext = {_q('.' + t.lstrip('.'))}")
            if ors:
                clauses.append("(" + " OR ".join(ors) + ")")
        elif key == "ext":
            clauses.append(f"ext = {_q('.' + val.lower().lstrip('.'))}")
        elif key in ("proj", "project"):
            clauses.append(f"lower(project) = {_q(val.lower())}")
        elif key == "in":
            prefix = os.path.normpath(val).rstrip("\\/") + os.sep
            clauses.append(f"starts_with(lower(path), {_q(prefix.lower())})")
        elif key in ("after", "before"):
            ts = _parse_date(val)
            if ts is not None:
                clauses.append(f"mtime {'>=' if key == 'after' else '<'} {ts}")
        return " "
    text = _FILTER_RE.sub(take, q)
    return " ".join(text.split()), " AND ".join(clauses)


def _fts_terms(text: str) -> str:
    return " ".join(re.findall(r"\w+", text))


class Searcher:
    def __init__(self, store: Store = None, embedder: Embedder = None):
        self.embedder = embedder or Embedder()
        self.store = store or Store(cfg.DATA_DIR, self.embedder.model, dim=None, read_only=True)

    def warm(self) -> None:
        self.embedder.warm(cpu=self._cpu())

    @staticmethod
    def _cpu() -> bool:
        return cfg.SEARCH_CPU_ON_BATTERY and not power.on_ac_power()

    def search(self, query: str, limit: int = None, current_project: str = None) -> List[Result]:
        if not cfg.SEARCH_ON_BATTERY and not power.on_ac_power():
            raise SearchDisabled("search is disabled on battery")
        limit = limit or cfg.SEARCH_RESULTS
        table = self.store.table
        text, where = parse_query(query)
        if table is None:
            return []
        n = cfg.SEARCH_CANDIDATES

        if not text:  # filters only: newest matching chunks
            q = table.search()
            if where:
                q = q.where(where)
            rows = q.select(_COLS).limit(n * 4).to_list()
            rows.sort(key=lambda r: r["mtime"], reverse=True)
            return self._group([(r, 1.0) for r in rows], limit, current_project)

        ranked: Dict[Tuple[str, str], list] = {}   # key -> [row, score]

        def add(rows, weight=1.0):
            for rank, r in enumerate(rows):
                key = (r["path"], r["chunk_hash"])
                entry = ranked.setdefault(key, [r, 0.0])
                entry[1] += weight / (cfg.RRF_K + rank + 1)

        # Vector leg (falls back to keyword-only if Ollama is unreachable).
        try:
            vec = self.embedder.embed_query(text, cpu=self._cpu())
            q = table.search(vec).metric("cosine")
            if where:
                q = q.where(where, prefilter=True)
            add(q.select(_COLS).limit(n).to_list())
        except EmbedError:
            pass

        # Keyword leg: exact identifiers / error strings.
        terms = _fts_terms(text)
        if terms:
            try:
                q = table.search(terms, query_type="fts")
                if where:
                    q = q.where(where, prefilter=True)
                add(q.select(_COLS).limit(n).to_list())
            except Exception:
                pass

        return self._group(list(ranked.values()), limit, current_project)

    def _group(self, pairs, limit: int, current_project: Optional[str]) -> List[Result]:
        best: Dict[str, Result] = {}
        for row, score in pairs:
            if current_project and row["project"].lower() == current_project.lower():
                score *= cfg.CURRENT_PROJECT_BOOST
            path = row["path"]
            cur = best.get(path)
            if cur is None:
                best[path] = _to_result(row, score)
            else:
                cur.extra_hits += 1
                if score > cur.score:
                    hits = cur.extra_hits
                    best[path] = _to_result(row, score)
                    best[path].extra_hits = hits
        out = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return out[:limit]


_COLS = ["text", "path", "project", "kind", "source", "ext", "symbol", "start_line",
         "end_line", "page", "chunk_hash", "mtime"]


def _to_result(row: dict, score: float) -> Result:
    text = row["text"]
    body = text.split("\n", 1)[1] if "\n" in text else text
    snippet = " ".join(body.split())[:260]
    return Result(path=row["path"], project=row["project"], kind=row["kind"], source=row["source"],
                  symbol=row["symbol"], start_line=row["start_line"], end_line=row["end_line"],
                  page=row["page"], snippet=snippet, score=score, ext=row["ext"], mtime=row["mtime"],
                  text=body)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="+")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--data-dir")
    a = ap.parse_args(argv)
    emb = Embedder()
    store = Store(Path(a.data_dir or cfg.DATA_DIR), emb.model, dim=None, read_only=True)
    s = Searcher(store, emb)
    t = time.time()
    res = s.search(" ".join(a.query), a.n)
    print(f"{len(res)} results in {(time.time() - t) * 1000:.0f} ms")
    for r in res:
        loc = f" ({r.location})" if r.location else ""
        print(f"{r.score:.4f}  [{r.kind}] {r.path}{loc}  {r.symbol}\n        {r.snippet[:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
