"""Compare embedding models on YOUR queries: recall@10 and MRR, plus index time and query latency.

    python eval/run.py                                   # models + corpus from eval/queries.yaml
    python eval/run.py --models qwen3-embedding:0.6b bge-m3 --corpus D:\\Projects\\myapp
    python eval/run.py --reuse                           # reuse indexes built by a previous run

Each model gets its own throw-away index under %LOCALAPPDATA%\\VectorEmbed\\eval\\ so the real index
is never touched. A query is a hit when any `expect` substring appears in a result's path
(case-insensitive, '/' and '\\' are equivalent). Results are collapsed to one entry per file.
"""
import argparse
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import indexer_config as cfg  # noqa: E402
from embedder import Embedder  # noqa: E402
from projects import Projects  # noqa: E402
from search import Searcher  # noqa: E402
from store import Store  # noqa: E402
from worker import Pipeline  # noqa: E402

K = 10


def norm(p: str) -> str:
    return p.replace("\\", "/").lower()


def rank_of(results, expect) -> int:
    """1-based rank of the first result whose path matches any expected substring, else 0."""
    wants = [norm(e) for e in expect]
    for i, r in enumerate(results[:K], 1):
        if any(w in norm(r.path) for w in wants):
            return i
    return 0


def evaluate(model: str, corpus, queries, reuse: bool, verbose: bool, exclude=()):
    emb = Embedder(model)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    data_dir = cfg.DATA_DIR / "eval" / safe
    if data_dir.exists() and not reuse:
        shutil.rmtree(data_dir, ignore_errors=True)
    store = Store(data_dir, model, emb.dim)
    index_s = 0.0
    if store.count() == 0:
        projects = Projects()
        pipe = Pipeline(store, emb, projects)
        skip = {norm(str(e)) for e in exclude}  # the answer key must not be part of the corpus
        paths = [p for root in corpus for p in projects.iter_files([root]) if norm(p) not in skip]
        t = time.time()
        pipe.index_paths(paths)
        store.maintain()
        index_s = time.time() - t
        print(f"  indexed {pipe.stats['files']} files / {pipe.stats['chunks']} chunks in {index_s:.0f}s "
              f"(max chunk {emb.max_chars_seen} chars)")
    searcher = Searcher(store, emb)
    searcher.warm()
    ranks, times = [], []
    for q in queries:
        t = time.time()
        res = searcher.search(q["q"], limit=K)
        times.append((time.time() - t) * 1000)
        r = rank_of(res, q["expect"])
        ranks.append(r)
        if verbose or r == 0:
            top = norm(res[0].path).split("/")[-1] if res else "-"
            print(f"  {'hit@%d' % r if r else 'MISS':7} {q['q'][:60]!r:64} top={top}")
    emb.unload()
    store.close()
    return {
        "model": model,
        f"recall@{K}": sum(1 for r in ranks if r) / len(ranks),
        "mrr": sum(1 / r for r in ranks if r) / len(ranks),
        "index_s": index_s,
        "q_ms": statistics.median(times),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=str(Path(__file__).with_name("queries.yaml")))
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--corpus", nargs="+")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every query, not just misses")
    a = ap.parse_args(argv)

    spec = yaml.safe_load(Path(a.queries).read_text(encoding="utf-8"))
    models = a.models or spec["models"]
    corpus = a.corpus or spec["corpus"]
    queries = spec["queries"]
    print(f"{len(queries)} queries, corpus: {corpus}")

    rows = []
    for m in models:
        print(f"== {m}")
        try:
            rows.append(evaluate(m, corpus, queries, a.reuse, a.verbose, exclude=[Path(a.queries).resolve()]))
        except Exception as e:  # missing model etc.: keep going
            print(f"  failed: {e}")
    print()
    print(f"{'model':28} {'recall@%d' % K:>10} {'MRR':>7} {'index s':>9} {'query ms':>9}")
    for r in sorted(rows, key=lambda r: (-r[f'recall@{K}'], -r['mrr'])):
        print(f"{r['model']:28} {r[f'recall@{K}']:>10.2f} {r['mrr']:>7.3f} {r['index_s']:>9.0f} {r['q_ms']:>9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
