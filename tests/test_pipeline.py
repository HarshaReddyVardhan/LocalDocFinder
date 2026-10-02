"""Pipeline behaviour with a fake embedder (fast, deterministic) + one real-Ollama smoke test."""
import os
import time

import numpy as np
import pytest

import indexer_config as cfg
import power
import worker
from embedder import Embedder, Interrupted
from projects import Projects
from store import Store


class FakeEmbedder:
    model = "fake-embed"
    dim = 8
    max_chars_seen = 0
    tokens_seen = 0

    def __init__(self):
        self.embedded_texts = []

    def embed_documents(self, texts, keep_alive=None, stop_check=None):
        if stop_check and stop_check():
            raise Interrupted()
        self.embedded_texts += list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            out[i] = rng.normal(size=self.dim)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def unload(self):
        pass


def fn(name: str, body_tag: str = "a") -> str:
    pad = "\n".join(f"    x{i} = '{body_tag}{i}' * 3" for i in range(20))
    return f"def {name}():\n{pad}\n    return x0\n"


def src(**tags) -> str:
    return "\n\n".join(fn(n, t) for n, t in tags.items())


@pytest.fixture
def env(tmp_path):
    proj = tmp_path / "proj1"
    (proj / ".git").mkdir(parents=True)
    emb = FakeEmbedder()
    store = Store(tmp_path / "data", emb.model, emb.dim)
    projects = Projects()
    pipe = worker.Pipeline(store, emb, projects)
    yield pipe, store, emb, proj, tmp_path
    store.close()


def code_chunks(store, path):
    return [r for r in store.table.search().where(f"path = '{path}'").limit(1000).to_list()]


def test_initial_index_and_project_tag(env):
    pipe, store, emb, proj, _ = env
    f = proj / "src" / "a.py"
    f.parent.mkdir()
    f.write_text(src(one="a", two="b", three="c"))
    pipe.index_paths([str(f)])
    rows = code_chunks(store, str(f))
    assert len(rows) >= 4  # outline + 3 functions
    assert {r["project"] for r in rows} == {"proj1"}
    assert {r["model_id"] for r in rows} == {"fake-embed"}
    assert store.manifest_get(str(f)) is not None


def test_edit_one_function_reembeds_one_chunk(env):
    pipe, store, emb, proj, _ = env
    f = proj / "a.py"
    f.write_text(src(one="a", two="b", three="c"))
    pipe.index_paths([str(f)])
    before = len(emb.embedded_texts)
    time.sleep(0.02)
    f.write_text(src(one="a", two="CHANGED", three="c"))
    pipe.index_paths([str(f)])
    assert len(emb.embedded_texts) - before == 1
    assert "CHANGED" in emb.embedded_texts[-1]


def test_touch_without_change_reembeds_nothing(env):
    pipe, store, emb, proj, _ = env
    f = proj / "a.py"
    f.write_text(src(one="a"))
    pipe.index_paths([str(f)])
    n = len(emb.embedded_texts)
    os.utime(f, (time.time() + 100, time.time() + 100))  # e.g. no-op `git checkout`
    pipe.index_paths([str(f)])
    assert len(emb.embedded_texts) == n


def test_delete_removes_rows(env):
    pipe, store, emb, proj, _ = env
    f = proj / "a.py"
    f.write_text(src(one="a"))
    pipe.index_paths([str(f)])
    assert code_chunks(store, str(f))
    f.unlink()
    pipe.process([(str(f), "upsert", 1)])  # file gone -> treated as delete
    assert not code_chunks(store, str(f))
    assert store.manifest_get(str(f)) is None


def test_identical_code_in_two_projects_embedded_once(env):
    pipe, store, emb, proj, tmp = env
    proj2 = tmp / "proj2"
    (proj2 / ".git").mkdir(parents=True)
    paths = []
    for p in (proj, proj2):
        f = p / "lib" / "util.py"
        f.parent.mkdir()
        f.write_text(src(helper="z"))
        paths.append(str(f))
    pipe.index_paths([paths[0]])
    n = len(emb.embedded_texts)
    pipe.index_paths([paths[1]])
    assert len(emb.embedded_texts) == n  # all reused by hash
    assert code_chunks(store, paths[1])[0]["project"] == "proj2"


def test_model_change_wipes_index(env):
    pipe, store, emb, proj, tmp = env
    f = proj / "a.py"
    f.write_text(src(one="a"))
    pipe.index_paths([str(f)])
    assert store.count() > 0
    other = Store(tmp / "data", "another-model", 8)
    assert other.count() == 0 and other.manifest_count() == 0
    other.close()


def test_reconcile_catches_missed_changes(env):
    pipe, store, emb, proj, tmp = env
    a, b = proj / "a.py", proj / "b.py"
    a.write_text(src(one="a"))
    b.write_text(src(two="b"))
    worker.reconcile(store, pipe.projects, [str(proj)])
    assert store.queue_size() == 2
    items = store.claim(10, ignore_debounce=True)
    store.done(pipe.process(items))
    assert store.queue_size() == 0
    # "watcher was off": modify one, delete one, add one.
    a.write_text(src(one="changed"))
    b.unlink()
    (proj / "c.py").write_text(src(three="c"))
    worker.reconcile(store, pipe.projects, [str(proj)])
    ops = {(os.path.basename(p), op) for p, op, _ in store.claim(10, ignore_debounce=True)}
    assert ops == {("a.py", "upsert"), ("c.py", "upsert"), ("b.py", "delete")}


def test_debounce_hides_fresh_events(env):
    pipe, store, emb, proj, _ = env
    store.enqueue(str(proj / "a.py"), delay=30)
    assert store.claim(10) == []
    assert len(store.claim(10, ignore_debounce=True)) == 1


def test_requeue_during_processing_is_not_lost(env):
    pipe, store, emb, proj, _ = env
    p = str(proj / "a.py")
    store.enqueue(p)
    (path, op, seq), = store.claim(1, True)
    time.sleep(0.001)
    store.enqueue(p)           # file changed again while being processed
    store.done([(path, seq)])  # stale seq: row must survive
    assert store.queue_size() == 1


def test_unplug_before_start_does_nothing(env, monkeypatch):
    pipe, store, emb, proj, _ = env
    f = proj / "a.py"
    f.write_text(src(one="a"))
    monkeypatch.setattr(power, "on_ac_power", lambda: False)
    pipe.stop_check = lambda: not power.worker_may_continue(False, respect_activity=False)[0]
    assert pipe.process([(str(f), "upsert", 1)]) == []  # item stays queued
    assert emb.embedded_texts == [] and store.count() == 0


def test_unplug_mid_batch_commits_nothing_and_keeps_queue(env):
    pipe, store, emb, proj, _ = env
    f = proj / "a.py"
    f.write_text(src(one="a", two="b"))
    store.enqueue(str(f))
    calls = {"n": 0}

    def stop():  # plugged in while extracting, unplugged by the time embedding starts
        calls["n"] += 1
        return calls["n"] >= 2

    pipe.stop_check = stop
    items = store.claim(10, ignore_debounce=True)
    with pytest.raises(Interrupted):
        pipe.process(items)
    assert store.count() == 0 and store.manifest_get(str(f)) is None
    assert store.queue_size() == 1  # resumes when plugged back in
    pipe.stop_check = lambda: False
    store.done(pipe.process(store.claim(10, ignore_debounce=True)))
    assert store.count() > 0 and store.queue_size() == 0


def test_power_policy_matrix(monkeypatch):
    monkeypatch.setattr(power, "on_ac_power", lambda: False)
    assert power.worker_may_continue(False, respect_activity=False) == (False, "unplugged")
    assert power.worker_may_continue(True, respect_activity=False)[0] is True  # --allow-battery
    monkeypatch.setattr(power, "on_ac_power", lambda: True)
    assert power.worker_may_continue(False, respect_activity=False)[0] is True


def test_ai_notes_are_tagged(env):
    pipe, store, emb, proj, tmp = env
    # Use a path-shaped copy under tmp so we never write into the real ~/.claude
    f = tmp / ".claude" / "plans" / "p.md"
    f.parent.mkdir(parents=True)
    f.write_text("# Plan\nSwitch the indexer to Ollama embeddings\n")
    assert cfg.is_valid_file(str(f))
    pipe.index_paths([str(f)])
    row = code_chunks(store, str(f))[0]
    assert row["kind"] == "ai-note" and row["source"] == "claude-plan"


# ---- real Ollama ---------------------------------------------------------------------------
def _ollama_ready() -> bool:
    try:
        import ollama
        names = [m.model for m in ollama.Client().list().models]
        return any(n.startswith(cfg.EMBED_MODEL.split(":")[0]) for n in names)
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_ready(), reason="Ollama / embedding model not available")
def test_real_embedding_and_hybrid_search(tmp_path):
    from search import Searcher
    emb = Embedder()
    store = Store(tmp_path / "data", emb.model, emb.dim)
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    (proj / "pay.py").write_text(
        "def charge_card(customer, amount):\n    '''Charge a customer's credit card via the payment gateway'''\n"
        + "    gateway.submit(customer, amount)\n" * 6)
    (proj / "weather.py").write_text(
        "def forecast(city):\n    '''Fetch tomorrow's weather forecast'''\n" + "    return fetch(city)\n" * 6)
    pipe = worker.Pipeline(store, emb, Projects())
    pipe.index_paths([str(proj / "pay.py"), str(proj / "weather.py")])
    s = Searcher(store, emb)
    hits = s.search("bill a customer with their credit card")
    assert hits and hits[0].path.endswith("pay.py")
    hits = s.search("charge_card")
    assert hits[0].path.endswith("pay.py")
    hits = s.search("charge proj:proj type:code ext:py")
    assert all(h.project == "proj" for h in hits)
    assert emb.max_chars_seen < cfg.NUM_CTX * 2  # chars well under num_ctx tokens
    emb.unload()
    store.close()
