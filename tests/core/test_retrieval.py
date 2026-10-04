from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tests.core.conftest import Env, Power

from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.retrieval import Candidate, fts_terms, hybrid_candidates
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.store.lance import CHUNKS, DOCUMENTS

COLUMNS = ["path", "chunk_hash", "text"]


def index(env: Env, name: str, text: str) -> str:
    path = env.root / name
    path.write_text(text, encoding="utf-8")
    env.indexer.index_paths([str(path)])
    return str(path)


def candidates(ctx: SkillContext, text: str, **kw: object) -> list[str]:
    ctx.store.maintain()
    found = hybrid_candidates(
        ctx.store, ctx.embedder, ctx.power, ctx.settings.search,
        table=CHUNKS, text=text, columns=COLUMNS, **kw,
    )  # type: ignore[arg-type]  # fmt: skip
    return [c.row["path"] for c in found]


def test_terms_keep_identifiers_and_drop_punctuation() -> None:
    assert fts_terms("charge_card(order)!") == "charge_card order"


def test_vector_and_keyword_legs_are_fused(env: Env, skill_ctx: SkillContext) -> None:
    pay = index(env, "pay.txt", "retry failed payments with backoff\n")
    index(env, "other.txt", "completely unrelated gardening notes\n")
    assert candidates(skill_ctx, "retry failed payments")[0] == pay


def test_scores_are_descending_and_unique_per_chunk(env: Env, skill_ctx: SkillContext) -> None:
    index(env, "a.txt", "alpha beta gamma\n")
    skill_ctx.store.maintain()
    found = hybrid_candidates(
        skill_ctx.store, skill_ctx.embedder, skill_ctx.power, skill_ctx.settings.search,
        table=CHUNKS, text="alpha beta", columns=COLUMNS,
    )  # fmt: skip
    scores = [c.score for c in found]
    assert scores == sorted(scores, reverse=True)
    assert len({(c.row["path"], c.row["chunk_hash"]) for c in found}) == len(found)


def test_candidates_keep_each_legs_raw_score_and_rank(env: Env, skill_ctx: SkillContext) -> None:
    pay = index(env, "pay.txt", "retry failed payments with backoff\n")
    index(env, "other.txt", "completely unrelated gardening notes\n")
    skill_ctx.store.maintain()
    found = hybrid_candidates(
        skill_ctx.store, skill_ctx.embedder, skill_ctx.power, skill_ctx.settings.search,
        table=CHUNKS, text="payments", columns=COLUMNS,
    )  # fmt: skip
    by_path = {c.row["path"]: c for c in found}
    top = by_path[pay]
    assert top.keyword_rank == 1
    assert top.vector_rank in (1, 2)
    assert top.similarity is not None and 0 < top.similarity <= 1
    assert top.bm25 is not None and top.bm25 > 0
    rrf_k = skill_ctx.settings.search.rrf_k
    assert top.score == pytest.approx(1 / (rrf_k + 1) + 1 / (rrf_k + top.vector_rank))
    other = next(c for p, c in by_path.items() if p != pay)
    assert other.keyword_rank is None and other.bm25 is None  # no keyword match
    assert other.vector_rank is not None and other.similarity is not None


def test_a_photo_without_text_is_found_by_name_never_by_meaning(
    env: Env, skill_ctx: SkillContext
) -> None:
    photo = env.root / "passport.png"
    Image.new("RGB", (300, 300), "white").save(photo)
    env.indexer.index_paths([str(photo)])
    index(env, "notes.txt", "renew the passport before the trip to japan\n")
    rows = skill_ctx.store.scan(CHUNKS, ["path", "content_chars"])
    assert {Path(r["path"]).name: r["content_chars"] for r in rows}["passport.png"] == 0

    found = {Path(c.row["path"]).name: c for c in _hybrid(skill_ctx, "trip to japan")}
    assert "passport.png" not in found  # no keyword match, and the vector leg skips it
    by_name = {Path(c.row["path"]).name: c for c in _hybrid(skill_ctx, "passport")}
    assert by_name["passport.png"].keyword_rank is not None
    assert by_name["passport.png"].vector_rank is None


def test_the_content_floor_can_be_turned_off(env: Env, skill_ctx: SkillContext) -> None:
    photo = env.root / "IMG_2041.png"
    Image.new("RGB", (300, 300), "white").save(photo)
    env.indexer.index_paths([str(photo)])
    skill_ctx.settings = skill_ctx.settings.model_copy(
        update={"search": skill_ctx.settings.search.model_copy(update={"min_content_chars": 0})}
    )
    assert any(c.vector_rank for c in _hybrid(skill_ctx, "beach holiday"))


def _hybrid(ctx: SkillContext, text: str) -> list[Candidate]:
    ctx.store.maintain()
    return hybrid_candidates(
        ctx.store, ctx.embedder, ctx.power, ctx.settings.search,
        table=CHUNKS, text=text, columns=COLUMNS,
    )  # fmt: skip


def test_force_cpu_embeds_the_query_on_the_cpu(env: Env, skill_ctx: SkillContext) -> None:
    index(env, "a.txt", "alpha beta gamma\n")
    seen: list[bool] = []
    original = skill_ctx.embedder.embed

    def spy(texts: list[str], kind: str = "doc", cpu: bool = False) -> np.ndarray:
        seen.append(cpu)
        return original(texts, kind, cpu)  # type: ignore[arg-type]

    skill_ctx.embedder.embed = spy  # type: ignore[method-assign]
    candidates(skill_ctx, "alpha", force_cpu=True)
    candidates(skill_ctx, "alpha")
    assert seen == [True, False]


def test_battery_forces_cpu_too(env: Env, skill_ctx: SkillContext, power_state: Power) -> None:
    index(env, "a.txt", "alpha beta gamma\n")
    power_state.on_ac = False
    seen: list[bool] = []
    original = skill_ctx.embedder.embed
    skill_ctx.embedder.embed = lambda texts, kind="doc", cpu=False: (  # type: ignore[method-assign]
        seen.append(cpu) or original(texts, kind, cpu)  # type: ignore[arg-type]
    )
    candidates(skill_ctx, "alpha")
    assert seen == [True]


def test_keyword_only_when_the_model_server_is_down(
    env: Env, skill_ctx: SkillContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = index(env, "ident.txt", "the function charge_card handles billing\n")

    def down(*_a: object, **_k: object) -> None:
        raise ProviderError("down")

    monkeypatch.setattr(skill_ctx.embedder, "embed", down)
    assert candidates(skill_ctx, "charge_card") == [path]


def test_works_on_the_documents_table(env: Env, skill_ctx: SkillContext) -> None:
    index(env, "resume.txt", "python developer building payment systems\n" * 3)
    skill_ctx.store.maintain()
    found = hybrid_candidates(
        skill_ctx.store, skill_ctx.embedder, skill_ctx.power, skill_ctx.settings.search,
        table=DOCUMENTS, text="python payment developer", columns=["path", "doc_type"],
        unique_key=("path", "path"),
    )  # fmt: skip
    assert len(found) == 1
