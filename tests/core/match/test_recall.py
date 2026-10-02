import os
from pathlib import Path

import pytest
from tests.core.conftest import Env

from vector_embed.core.documents import DocumentError, DocumentLoader
from vector_embed.core.match import recall as rc
from vector_embed.core.match.recall import MatchCandidate, Recall
from vector_embed.core.skills.base import SkillContext

JD = (
    "Senior backend engineer wanted. Responsibilities: design payment APIs in Python and "
    "PostgreSQL. Requirements: five years of experience with Python, PostgreSQL, payments, "
    "Kubernetes and AWS."
)


def resume(*skills: str, name: str = "Jane Doe") -> str:
    return (
        f"{name}\nSummary\nBackend engineer building payment systems\nWork Experience\n"
        f"Built payment APIs with {', '.join(skills)}\nEducation\nBSc Computer Science\n"
        f"Skills\n{', '.join(skills)}\nProjects\nSearch engine\n"
    )


def write(env: Env, rel: str, text: str, mtime: int | None = None) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return str(path)


@pytest.fixture
def library(env: Env) -> dict[str, str]:
    paths = {
        "v1": write(env, "Resume_v1.txt", resume("Python", "PostgreSQL", "AWS"), 1_700_000_000),
        "v2": write(env, "Resume_final.txt", resume("Python", "PostgreSQL", "AWS"), 1_750_000_000),
        "java": write(
            env, "Resume_java.txt", resume("Java", "Spring", "Oracle", name="Joe Bloggs")
        ),
        "jd": write(env, "JD_acme.txt", JD),
        "notes": write(env, "groceries.txt", "milk eggs bread\n" * 5),
    }
    env.indexer.index_paths(list(paths.values()))
    env.indexer.assign_version_groups()
    env.store.maintain()
    return paths


@pytest.fixture
def recall(env: Env, skill_ctx: SkillContext) -> Recall:
    return Recall(skill_ctx, DocumentLoader(env.store, env.scope, lambda: env.extractors))


def names(candidates: list[MatchCandidate]) -> list[str]:
    return [c.name for c in candidates]


class TestRecall:
    def test_only_resumes_best_match_first_and_newest_version_only(
        self, recall: Recall, library: dict[str, str]
    ) -> None:
        found = recall.recall(JD)
        assert names(found) == ["Resume_final.txt", "Resume_java.txt"]
        assert found[0].similarity > found[1].similarity
        assert found[0].versions == 2
        assert found[0].is_latest
        assert found[1].versions == 1

    def test_old_versions_can_be_shown_but_are_not_ticked(
        self, recall: Recall, library: dict[str, str]
    ) -> None:
        found = {c.name: c for c in recall.recall(JD, include_old_versions=True)}
        assert set(found) == {"Resume_v1.txt", "Resume_final.txt", "Resume_java.txt"}
        assert found["Resume_v1.txt"].is_latest is False
        assert found["Resume_v1.txt"].selected is False
        assert found["Resume_final.txt"].is_latest

    def test_default_ticks_follow_the_similarity_threshold(
        self, recall: Recall, library: dict[str, str], skill_ctx: SkillContext
    ) -> None:
        found = {c.name: c for c in recall.recall(JD)}
        assert found["Resume_final.txt"].selected
        skill_ctx.settings = skill_ctx.settings.model_copy(
            update={
                "match": skill_ctx.settings.match.model_copy(update={"similarity_threshold": 0.99})
            }
        )
        strict = Recall(skill_ctx, recall._loader)
        assert not any(c.selected for c in strict.recall(JD))

    def test_other_document_types_can_be_searched(
        self, recall: Recall, library: dict[str, str]
    ) -> None:
        assert names(recall.recall("milk eggs bread", doc_type="other")) == ["groceries.txt"]
        everything = names(recall.recall(JD, doc_type=""))
        assert "JD_acme.txt" in everything
        assert "groceries.txt" in everything

    def test_locked_paths_are_flagged(
        self, env: Env, skill_ctx: SkillContext, library: dict[str, str]
    ) -> None:
        loader = DocumentLoader(env.store, env.scope, lambda: env.extractors)
        locked = Recall(skill_ctx, loader, is_locked=lambda p: p.endswith("Resume_java.txt"))
        flags = {c.name: c.locked for c in locked.recall(JD)}
        assert flags == {"Resume_final.txt": False, "Resume_java.txt": True}

    def test_tokens_estimate_and_large_document_fallback(
        self, env: Env, skill_ctx: SkillContext
    ) -> None:
        env.settings = env.settings.model_copy(
            update={
                "doctypes": env.settings.doctypes.model_copy(update={"full_text_max_chars": 30})
            }
        )
        env.indexer.settings = env.settings
        path = write(env, "Resume_big.txt", resume("Python", "PostgreSQL") * 5)
        env.indexer.index_paths([path])
        env.store.maintain()
        loader = DocumentLoader(env.store, env.scope, lambda: env.extractors)
        (candidate,) = Recall(skill_ctx, loader).recall(JD)
        assert candidate.tokens == Path(path).stat().st_size // 4

    def test_empty_library_returns_nothing(self, recall: Recall) -> None:
        assert recall.recall(JD) == []


class TestEditingTheList:
    def candidates(self) -> list[MatchCandidate]:
        def make(name: str, sim: float, latest: bool = True) -> MatchCandidate:
            return MatchCandidate(name, name, "resume", 1, sim, 100, is_latest=latest)

        return [make("a", 0.9), make("b", 0.8), make("c", 0.7), make("old", 0.95, latest=False)]

    def test_select_helpers(self) -> None:
        items = self.candidates()
        rc.select_all(items)
        assert len(rc.selected(items)) == 4
        rc.select_none(items)
        assert rc.selected(items) == []
        rc.select_top(items, 2)
        assert [c.path for c in rc.selected(items)] == ["a", "b"]  # newest versions only

    def test_add_file_includes_a_document_recall_missed(
        self, env: Env, skill_ctx: SkillContext, recall: Recall
    ) -> None:
        extra = write(env, "elsewhere/Resume_other.txt", resume("Python", "PostgreSQL"))
        candidate = recall.add_file(extra, JD)
        assert candidate.selected
        assert candidate.name == "Resume_other.txt"
        assert candidate.similarity > 0.3
        assert candidate.tokens > 0

    def test_add_file_refuses_secrets(self, env: Env, recall: Recall) -> None:
        with pytest.raises(DocumentError, match="secret"):
            recall.add_file(write(env, ".env", "TOKEN=1"), JD)


def test_keyword_query_dedupes_and_caps() -> None:
    assert rc.keyword_query("Python python PYTHON sql, go", 10) == "python sql"
    assert rc.keyword_query("alpha beta gamma delta", 2) == "alpha beta"
    assert rc.keyword_query("", 5) == ""
