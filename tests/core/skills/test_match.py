import pytest
from tests.core.conftest import Chat, Env
from tests.core.match.test_pipeline import JD, faithful_model, resume, write

from vector_embed.core.match.pipeline import MatchRun
from vector_embed.core.skills.base import SkillContext, create_skill
from vector_embed.core.skills.match import MatchInput, MatchSkill, format_scores, pipeline_of


@pytest.fixture
def skill(env: Env, chat: Chat, skill_ctx: SkillContext) -> MatchSkill:
    chat.client.chat_json_fn = faithful_model
    paths = [
        write(env, "Resume_a.txt", resume("Jane Doe", "Python", "PostgreSQL", "Kubernetes", "AWS")),
        write(env, "Resume_b.txt", resume("Sam Roe", "Python")),
        write(env, "Resume_c.txt", resume("Joe Bloggs", "Java", "Spring")),
    ]
    env.indexer.index_paths(paths)
    env.store.maintain()
    return MatchSkill(skill_ctx)


def test_run_scores_the_ticked_candidates(skill: MatchSkill) -> None:
    run = skill.run(MatchInput(jd=JD, top=2))
    assert isinstance(run, MatchRun)
    assert len(run.scores) == 2
    assert run.ranked()[0].candidate.name == "Resume_a.txt"
    table = skill.render(run)
    assert table.splitlines()[0].startswith(" 1. 100  Resume_a.txt")
    assert "2/2 must-haves met" in table


def test_job_description_can_come_from_a_file(skill: MatchSkill, env: Env) -> None:
    jd_file = write(env, "jd.txt", JD)
    run = skill.run(MatchInput(jd_file=jd_file, top=1))
    assert run.jd_text.startswith("Senior backend engineer")
    assert len(run.scores) == 1


def test_missing_input_is_a_clear_error(skill: MatchSkill) -> None:
    with pytest.raises(RuntimeError, match="--jd-file or --jd"):
        skill.run(MatchInput())
    with pytest.raises(RuntimeError, match="--jd-file or --jd"):
        skill.run(MatchInput(jd="   "))


def test_stream_reports_progress_table_and_verdict(skill: MatchSkill, chat: Chat) -> None:
    chat.client.chat_reply = ["Jane is the best fit."]
    text = "".join(skill.stream(MatchInput(jd=JD, top=2)))
    assert "Recalled" in text
    assert "scoring 1/2" in text
    assert "Resume_a.txt" in text
    assert text.rstrip().endswith("Jane is the best fit.")


def test_nothing_above_the_threshold_scores_nothing(
    skill: MatchSkill, skill_ctx: SkillContext
) -> None:
    skill_ctx.settings = skill_ctx.settings.model_copy(
        update={"match": skill_ctx.settings.match.model_copy(update={"similarity_threshold": 1.0})}
    )
    text = "".join(MatchSkill(skill_ctx).stream(MatchInput(jd=JD)))
    assert "Nothing to score" in text
    run = MatchSkill(skill_ctx).run(MatchInput(jd=JD))
    assert run.scores == []
    assert MatchSkill(skill_ctx).render(run) == "no documents scored"


def test_unrelated_document_types_return_no_candidates(skill: MatchSkill) -> None:
    run = skill.run(MatchInput(jd=JD, doc_type="invoice"))
    assert run.candidates == []
    assert "no documents scored" in skill.render(run)


def test_format_scores_marks_unverified_cut_and_failed_documents(skill: MatchSkill) -> None:
    run = skill.run(MatchInput(jd=JD, top=3))
    first = run.scores[0]
    assert first.breakdown and first.judgement
    first.breakdown.unverified = 1
    object.__setattr__(first.judgement, "reduced", True)
    run.scores[1].breakdown = None
    run.scores[1].error = "boom"
    table = format_scores(run)
    assert "⚠" in table
    assert "(reduced: cut to fit)" in table
    assert "boom" in table


def test_registry_and_roles(skill_ctx: SkillContext, chat: Chat) -> None:
    created = create_skill("match", skill_ctx)
    assert isinstance(created, MatchSkill)
    assert created.ui_hint == "table"
    assert "match_scorer" in created.roles


def test_pipeline_needs_a_document_loader(skill_ctx: SkillContext, chat: Chat) -> None:
    skill_ctx.extras.pop("documents")
    with pytest.raises(RuntimeError, match="document loader"):
        pipeline_of(skill_ctx)
