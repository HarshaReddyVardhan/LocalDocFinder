import json
from pathlib import Path
from typing import Any

import pytest
from tests.core.conftest import Chat, CloudRig, Env
from tests.core.match.test_pipeline import CHECKLIST, faithful_model, resume

from vector_embed.app.match_controller import MatchController
from vector_embed.core.documents import DocumentLoader
from vector_embed.core.llm import ChatBlockedError
from vector_embed.core.match.pipeline import MatchPipeline
from vector_embed.core.match.recall import select_all
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.providers.base import Message
from vector_embed.core.skills.ask import AskInput, AskSkill
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.chat import ChatInput, ChatSkill
from vector_embed.core.skills.match import MatchInput, MatchSkill
from vector_embed.core.store.lance import sql_quote

JD = "Senior backend engineer. Requirements: Python, PostgreSQL. Nice to have: Kubernetes, AWS."
PUBLIC_NOTE = (
    "# Payments\n\nWe retry failed payments with backoff. Ticket SSN 123-45-6789 leaked once.\n"
)
PRIVATE_NOTE = "# Private\n\nWe also retry failed payments with a secret strategy.\n"


def write(env: Env, rel: str, text: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def sent_text(inner: Any) -> str:
    return "\n".join(m.content for batch in inner.sent for m in batch)


@pytest.fixture
def notes(env: Env, chat: Chat, skill_ctx: SkillContext) -> dict[str, str]:
    paths = {
        "public": write(env, "payments.md", PUBLIC_NOTE),
        "private": write(env, ".claude/projects/demo/memory/payments-private.md", PRIVATE_NOTE),
    }
    env.indexer.index_paths(list(paths.values()))
    env.store.maintain()
    return paths


class TestAsk:
    def test_cloud_requests_exclude_private_files_and_say_so(
        self, chat: Chat, cloud: CloudRig, skill_ctx: SkillContext, notes: dict[str, str]
    ) -> None:
        cloud.inner.reply = ["Retries use backoff [1]."]
        run = AskSkill(skill_ctx).prepare("how do we retry failed payments")
        list(run.deltas())
        sent = sent_text(cloud.inner)
        assert "retry failed payments with backoff" in sent
        assert "secret strategy" not in sent
        assert run.result.withheld == 1
        assert "1 private file was not sent to the cloud" in run.footer()
        assert chat.chat_calls() == []  # nothing went to the local model either

    def test_local_requests_may_use_private_files(
        self, chat: Chat, skill_ctx: SkillContext, notes: dict[str, str], cloud: CloudRig
    ) -> None:
        cloud.router._settings = cloud.router._settings.model_copy(update={"routing": {}})
        result = AskSkill(skill_ctx).run(AskInput(question="how do we retry failed payments"))
        assert result.withheld == 0
        assert any("payments-private.md" in s.path for s in result.sources)
        assert cloud.inner.sent == []

    def test_ids_in_sources_are_masked_before_leaving(
        self, cloud: CloudRig, skill_ctx: SkillContext, notes: dict[str, str]
    ) -> None:
        AskSkill(skill_ctx).run(AskInput(question="how do we retry failed payments"))
        sent = sent_text(cloud.inner)
        assert "123-45-6789" not in sent
        assert "[SSN REMOVED]" in sent

    def test_only_private_hits_means_not_found_and_no_cloud_call(
        self, env: Env, cloud: CloudRig, skill_ctx: SkillContext
    ) -> None:
        env.indexer.index_paths(
            [write(env, ".claude/projects/demo/memory/only.md", "# Only\n\nzebra quokka secret\n")]
        )
        env.store.maintain()
        result = AskSkill(skill_ctx).run(AskInput(question="zebra quokka"))
        assert result.not_found
        assert cloud.inner.sent == []

    def test_consent_is_required(
        self, cloud: CloudRig, skill_ctx: SkillContext, notes: dict[str, str]
    ) -> None:
        cloud.consent.revoke()
        with pytest.raises(ChatBlockedError, match="consent"):
            AskSkill(skill_ctx).run(AskInput(question="how do we retry failed payments"))
        assert cloud.inner.sent == []


CODE_PUBLIC = """def retry_payment(charge):
    return backoff(charge)  # retry failed payment
"""
CODE_PRIVATE = """def retry_payment_secretly(charge):
    return vault(charge)  # retry failed payment
"""


class TestBlockedDocTypes:
    def test_a_blocked_document_type_never_reaches_the_cloud(
        self, env: Env, chat: Chat, cloud: CloudRig, skill_ctx: SkillContext
    ) -> None:
        env.indexer.index_paths(
            [
                write(env, "retries.md", "# Retries\n\nWe retry failed payments with backoff.\n"),
                write(env, "billing.md", "# Billing\n\nWe retry failed payments invoice 4411.\n"),
            ]
        )
        env.store.maintain()
        env.store.documents.update(  # type: ignore[union-attr]
            where=f"path = {sql_quote(str(env.root / 'billing.md'))}",
            values={"doc_type": "invoice"},
        )
        skill_ctx.extras["privacy"] = PrivacyFilter(
            env.settings.privacy.model_copy(
                update={"never_send_doc_types": frozenset({"invoice"})}
            ),
            env.scope,
            env.store.doc_types_for,
        )
        cloud.inner.reply = ["Backoff [1]."]
        run = AskSkill(skill_ctx).prepare("how do we retry failed payments")
        list(run.deltas())
        sent = sent_text(cloud.inner)
        assert "retry failed payments with backoff" in sent
        assert "invoice 4411" not in sent
        assert run.result.withheld == 1


class TestCodeRouting:
    def test_code_chat_to_the_cloud_never_carries_private_files(
        self, env: Env, chat: Chat, cloud: CloudRig, skill_ctx: SkillContext
    ) -> None:
        # chat stays local, code_chat goes to the cloud: the privacy rule must follow the role
        # that really answers (code-heavy context), not the plain chat role.
        cloud.router._settings = cloud.router._settings.model_copy(
            update={"routing": {"chat": "local", "code_chat": "cloud"}}
        )
        skill_ctx.extras["privacy"] = PrivacyFilter(
            env.settings.privacy.model_copy(update={"never_send_globs": ("**/confidential/**",)}),
            env.scope,
        )
        env.indexer.index_paths(
            [
                write(env, "pay/retry.py", CODE_PUBLIC),
                write(env, "confidential/vault.py", CODE_PRIVATE),
            ]
        )
        env.store.maintain()
        cloud.inner.reply = ["Use backoff [1]."]
        run = AskSkill(skill_ctx).prepare("retry failed payment")
        list(run.deltas())
        assert run.result.role == "code_chat"
        sent = sent_text(cloud.inner)
        assert "backoff(charge)" in sent
        assert "vault(charge)" not in sent
        assert run.result.withheld == 1

    def test_the_route_decided_for_privacy_is_the_route_used_for_sending(
        self, env: Env, chat: Chat, cloud: CloudRig, skill_ctx: SkillContext
    ) -> None:
        env.indexer.index_paths([write(env, "payments.md", PUBLIC_NOTE)])
        env.store.maintain()
        cloud.router._settings = cloud.router._settings.model_copy(
            update={"routing": {"chat": "local"}}
        )
        run = AskSkill(skill_ctx).prepare("how do we retry failed payments")
        # the policy flips to cloud between the privacy decision and the send
        cloud.router._settings = cloud.router._settings.model_copy(
            update={"routing": {"chat": "cloud"}}
        )
        list(run.deltas())
        assert cloud.inner.sent == []  # still went to the local model that was approved
        assert chat.chat_calls()


class TestChat:
    def test_private_files_cannot_be_pinned_in_a_cloud_chat(
        self, cloud: CloudRig, skill_ctx: SkillContext, notes: dict[str, str]
    ) -> None:
        skill = ChatSkill(skill_ctx)
        session = skill.open_session("t", [notes["private"]])
        with pytest.raises(ChatBlockedError, match="private and cannot be sent"):
            skill.run(ChatInput(message="summarise", session=session))
        assert cloud.inner.sent == []

    def test_public_files_and_pasted_text_are_masked_and_sent(
        self, cloud: CloudRig, skill_ctx: SkillContext, notes: dict[str, str]
    ) -> None:
        skill = ChatSkill(skill_ctx)
        skill.run(
            ChatInput(message="summarise", pin=[notes["public"]], scratch="Passport No: K1234567")
        )
        sent = sent_text(cloud.inner)
        assert "retry failed payments" in sent
        assert "K1234567" not in sent
        assert "[PASSPORT REMOVED]" in sent
        assert "123-45-6789" not in sent

    def test_the_same_pin_is_fine_for_a_local_chat(
        self, chat: Chat, cloud: CloudRig, skill_ctx: SkillContext, notes: dict[str, str]
    ) -> None:
        cloud.router._settings = cloud.router._settings.model_copy(update={"routing": {}})
        ChatSkill(skill_ctx).run(ChatInput(message="summarise", pin=[notes["private"]]))
        assert chat.chat_calls()
        assert cloud.inner.sent == []

    def test_retrieval_fallback_skips_private_files_in_the_cloud(
        self, cloud: CloudRig, skill_ctx: SkillContext, notes: dict[str, str]
    ) -> None:
        ChatSkill(skill_ctx).run(ChatInput(message="how do we retry failed payments"))
        sent = sent_text(cloud.inner)
        assert "retry failed payments with backoff" in sent
        assert "secret strategy" not in sent


class TestPrivateJobDescription:
    def test_a_private_jd_file_cannot_be_scored_in_the_cloud(
        self, env: Env, chat: Chat, cloud: CloudRig, skill_ctx: SkillContext
    ) -> None:
        jd = write(env, ".claude/projects/demo/memory/jd.md", JD)
        skill = MatchSkill(skill_ctx)
        with pytest.raises(RuntimeError, match="private"):
            skill.prepare(MatchInput(jd_file=jd, top=1))
        assert cloud.inner.sent == []

    def test_the_same_file_is_fine_for_a_local_model(
        self, env: Env, chat: Chat, cloud: CloudRig, skill_ctx: SkillContext
    ) -> None:
        cloud.router._settings = cloud.router._settings.model_copy(update={"routing": {}})
        jd = write(env, ".claude/projects/demo/memory/jd.md", JD)
        run = MatchSkill(skill_ctx).prepare(MatchInput(jd_file=jd, top=1))
        assert run.jd_text.startswith("Senior backend engineer")


class TestMatch:
    @pytest.fixture
    def library(self, env: Env, chat: Chat) -> dict[str, str]:
        chat.client.chat_json_fn = faithful_model
        private_resume = resume("Pat Quinn", "Python", "AWS") + "\nSSN 987-65-4321\n"
        paths = {
            "a": write(
                env, "Resume_a.txt", resume("Jane Doe", "Python", "PostgreSQL", "Kubernetes")
            ),
            "b": write(
                env, "Resume_b.txt", resume("Sam Roe", "Python") + "\nPassport No: K1234567\n"
            ),
            "private": write(env, ".claude/memory/Resume_private.md", private_resume),
        }
        env.indexer.index_paths(list(paths.values()))
        env.store.maintain()
        return paths

    def cloud_judge(self, cloud: CloudRig) -> None:
        def judge(messages: list[Message]) -> Any:
            if messages[0].content.startswith("You extract"):
                return CHECKLIST
            doc = messages[-1].content.split("):\n", 1)[1].lower()
            rows = [
                {"id": rid, "status": "met", "evidence_quote": word}
                if word in doc and len(word) >= 8
                else {"id": rid, "status": "missing", "evidence_quote": ""}
                for rid, word in {1: "python", 2: "postgresql", 3: "kubernetes", 4: "aws"}.items()
            ]
            return {"results": rows, "seniority_fit": "fit", "summary": "cloud"}

        cloud.inner.json_fn = judge

    def pipeline(self, env: Env, chat: Chat, skill_ctx: SkillContext) -> MatchPipeline:
        loader = DocumentLoader(env.store, env.scope, lambda: env.extractors)
        return MatchPipeline(skill_ctx, chat.gateway, loader)

    def test_private_resumes_are_locked_and_scored_locally_while_others_go_to_the_cloud(
        self,
        env: Env,
        chat: Chat,
        cloud: CloudRig,
        skill_ctx: SkillContext,
        library: dict[str, str],
    ) -> None:
        self.cloud_judge(cloud)
        pipeline = self.pipeline(env, chat, skill_ctx)
        run = pipeline.start(JD)
        locked = {c.name: c.locked for c in run.candidates}
        assert locked == {"Resume_a.txt": False, "Resume_b.txt": False, "Resume_private.md": True}
        select_all(run.candidates)
        progress: list[str] = []
        pipeline.score(run, progress.append)
        cloud_text = sent_text(cloud.inner)
        assert "Resume (Resume_a.txt)" in cloud_text
        assert "Resume (Resume_b.txt)" in cloud_text
        assert "Resume (Resume_private.md)" not in cloud_text
        assert "987-65-4321" not in cloud_text
        local_prompts = json.dumps([c["messages"] for c in chat.chat_calls()])
        assert "Resume (Resume_private.md)" in local_prompts
        assert [s.candidate.name for s in run.scores] == [
            c.name for c in run.candidates if c.selected
        ]
        assert any("locally (private)" in line for line in progress)

    def test_ids_are_masked_in_every_cloud_request(
        self,
        env: Env,
        chat: Chat,
        cloud: CloudRig,
        skill_ctx: SkillContext,
        library: dict[str, str],
    ) -> None:
        self.cloud_judge(cloud)
        pipeline = self.pipeline(env, chat, skill_ctx)
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        cloud_text = sent_text(cloud.inner)
        assert "K1234567" not in cloud_text
        assert "[PASSPORT REMOVED]" in cloud_text
        assert cloud.provider.last_outbound is not None

    def test_the_footer_names_the_destination_and_estimates_the_cost(
        self,
        env: Env,
        chat: Chat,
        cloud: CloudRig,
        skill_ctx: SkillContext,
        library: dict[str, str],
    ) -> None:
        controller = MatchController(lambda: skill_ctx)
        run = controller.start(JD)
        select_all(run.candidates)
        footer = controller.footer()
        assert "☁ OpenRouter / vendor/judge" in footer
        assert "est. $" in footer
        assert "🔒 1 scored locally" in footer
        rows = {c.name: controller.candidate_row(c)[-1] for c in run.candidates}
        assert rows["Resume_private.md"] == "🔒 🖥 local"
        assert rows["Resume_a.txt"].startswith("☁")

    def test_a_local_run_has_no_price_and_stays_local(
        self, env: Env, chat: Chat, skill_ctx: SkillContext, library: dict[str, str]
    ) -> None:
        controller = MatchController(lambda: skill_ctx)
        run = controller.start(JD)
        select_all(run.candidates)
        footer = controller.footer()
        assert "est. $" not in footer
        assert "🖥 local" in footer

    def test_the_verdict_stays_local_when_a_locked_document_was_scored(
        self,
        env: Env,
        chat: Chat,
        cloud: CloudRig,
        skill_ctx: SkillContext,
        library: dict[str, str],
    ) -> None:
        self.cloud_judge(cloud)
        chat.client.chat_reply = ["Local verdict."]
        pipeline = self.pipeline(env, chat, skill_ctx)
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        before = len(cloud.inner.sent)
        assert "".join(pipeline.stream_verdict(run)) == "Local verdict."
        assert len(cloud.inner.sent) == before


def test_the_cloud_rig_fixture_is_available(cloud: CloudRig, tmp_path: Path) -> None:
    assert cloud.consent.granted
    assert tmp_path.exists()


class TestMatchConsent:
    library = TestMatch.library
    cloud_judge = TestMatch.cloud_judge

    def test_the_preview_is_what_scoring_then_sends_and_shows_the_shield(
        self,
        env: Env,
        chat: Chat,
        cloud: CloudRig,
        skill_ctx: SkillContext,
        library: dict[str, str],
    ) -> None:
        self.cloud_judge(cloud)
        controller = MatchController(lambda: skill_ctx)
        run = controller.start(JD)
        select_all(run.candidates)
        first = controller.cloud_preview("checklist")
        assert first is not None
        assert "Senior backend engineer" in first.text  # the job description is what leaves
        assert controller.cloud_preview("score") is None  # nothing to judge before a checklist
        controller.grant_cloud_consent()
        controller.checklist()
        preview = controller.cloud_preview("score")
        assert preview is not None
        assert "OpenRouter" in preview.destination
        assert "Resume_a.txt" in preview.text
        assert "Resume_private.md" not in preview.text  # locked: scored locally, never shown
        assert "K1234567" not in preview.text
        assert preview.shield.startswith("🛡")
        before = len(cloud.inner.sent)
        assert controller.cloud_preview("score") is not None
        assert len(cloud.inner.sent) == before  # building a preview sends nothing
        controller.score()
        scored = [m.content for batch in cloud.inner.sent for m in batch if "Resume (" in m.content]
        assert scored
        for text in scored:
            if "Resume (Resume_a.txt)" in text:
                assert text in preview.text  # the very text that was previewed

    def test_scoring_without_consent_is_refused_and_a_new_run_revokes_it(
        self,
        env: Env,
        chat: Chat,
        cloud: CloudRig,
        skill_ctx: SkillContext,
        library: dict[str, str],
    ) -> None:
        self.cloud_judge(cloud)
        cloud.consent.revoke()
        controller = MatchController(lambda: skill_ctx)
        run = controller.start(JD)
        select_all(run.candidates)
        with pytest.raises(ChatBlockedError, match="consent"):
            controller.checklist()
        controller.grant_cloud_consent()
        assert cloud.consent.granted
        controller.start(JD)  # a new Match run
        assert not cloud.consent.granted

    def test_local_scoring_needs_no_preview(
        self,
        env: Env,
        chat: Chat,
        cloud: CloudRig,
        skill_ctx: SkillContext,
        library: dict[str, str],
    ) -> None:
        cloud.router._settings = cloud.router._settings.model_copy(update={"routing": {}})
        controller = MatchController(lambda: skill_ctx)
        select_all(controller.start(JD).candidates)
        assert controller.cloud_preview() is None
