from collections.abc import Sequence
from pathlib import Path

import pytest

from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.providers.base import Message
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import PrivacySettings, ScopeSettings

RESUME = (
    "Jane Doe\njane.doe@example.com | +1 415 555 0132\n"
    "Passport No: K1234567\nSSN: 123-45-6789\nBuilt payment systems in Python (2019-2024).\n"
)


def make(**kw: object) -> PrivacyFilter:
    scope = ScopePolicy(ScopeSettings())
    return PrivacyFilter(PrivacySettings(**kw), scope)  # type: ignore[arg-type]


def request(*documents: tuple[str, str]) -> list[Message]:
    body = "\n\n".join(f"Resume ({name}):\n{text}" for name, text in documents)
    return [Message("system", "Judge the resume."), Message("user", body)]


class TestNeverSend:
    def test_secrets_memory_notes_and_doc_types_are_blocked(self) -> None:
        f = make(never_send_doc_types=frozenset({"invoice"}))
        home = Path.home()
        assert f.is_never_send(home / "proj" / ".env")
        assert f.is_never_send(home / ".claude" / "projects" / "x" / "memory" / "note.md")
        assert f.is_never_send(home / ".claude" / "memory" / "note.md")
        assert f.is_never_send(home / "docs" / "bill.pdf", doc_type="invoice")
        assert not f.is_never_send(home / "docs" / "resume.pdf", doc_type="resume")
        assert not f.is_never_send(home / ".claude" / "plans" / "plan.md")

    def test_stored_doc_type_is_looked_up_when_the_caller_does_not_know_it(self) -> None:
        stored = {"D:/docs/bill.pdf": "invoice", "D:/docs/cv.pdf": "resume"}
        lookups: list[list[str]] = []

        def lookup(paths: Sequence[str]) -> dict[str, str]:
            lookups.append(list(paths))
            return {p: stored[p] for p in paths if p in stored}

        f = PrivacyFilter(
            PrivacySettings(never_send_doc_types=frozenset({"invoice"})),
            ScopePolicy(ScopeSettings()),
            lookup,
        )
        assert f.is_never_send("D:/docs/bill.pdf")
        assert not f.is_never_send("D:/docs/cv.pdf")
        assert not f.is_never_send("D:/docs/unindexed.pdf")

    def test_no_lookup_is_made_when_no_type_is_blocked(self) -> None:
        def lookup(paths: Sequence[str]) -> dict[str, str]:
            raise AssertionError("the index must not be queried for nothing")

        f = PrivacyFilter(PrivacySettings(), ScopePolicy(ScopeSettings()), lookup)
        assert not f.is_never_send("D:/docs/bill.pdf")

    def test_custom_globs(self) -> None:
        f = make(never_send_globs=("**/private/**",))
        assert f.is_never_send(Path("D:/work/private/cv.pdf"))
        assert not f.is_never_send(Path("D:/work/public/cv.pdf"))
        assert not f.is_never_send(Path.home() / ".claude" / "memory" / "n.md")


class TestPrepare:
    def test_ids_are_always_masked_and_the_original_is_untouched(self) -> None:
        original = request(("Resume_v2.pdf", RESUME))
        outbound = make().prepare(original)
        body = outbound.messages[1].content
        assert "[PASSPORT REMOVED]" in body
        assert "[SSN REMOVED]" in body
        assert "K1234567" not in body
        assert "123-45-6789" not in body
        assert "2019-2024" in body
        assert original[1].content.count("K1234567") == 1
        assert outbound.messages[0].content == "Judge the resume."

    def test_personal_details_are_kept_by_default(self) -> None:
        body = make().prepare(request(("r.pdf", RESUME))).messages[1].content
        assert "jane.doe@example.com" in body
        assert "415 555 0132" in body
        assert "Jane Doe" in body

    def test_personal_redaction_is_optional_reversible_and_consistent(self) -> None:
        f = make(redact_personal=True)
        outbound = f.prepare(request(("a.pdf", RESUME), ("b.pdf", RESUME)))
        body = outbound.messages[1].content
        assert "jane.doe@example.com" not in body
        assert body.count("[EMAIL_1]") == 2
        assert outbound.personal_redacted >= 3
        answer = "Contact [NAME_1] at [EMAIL_1]."
        assert outbound.restore(answer) == "Contact Jane Doe at jane.doe@example.com."
        assert "[PASSPORT REMOVED]" in body  # IDs are masked either way

    def test_findings_name_the_document_they_came_from(self) -> None:
        clean = "Built systems in Python."
        outbound = make().prepare(request(("old.docx", clean), ("Resume_v2.pdf", RESUME)))
        assert {(i.finding.kind, i.source) for i in outbound.findings} == {
            ("passport", "Resume_v2.pdf"),
            ("ssn", "Resume_v2.pdf"),
        }

    def test_sources_are_recognised_in_other_prompt_layouts(self) -> None:
        pinned = Message("system", "=== resume.pdf ===\nPassport No: K1234567")
        sourced = Message("user", "[1] a.py\nSSN: 123-45-6789")
        outbound = make().prepare([pinned, sourced])
        assert [i.source for i in outbound.findings] == ["resume.pdf", "a.py"]

    def test_clean_requests_pass_through_unchanged(self) -> None:
        messages = request(("r.pdf", "Built systems in Python."))
        outbound = make().prepare(messages)
        assert outbound.messages == messages
        assert outbound.findings == []
        assert outbound.placeholders == {}
        assert outbound.restore("nothing to restore") == "nothing to restore"


class TestTransparency:
    def test_shield_note_lists_what_will_be_masked(self) -> None:
        f = make()
        outbound = f.prepare(request(("Resume_v2.pdf", RESUME), ("old.docx", "DL# S1234567")))
        note = f.shield_note(outbound)
        assert note.startswith("🛡 3 sensitive items will be masked: ")
        assert "Passport (Resume_v2.pdf)" in note
        assert "Drivers License (old.docx)" in note
        assert f.shield_note(f.prepare(request(("r.pdf", "clean")))) == ""

    def test_single_item_and_repeats(self) -> None:
        f = make()
        one = f.prepare(request(("a.pdf", "Passport No: K1234567")))
        assert f.shield_note(one) == "🛡 1 sensitive item will be masked: Passport (a.pdf)"
        twice = f.prepare(request(("a.pdf", "Passport No: K1234567 and passport no: M7654321")))
        assert "Passport (a.pdf) x2" in f.shield_note(twice)

    def test_preview_is_exactly_what_is_sent(self) -> None:
        f = make()
        outbound = f.prepare(request(("r.pdf", RESUME)))
        preview = f.preview(outbound)
        for message in outbound.messages:
            assert message.content in preview
        assert "K1234567" not in preview
        assert "--- system ---" in preview

    def test_badge_counts_excerpts_and_tokens(self) -> None:
        f = make()
        outbound = f.prepare(request(("r.pdf", "x" * 4000)))
        assert f.badge(outbound, "OpenRouter / gpt-x", excerpts=6).startswith(
            "☁ Sending 6 excerpts (≈1.0k tokens) to OpenRouter / gpt-x"
        )
        assert f.badge(f.prepare([Message("user", "hi")]), "p / m").startswith(
            "☁ Sending 1 excerpt (≈"
        )

    @pytest.mark.parametrize("empty", [[], [Message("user", "")]])
    def test_empty_requests_are_safe(self, empty: list[Message]) -> None:
        outbound = make().prepare(empty)
        assert outbound.findings == []
        assert outbound.tokens >= 0


class TestNamesAcrossTurns:
    @staticmethod
    def make() -> PrivacyFilter:
        settings = PrivacySettings(redact_personal=True)
        return PrivacyFilter(settings, ScopePolicy(ScopeSettings()))

    def test_a_name_in_an_earlier_message_is_masked_even_if_found_in_a_later_one(self) -> None:
        outbound = self.make().prepare(
            [
                Message("user", "How strong is Priya Nair for the role?"),
                Message("system", "Name: Priya Nair\nSkills: Python"),
            ]
        )
        assert all("Priya Nair" not in m.content for m in outbound.messages)

    def test_later_turns_still_mask_a_name_that_was_restored_into_the_history(self) -> None:
        policy = self.make()
        first = policy.prepare([Message("system", "Name: Priya Nair\nSkills: Python")])
        answer = first.restore("[NAME_1] is a strong fit.")
        assert "Priya Nair" in answer  # the user sees the real name
        # turn two: no resume block this time, but the history now carries the real name
        second = policy.prepare(
            [Message("assistant", answer), Message("user", "What about Priya Nair's Go skills?")]
        )
        assert all("Priya Nair" not in m.content for m in second.messages)

    def test_a_new_chat_can_start_clean(self) -> None:
        policy = self.make()
        policy.prepare([Message("system", "Name: Priya Nair\nx")])
        policy.forget_names()
        fresh = policy.prepare([Message("user", "Is Priya Nair a good fit?")])
        assert "Priya Nair" in fresh.messages[0].content  # nothing remembered from before


def test_the_personal_details_choice_can_be_changed_for_the_session() -> None:
    from vector_embed.core.privacy.policy import PrivacyFilter
    from vector_embed.core.scope import ScopePolicy
    from vector_embed.core.settings import PrivacySettings, ScopeSettings

    privacy = PrivacyFilter(PrivacySettings(), ScopePolicy(ScopeSettings()))
    message = [Message("user", "Jane Doe\njane@example.com")]
    assert not privacy.redacts_personal
    assert "jane@example.com" in privacy.prepare(message).messages[0].content
    privacy.set_redact_personal(True)
    assert privacy.redacts_personal
    assert "jane@example.com" not in privacy.prepare(message).messages[0].content
