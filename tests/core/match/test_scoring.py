import time

import pytest

from localdoc_finder.core.match.scoring import (
    Requirement,
    RowResult,
    compute_score,
    key_terms,
    normalise,
    quote_in_text,
    verify_rows,
)
from localdoc_finder.core.settings import MatchSettings

SETTINGS = MatchSettings()
RESUME = """
Jane Doe
Work Experience
Senior Backend Engineer, Acme Corp (2019-2024)
- Built payment systems in Python and PostgreSQL serving 2M users
- Led migration of services to Kubernetes on AWS
Skills: Python, SQL, Docker
"""


def req(id: int, kind: str = "must", **kw: object) -> Requirement:
    return Requirement(id=id, text=f"requirement {id}", kind=kind, **kw)  # type: ignore[arg-type]


class TestQuotes:
    def test_exact_quote_is_found_ignoring_case_space_and_punctuation(self) -> None:
        assert quote_in_text("built payment systems in PYTHON and postgresql", RESUME)
        assert quote_in_text("Led   migration of services\nto Kubernetes", RESUME)
        assert quote_in_text("Skills - Python, SQL; Docker!", RESUME)

    def test_small_edits_still_match_fuzzily(self) -> None:
        assert quote_in_text(
            "Built payments systems in Python and Postgres SQL serving 2M user", RESUME
        )

    def test_fabricated_quotes_do_not_match(self) -> None:
        assert not quote_in_text("Managed a team of twelve data scientists at Google", RESUME)
        assert not quote_in_text("Certified Kubernetes Administrator since 2021", RESUME)

    def test_empty_or_tiny_quotes_never_verify(self) -> None:
        assert not quote_in_text("", RESUME)
        assert not quote_in_text("D", RESUME)
        assert not quote_in_text("--", RESUME)

    def test_short_quotes_verify_as_exact_whole_words(self) -> None:
        text = RESUME + "Languages: Go, C++, C#\n"
        assert quote_in_text("Python", text)
        assert quote_in_text("AWS", text)
        assert quote_in_text("Go", text)
        assert quote_in_text("c++", text)
        assert quote_in_text("C#", text)
        assert not quote_in_text("Pyth", text)  # part of a word
        assert not quote_in_text("Rust", text)
        assert not quote_in_text("C", text)  # "C++" is not "C"

    def test_a_swapped_technology_is_not_close_enough(self) -> None:
        swapped = "Built payment systems in Python and MySQL serving 2M users"
        assert quote_in_text(swapped, RESUME)  # fuzzily close without the requirement's terms
        assert not quote_in_text(swapped, RESUME, terms=key_terms("MySQL database experience"))

    def test_terms_the_quote_does_not_use_are_not_required(self) -> None:
        quote = "Built payments systems in Python and Postgres SQL serving 2M user"
        assert quote_in_text(quote, RESUME, terms=key_terms("Relational databases, e.g. Oracle"))
        assert quote_in_text(quote, RESUME, terms=key_terms("3+ years of Python"))

    def test_key_terms_drop_filler_words(self) -> None:
        assert key_terms("5+ years of experience with Kubernetes and AWS") == {
            "5",
            "kubernetes",
            "aws",
        }

    def test_long_documents_are_checked_quickly(self) -> None:
        filler = " ".join(f"word{i} lorem ipsum dolor" for i in range(20_000))
        text = filler + RESUME
        started = time.perf_counter()
        assert quote_in_text("Led migration of service to Kubernetes on AWS", text)
        assert not quote_in_text("Designed a compiler for a new functional language", text)
        assert time.perf_counter() - started < 5

    def test_threshold_is_configurable(self) -> None:
        loose = "Built payment systems in Ruby and MySQL serving 2M users"
        assert not quote_in_text(loose, RESUME, threshold=0.99)
        assert quote_in_text(loose, RESUME, threshold=0.6)

    def test_normalise(self) -> None:
        assert normalise("  Hello,\n WORLD!! ") == "hello world"


class TestVerifyRows:
    def test_unsupported_matches_become_unverified(self) -> None:
        rows = [
            RowResult(1, "met", "Built payment systems in Python and PostgreSQL"),
            RowResult(2, "met", "Ten years of Rust at NASA"),
            RowResult(3, "partial", "Led migration of services to Kubernetes"),
            RowResult(4, "partial", "Some Haskell"),
            RowResult(5, "missing", ""),
        ]
        out = verify_rows(rows, RESUME, SETTINGS)
        assert [r.status for r in out] == ["met", "unverified", "partial", "unverified", "missing"]
        assert out[1].evidence == "Ten years of Rust at NASA"
        assert not out[1].verified
        assert out[0].verified

    def test_requirement_terms_are_used_when_given(self) -> None:
        reqs = [Requirement(id=1, text="MySQL"), Requirement(id=2, text="PostgreSQL")]
        quote = "Built payment systems in Python and MySQL serving 2M users"
        rows = [RowResult(1, "met", quote), RowResult(2, "met", quote)]
        out = verify_rows(rows, RESUME, SETTINGS, reqs)
        assert [r.status for r in out] == ["unverified", "met"]

    def test_missing_rows_need_no_evidence(self) -> None:
        assert verify_rows([RowResult(1, "missing")], RESUME, SETTINGS)[0].status == "missing"


class TestScore:
    def test_must_haves_count_double_and_partial_counts_half(self) -> None:
        reqs = [req(1), req(2), req(3, "nice")]
        rows = [RowResult(1, "met"), RowResult(2, "partial"), RowResult(3, "met")]
        result = compute_score(reqs, rows, SETTINGS)
        # earned: 2*1 + 2*0.5 + 1*1 = 4 of 5 possible
        assert result.score == 80
        assert (result.met_must, result.total_must) == (1, 2)
        assert (result.met_nice, result.total_nice) == (1, 1)
        assert result.summary_line == "1/2 must-haves met, 1/1 nice-to-haves"

    def test_missing_rows_default_to_missing(self) -> None:
        result = compute_score([req(1), req(2)], [RowResult(1, "met")], SETTINGS)
        assert result.score == 50

    def test_unverified_earns_nothing_and_is_counted(self) -> None:
        result = compute_score(
            [req(1), req(2)], [RowResult(1, "met"), RowResult(2, "unverified")], SETTINGS
        )
        assert result.score == 50
        assert result.unverified == 1
        assert result.summary_line.endswith("1 unverified")

    def test_disabled_requirements_are_ignored_and_weights_apply(self) -> None:
        reqs = [req(1), req(2, enabled=False), req(3, weight=3.0)]
        rows = [RowResult(1, "met"), RowResult(2, "missing"), RowResult(3, "missing")]
        result = compute_score(reqs, rows, SETTINGS)
        assert result.score == round(100 * 2 / (2 + 6))
        assert result.total_must == 2
        assert {r.requirement_id for r in result.rows} == {1, 3}

    def test_no_enabled_requirements_scores_zero(self) -> None:
        assert compute_score([req(1, enabled=False)], [], SETTINGS).score == 0
        assert compute_score([], [], SETTINGS).summary_line == "0/0 must-haves met"

    def test_score_is_reproducible_and_order_independent(self) -> None:
        reqs = [req(i, "must" if i % 2 else "nice") for i in range(1, 8)]
        rows = [RowResult(i, ["met", "partial", "missing"][i % 3]) for i in range(1, 8)]
        first = compute_score(reqs, rows, SETTINGS)
        second = compute_score(list(reversed(reqs)), list(reversed(rows)), SETTINGS)
        assert first.score == second.score

    def test_custom_weights(self) -> None:
        settings = MatchSettings(must_weight=3.0, nice_weight=1.0)
        result = compute_score(
            [req(1), req(2, "nice")], [RowResult(1, "met"), RowResult(2, "missing")], settings
        )
        assert result.score == 75

    @pytest.mark.parametrize("kind", ["must", "nice"])
    def test_requirement_defaults(self, kind: str) -> None:
        r = req(1, kind)
        assert r.enabled and r.weight == 1.0 and r.years is None
