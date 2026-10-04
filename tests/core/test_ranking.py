import pytest

from localdoc_finder.core.ranking import (
    FileEvidence,
    NameMatch,
    file_score,
    name_match,
    query_terms,
    rank_files,
    term_coverage,
    vouched_for,
)
from localdoc_finder.core.retrieval import scaled_similarity
from localdoc_finder.core.settings import SearchSettings

CFG = SearchSettings()
FLOOR = 0.5


class TestQueryTerms:
    def test_stop_words_and_short_words_are_dropped(self) -> None:
        assert query_terms("Where is the list of my PDF files?") == ["list", "pdf", "files"]

    def test_identifiers_stay_whole(self) -> None:
        assert query_terms("charge_card retry") == ["charge_card", "retry"]


class TestTermCoverage:
    def test_counts_query_terms_present_as_words(self) -> None:
        terms = query_terms("retry failed payments backoff")
        text = "def retry_failed(order):\n    # retried the payment with exponential backoff"
        assert term_coverage(terms, text) == pytest.approx(3 / 4)  # retry_failed is one word

    def test_light_stemming_matches_inflections(self) -> None:
        assert term_coverage(["retried", "payments"], "retry the payment") == 1.0
        assert term_coverage(["migrations"], "data migration") == 1.0
        assert term_coverage(["running"], "runs") == 0.0  # deliberately light

    def test_no_terms_cover_nothing(self) -> None:
        assert term_coverage([], "anything") == 0.0


class TestNameMatch:
    @pytest.mark.parametrize(
        ("path", "query", "share"),
        [
            ("D:/src/catalog.py", "log", 0.0),  # a part of a word is not a word
            ("D:/src/log_reader.py", "log", 1.0),
            ("D:/src/retryPayments.py", "retry payment", 1.0),  # camelCase and plural
            ("D:/src/retryPayments.py", "retry invoices", 0.5),
            ("D:/a/notes.txt", "txt", 0.0),  # the extension alone never counts
            ("D:/a/notes.txt", "notes budget", 0.5),
            ("D:/a/logs.txt", "log", 0.0),  # too short to treat as a plural
            ("D:/a/notes.txt", "the of", 0.0),  # stop words only
        ],
    )
    def test_whole_words_score_by_share(self, path: str, query: str, share: float) -> None:
        assert name_match(path, query).share == pytest.approx(share)

    @pytest.mark.parametrize("query", ["notes.txt", "notes", " NOTES "])
    def test_the_query_naming_the_file_is_exact(self, query: str) -> None:
        assert name_match("D:/a/notes.txt", query) == NameMatch(1.0, exact=True)


class TestFusion:
    @pytest.mark.parametrize(
        ("similarity", "expected"),
        [(None, 0.0), (0.3, 0.0), (0.5, 0.0), (0.75, 0.5), (1.0, 1.0), (1.2, 1.0)],
    )
    def test_similarity_is_scaled_from_the_floor(
        self, similarity: float | None, expected: float
    ) -> None:
        assert scaled_similarity(similarity, FLOOR) == pytest.approx(expected)

    def test_a_floor_of_one_scores_nothing(self) -> None:
        assert scaled_similarity(1.0, 1.0) == 0.0


def evidence(path: str = "a.py", **kw: object) -> FileEvidence:
    return FileEvidence(path, **kw)  # type: ignore[arg-type]


class TestFileScore:
    def test_a_second_chunk_adds_a_small_bonus(self) -> None:
        one = evidence()
        one.add_chunk(0.6, 0.8, coverage=None)
        two = evidence()
        two.add_chunk(0.6, 0.8, coverage=None)
        two.add_chunk(0.4, 0.7, coverage=1.0)
        assert file_score(two, CFG) == pytest.approx(0.6 + CFG.multi_hit_bonus * 0.4)
        assert file_score(one, CFG) == pytest.approx(0.6)
        assert two.coverage == 1.0 and two.similarity == 0.8
        assert one.coverage == 0.0  # BM25 never matched it

    def test_chunks_arriving_out_of_order_keep_the_best_two(self) -> None:
        found = evidence()
        for fused in (0.2, 0.9, 0.5):
            found.add_chunk(fused, None, coverage=None)
        assert (found.best, found.second) == (0.9, 0.5)
        assert found.similarity is None

    def test_file_names_and_the_current_project_add_up(self) -> None:
        named = evidence(name=NameMatch(0.5))
        assert file_score(named, CFG) == pytest.approx(CFG.filename_weight * 0.5)
        exact = evidence(name=NameMatch(1.0, exact=True))
        assert file_score(exact, CFG) == pytest.approx(
            CFG.filename_weight + CFG.filename_exact_bonus
        )
        local = evidence(best=0.5, in_current_project=True)
        assert file_score(local, CFG) == pytest.approx(0.5 * CFG.current_project_boost)


class TestTiers:
    def test_evidence_is_similarity_over_the_floor_or_enough_of_the_query(self) -> None:
        cover = CFG.min_term_coverage
        assert not vouched_for(evidence(similarity=0.49), FLOOR, cover)
        assert vouched_for(evidence(similarity=0.5), FLOOR, cover)
        assert vouched_for(evidence(coverage=0.5), FLOOR, cover)
        assert vouched_for(evidence(name=NameMatch(0.5)), FLOOR, cover)

    def test_one_incidental_word_is_not_evidence(self) -> None:
        # "chip" in a UI file for "chocolate chip cookie recipe brown butter"
        cover = CFG.min_term_coverage
        assert not vouched_for(evidence(coverage=1 / 6, similarity=0.2), FLOOR, cover)
        assert not vouched_for(evidence(name=NameMatch(0.25)), FLOOR, cover)  # doctor.py

    def test_a_result_nothing_vouches_for_is_weak_however_it_ranks(self) -> None:
        junk = evidence("scan_0042.pdf", similarity=0.45)  # below the floor, the nearest anyway
        ranked = rank_files([junk], CFG, FLOOR)
        assert ranked[0].weak
        assert ranked[0].relevance == 0

    def test_results_far_behind_the_best_are_weak_and_sorted_last(self) -> None:
        best = evidence("best.py", best=0.9, similarity=0.9, coverage=1.0)
        close = evidence("close.py", best=0.6, similarity=0.8)
        far = evidence("far.py", best=0.3, similarity=0.6)  # under half of the best
        junk = evidence("junk.pdf", best=0.0, similarity=0.4)
        ranked = rank_files([far, junk, close, best], CFG, FLOOR)
        assert [(r.evidence.path, r.weak) for r in ranked] == [
            ("best.py", False),
            ("close.py", False),
            ("far.py", True),
            ("junk.pdf", True),
        ]
        assert [r.relevance for r in ranked] == [90, 60, 30, 0]

    def test_the_tier_sorts_before_the_score(self) -> None:
        strong = evidence("doc.md", best=0.2, coverage=1.0)
        # Inconsistent on purpose (a score with nothing vouching for it): the tier still wins.
        unvouched = evidence("other.md", best=0.5, similarity=0.1)
        ranked = rank_files([unvouched, strong], CFG, FLOOR)
        assert [(r.evidence.path, r.weak) for r in ranked] == [
            ("doc.md", False),
            ("other.md", True),
        ]

    def test_relevance_never_rises_down_the_list(self) -> None:
        strong = evidence("dl600.pdf", best=0.25, similarity=0.6)
        unvouched = evidence("agreement.pdf", best=0.30, similarity=0.4, coverage=0.25)
        ranked = rank_files([unvouched, strong], CFG, FLOOR)
        assert [(r.evidence.path, r.weak, r.relevance) for r in ranked] == [
            ("dl600.pdf", False, 25),
            ("agreement.pdf", True, 25),
        ]
        assert ranked[1].score == pytest.approx(0.30)  # only the shown number is capped

    def test_relevance_is_capped_at_100(self) -> None:
        exact = evidence(best=0.9, coverage=1.0, name=NameMatch(1.0, exact=True))
        assert rank_files([exact], CFG, FLOOR)[0].relevance == 100

    def test_with_no_evidence_anywhere_everything_is_weak(self) -> None:
        ranked = rank_files([evidence("a"), evidence("b", similarity=0.1)], CFG, FLOOR)
        assert all(r.weak for r in ranked)
