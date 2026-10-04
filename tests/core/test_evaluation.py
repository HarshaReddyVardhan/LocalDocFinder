import math
from pathlib import Path

import pytest
from PIL import Image
from tests.core.fakes import FakeEmbedder

from localdoc_finder.core import evaluation as ev
from localdoc_finder.core.evaluation import (
    EvalQuery,
    EvalSpec,
    EvaluationError,
    Evaluator,
    LegScore,
    ModelResult,
    format_calibration,
    format_results,
    is_false_positive,
    load_spec,
    ndcg_at,
    percentiles,
    precision_at,
    rank_of,
    score_leg,
)
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.settings import ScopeSettings, Settings


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "payments.py").write_text(
        "def retry_failed_payments(order):\n"
        "    # retry charging the credit card after a failure\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (root / "gardening.md").write_text(
        "# Gardening\n\nWater the tomato plants every morning.\n", encoding="utf-8"
    )
    (root / "answers.md").write_text(
        "# Answer key\n\nretry failed payments -> payments.py\n", encoding="utf-8"
    )
    return root


def settings_and_scope() -> tuple[Settings, ScopePolicy]:
    scope_settings = ScopeSettings(
        blocked_dirs=ScopeSettings().blocked_dirs - {"appdata"}, file_types="everything"
    )
    return Settings(scope=scope_settings), ScopePolicy(scope_settings)


def make_evaluator(tmp_path: Path, factory=None, exclude=()):  # type: ignore[no-untyped-def]
    settings, scope = settings_and_scope()
    return Evaluator(
        settings,
        scope,
        factory or (lambda _model: FakeEmbedder()),
        tmp_path / "eval",
        exclude=exclude,
    )


class TestRank:
    def test_first_matching_result_wins_and_separators_are_equivalent(self) -> None:
        paths = ["D:\\a\\x.py", "D:\\a\\core\\scope.py", "D:\\a\\y.py"]
        assert rank_of(paths, ["core/scope.py"]) == 2
        assert rank_of(paths, ["CORE\\SCOPE.PY", "y.py"]) == 2
        assert rank_of(paths, ["nope"]) == 0

    def test_only_the_top_k_count(self) -> None:
        paths = [f"f{i}.py" for i in range(20)]
        assert rank_of(paths, ["f9.py"]) == 10
        assert rank_of(paths, ["f10.py"]) == 0


class TestMetrics:
    def test_ndcg_rewards_early_hits_and_counts_each_pattern_once(self) -> None:
        assert ndcg_at(["a.py", "x.py"], ["a.py"]) == pytest.approx(1.0)
        assert ndcg_at(["x.py", "a.py"], ["a.py"]) == pytest.approx(1 / math.log2(3))
        assert ndcg_at(["a.py", "sub/a.py"], ["a.py"]) == pytest.approx(1.0)  # not above 1
        assert ndcg_at(["b.py", "a.py"], ["a.py", "b.py"]) == pytest.approx(1.0)
        assert ndcg_at(["x.py"], ["a.py"]) == 0.0
        assert ndcg_at(["a.py"], []) == 0.0

    def test_ndcg_ignores_results_past_k(self) -> None:
        paths = [f"f{i}.py" for i in range(12)]
        assert ndcg_at(paths, ["f11.py"]) == 0.0

    def test_precision_counts_relevant_slots_in_the_top_five(self) -> None:
        assert precision_at(["a.py", "b.py", "x", "y", "z", "a2.py"], ["a", "b.py"]) == 0.4

    def test_a_query_nothing_answers_is_a_false_positive_when_anything_returns(self) -> None:
        none = EvalQuery("cookie recipe", expect_none=True)
        assert is_false_positive(["x.py"], none)
        assert not is_false_positive([], none)
        assert not is_false_positive(["x.py"], none, judge_none=False)

    def test_a_forbidden_path_in_the_top_five_is_a_false_positive(self) -> None:
        query = EvalQuery("rent", ("lease.pdf",), ("scan_0042.pdf",))
        assert is_false_positive(["lease.pdf", "D:\\docs\\SCAN_0042.pdf"], query)
        assert not is_false_positive(["lease.pdf", "a", "b", "c", "d", "scan_0042.pdf"], query)

    def test_score_leg_averages_positives_and_rates_negatives(self) -> None:
        runs = [
            (EvalQuery("hit", ("a.py",)), ["a.py"]),
            (EvalQuery("miss", ("b.py",), ("junk",)), ["junk.pdf"]),
            (EvalQuery("nothing", expect_none=True), ["a.py"]),
            (EvalQuery("clean", expect_none=True), []),
        ]
        leg = score_leg(runs)
        assert leg.recall == 0.5
        assert leg.mrr == 0.5
        assert leg.ndcg == pytest.approx(0.5)
        assert leg.p_at_5 == pytest.approx(0.1)
        assert leg.missed == ["miss"]
        assert leg.false_positives == ["miss", "nothing"]
        assert leg.fp_rate == pytest.approx(2 / 3)

    def test_the_vector_leg_is_not_judged_on_queries_nothing_answers(self) -> None:
        runs = [(EvalQuery("nothing", expect_none=True), ["a.py"])]
        assert score_leg(runs, judge_none=False) == LegScore()

    def test_percentiles(self) -> None:
        assert percentiles([]) == []
        assert percentiles([0.0, 1.0])[2] == pytest.approx(0.5)


class TestSpec:
    def write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "queries.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_relative_corpus_models_and_queries(self, tmp_path: Path) -> None:
        spec = load_spec(
            self.write(
                tmp_path,
                "corpus: [../src]\nmodels: [a, b]\nqueries:\n  - q: hello\n    expect: [x.py]\n",
            )
        )
        assert spec.corpus == ((tmp_path / "../src").resolve(),)
        assert spec.models == ("a", "b")
        assert spec.queries == (EvalQuery("hello", ("x.py",)),)

    def test_loads_negative_queries(self, tmp_path: Path) -> None:
        spec = load_spec(
            self.write(
                tmp_path,
                "corpus: [x]\nqueries:\n"
                "  - q: rent\n    expect: [lease.pdf]\n    not: [scan.pdf]\n"
                "  - q: cookies\n    expect_none: true\n",
            )
        )
        assert spec.queries == (
            EvalQuery("rent", ("lease.pdf",), ("scan.pdf",)),
            EvalQuery("cookies", expect_none=True),
        )
        assert [q.negative for q in spec.queries] == [True, True]
        assert [q.positive for q in spec.queries] == [True, False]

    def test_absolute_corpus_paths_are_kept(self, tmp_path: Path) -> None:
        absolute = tmp_path / "abs"
        spec = load_spec(
            self.write(
                tmp_path,
                f"corpus: ['{absolute.as_posix()}']\nqueries:\n  - q: a\n    expect: [b]\n",
            )
        )
        assert spec.corpus == (absolute,)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("- a list\n", "must be a mapping"),
            ("corpus: [x]\nqueries: []\n", "no queries"),
            ("corpus: [x]\nqueries:\n  - q: a\n", "either 'expect' or 'expect_none'"),
            (
                "corpus: [x]\nqueries:\n  - q: a\n    expect: [b]\n    expect_none: true\n",
                "either 'expect' or 'expect_none'",
            ),
            ("corpus: [x]\nqueries:\n  - expect: [b]\n", "needs 'q'"),
            ("corpus: [x]\nqueries:\n  - q: a\n    expect: b.py\n", "expected a list"),
            ("queries:\n  - q: a\n    expect: [b]\n", "no corpus"),
            ("corpus: [x\n", "cannot read"),
        ],
    )
    def test_invalid_specs(self, tmp_path: Path, text: str, message: str) -> None:
        with pytest.raises(EvaluationError, match=message):
            load_spec(self.write(tmp_path, text))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="cannot read"):
            load_spec(tmp_path / "nope.yaml")


class TestEvaluator:
    def spec(self, corpus: Path) -> EvalSpec:
        return EvalSpec(
            (corpus,),
            ("fake",),
            (
                EvalQuery("retry failed payments after a credit card failure", ("payments.py",)),
                EvalQuery("water the tomato plants", ("gardening.md",)),
                EvalQuery("something nobody wrote", ("does-not-exist.py",)),
                EvalQuery("how to bake sourdough bread", expect_none=True),
                EvalQuery("tomato plants", ("gardening.md",), ("payments.py",)),
            ),
        )

    def test_scores_recall_precision_and_speed(self, tmp_path: Path, corpus: Path) -> None:
        result = make_evaluator(tmp_path).evaluate("fake", self.spec(corpus))
        assert result.error == ""
        assert result.files == 3
        assert result.full.recall == pytest.approx(3 / 4)
        assert result.full.mrr > 0.6
        assert 0 < result.full.ndcg <= result.full.recall
        assert result.full.p_at_5 > 0
        assert result.full.missed == ["something nobody wrote"]
        # Without a relevance floor, search always returns something.
        assert "how to bake sourdough bread" in result.full.false_positives
        assert result.index_seconds >= 0
        assert result.p50_ms >= 0

    def test_the_vector_leg_is_scored_on_its_own(self, tmp_path: Path, corpus: Path) -> None:
        result = make_evaluator(tmp_path).evaluate("fake", self.spec(corpus))
        assert result.vector.recall == pytest.approx(3 / 4)
        assert "how to bake sourdough bread" not in result.vector.false_positives

    def test_similarities_are_collected_for_calibration(self, tmp_path: Path, corpus: Path) -> None:
        result = make_evaluator(tmp_path).evaluate("fake", self.spec(corpus))
        assert result.true_similarities
        assert result.negative_similarities
        assert all(-1.0 <= s <= 1.0 for s in result.true_similarities)
        assert min(result.true_similarities) > min(result.negative_similarities)

    def test_ocr_is_passed_to_the_extractors(self, tmp_path: Path, corpus: Path) -> None:
        class SeeingOcr:
            def available(self) -> bool:
                return True

            def ocr_image(self, image: object) -> str:
                return "a photo of tomato plants in the garden"

            def ocr_lines(self, image: object) -> list[str]:
                return [self.ocr_image(image)]

        Image.new("RGB", (400, 300), (40, 160, 60)).save(corpus / "IMG_1.jpg")
        settings, scope = settings_and_scope()
        evaluator = Evaluator(
            settings, scope, lambda _m: FakeEmbedder(), tmp_path / "eval", ocr=SeeingOcr()
        )
        spec = EvalSpec((corpus,), ("fake",), (EvalQuery("tomato garden photo", ("IMG_1",)),))
        assert evaluator.evaluate("fake", spec).full.recall == 1.0

    def test_the_answer_key_is_excluded_from_the_corpus(self, tmp_path: Path, corpus: Path) -> None:
        evaluator = make_evaluator(tmp_path, exclude=[corpus / "answers.md"])
        assert evaluator.evaluate("fake", self.spec(corpus)).files == 2

    def test_reusing_an_index_skips_reindexing(self, tmp_path: Path, corpus: Path) -> None:
        embedder = FakeEmbedder()
        evaluator = make_evaluator(tmp_path, factory=lambda _m: embedder)
        evaluator.evaluate("fake", self.spec(corpus))
        calls = len(embedder.calls)
        again = evaluator.evaluate("fake", self.spec(corpus), reuse=True)
        assert again.files == 0
        # Only query embeddings: five queries, each through the full pipeline and the vector leg.
        assert len(embedder.calls) - calls == 10

    def test_a_fresh_run_rebuilds_the_index(self, tmp_path: Path, corpus: Path) -> None:
        evaluator = make_evaluator(tmp_path)
        evaluator.evaluate("fake", self.spec(corpus))
        assert evaluator.evaluate("fake", self.spec(corpus)).files == 3

    def test_a_broken_model_is_reported_not_raised(self, tmp_path: Path, corpus: Path) -> None:
        def factory(model: str) -> FakeEmbedder:
            if model == "bad":
                raise RuntimeError("model not pulled")
            return FakeEmbedder()

        evaluator = make_evaluator(tmp_path, factory=factory)
        bad = evaluator.evaluate("bad", self.spec(corpus))
        assert bad.error == "RuntimeError: model not pulled"
        assert evaluator.evaluate("good", self.spec(corpus)).error == ""

    def test_model_names_become_safe_directory_names(self, tmp_path: Path, corpus: Path) -> None:
        make_evaluator(tmp_path).evaluate("qwen3-embedding:0.6b", self.spec(corpus))
        assert (tmp_path / "eval" / "qwen3-embedding_0.6b").is_dir()


def test_format_results_sorts_by_quality_and_lists_failures() -> None:
    table = format_results(
        [
            ModelResult(
                "weak",
                full=LegScore(recall=0.5, mrr=0.2, ndcg=0.3, missed=["q1"]),
                files=10,
                index_seconds=3.0,
                p50_ms=40,
            ),
            ModelResult(
                "strong",
                full=LegScore(recall=0.9, mrr=0.7, ndcg=0.8, false_positives=["junk q"]),
                vector=LegScore(recall=0.8),
                files=10,
                index_seconds=5.0,
                p50_ms=60,
            ),
            ModelResult("broken", error="RuntimeError: not pulled"),
        ]
    )
    lines = table.splitlines()
    assert lines[0].startswith("model")
    assert "nDCG@10" in lines[0]
    order = [ln.split()[0] for ln in lines[2:] if not ln.startswith(" ")]
    assert order == ["strong", "weak", "broken"]
    assert lines[3].startswith("  vector leg only")
    assert "FAILED: RuntimeError: not pulled" in table
    assert "missed: q1" in table
    assert "false positives: junk q" in table
    assert ev.K == 10


def test_format_calibration_prints_percentiles_per_model() -> None:
    report = format_calibration(
        [
            ModelResult("m", true_similarities=[0.6, 0.7], negative_similarities=[]),
            ModelResult("broken", error="x"),
        ]
    )
    lines = report.splitlines()
    assert lines[0] == "m"
    assert "p50" in lines[1]
    assert lines[2].split()[:3] == ["true", "hits", "2"]
    assert "0.650" in lines[2]
    assert "(none)" in lines[3]
    assert "broken" not in report
