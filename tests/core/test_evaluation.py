from pathlib import Path

import pytest
from tests.core.fakes import FakeEmbedder

from localdoc_finder.core import evaluation as ev
from localdoc_finder.core.evaluation import (
    EvalQuery,
    EvalSpec,
    EvaluationError,
    Evaluator,
    ModelResult,
    format_results,
    load_spec,
    rank_of,
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
        settings, scope, factory or (lambda _model: FakeEmbedder()), tmp_path / "eval", exclude
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
            ("corpus: [x]\nqueries:\n  - q: a\n", "needs 'q' and 'expect'"),
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
            ),
        )

    def test_scores_recall_mrr_and_speed(self, tmp_path: Path, corpus: Path) -> None:
        result = make_evaluator(tmp_path).evaluate("fake", self.spec(corpus))
        assert result.error == ""
        assert result.files == 3
        assert result.recall == pytest.approx(2 / 3)
        assert result.mrr > 0.6
        assert result.missed == ["something nobody wrote"]
        assert result.index_seconds >= 0
        assert result.p50_ms >= 0

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
        assert len(embedder.calls) - calls == 3  # only the three query embeddings

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
                "weak", recall=0.5, mrr=0.2, files=10, index_seconds=3.0, p50_ms=40, missed=["q1"]
            ),
            ModelResult("strong", recall=0.9, mrr=0.7, files=10, index_seconds=5.0, p50_ms=60),
            ModelResult("broken", error="RuntimeError: not pulled"),
        ]
    )
    lines = table.splitlines()
    assert lines[0].startswith("model")
    order = [ln.split()[0] for ln in lines[2:] if not ln.startswith("    ")]
    assert order == ["strong", "weak", "broken"]
    assert "FAILED: RuntimeError: not pulled" in table
    assert "missed: q1" in table
    assert ev.K == 10
