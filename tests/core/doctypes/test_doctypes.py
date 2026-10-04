from pathlib import Path

import numpy as np
import pytest

from localdoc_finder.core.doctypes import base
from localdoc_finder.core.doctypes.base import (
    DOC_TYPE_CODE,
    DOC_TYPE_OTHER,
    DocInfo,
    DocTypeClassifier,
    DocTypeClassifierSet,
    build_prototypes,
    register_doctype,
)
from localdoc_finder.core.doctypes.versions import (
    VersionCandidate,
    group_versions,
    jaccard,
    newest_per_group,
    shingles,
)
from localdoc_finder.core.settings import DocTypeSettings, ScopeSettings

RESUME = """Jane Doe
Summary
Backend engineer.
Work Experience
Built payment systems.
Education
BSc Computer Science
Skills
Python, SQL
Projects
Search engine
"""

JD = """Senior Backend Engineer
About the role
We are hiring a backend engineer.
Responsibilities
Design APIs
Requirements
5 years of experience with Python
Nice to have
Kubernetes
"""


@pytest.fixture
def classifiers() -> DocTypeClassifierSet:
    return DocTypeClassifierSet(DocTypeSettings(), ScopeSettings())


def info(name: str, text: str, vector: np.ndarray | None = None) -> DocInfo:
    return DocInfo(Path("D:/docs") / name, name, text, vector)


class TestClassification:
    def test_resume_by_content_even_with_neutral_name(
        self, classifiers: DocTypeClassifierSet
    ) -> None:
        assert classifiers.classify(info("jane.docx", RESUME)).doc_type == "resume"

    def test_resume_by_name_and_content_is_confident(
        self, classifiers: DocTypeClassifierSet
    ) -> None:
        result = classifiers.classify(info("Resume_v2.pdf", RESUME))
        assert result.doc_type == "resume"
        assert result.confidence == 1.0

    def test_job_description(self, classifiers: DocTypeClassifierSet) -> None:
        assert classifiers.classify(info("backend.txt", JD)).doc_type == "jd"
        assert classifiers.classify(info("JD_acme.pdf", "short")).doc_type == "jd"

    def test_cover_letter_invoice_paper(self, classifiers: DocTypeClassifierSet) -> None:
        letter = "Dear Hiring Manager,\nI am writing to apply for the role.\nSincerely,\nJane"
        assert classifiers.classify(info("cover_letter.docx", letter)).doc_type == "cover_letter"
        invoice = "Invoice #42\nBill To\nAcme\nSubtotal 10\nTotal 12\nAmount due 12"
        assert classifiers.classify(info("inv-42.pdf", invoice)).doc_type == "invoice"
        paper = (
            "Abstract\nWe propose.\nIntroduction\nText\nRelated Work\nText\nReferences\n[1] et al."
        )
        assert classifiers.classify(info("attention.pdf", paper)).doc_type == "paper"

    def test_plan_and_notes_by_path(self, classifiers: DocTypeClassifierSet) -> None:
        plan = DocInfo(
            Path("C:/Users/x/.claude/plans/cosmos.md"), "plan", "# Context\n# Verification"
        )
        assert classifiers.classify(plan).doc_type == "plan"
        memory = DocInfo(Path("C:/Users/x/.claude/projects/p/memory/note.md"), "n", "remember")
        assert classifiers.classify(memory).doc_type in {"notes", "plan"}

    def test_code_is_decided_by_extension(self, classifiers: DocTypeClassifierSet) -> None:
        result = classifiers.classify(info("resume_parser.py", RESUME))
        assert result.doc_type == DOC_TYPE_CODE

    def test_unrecognised_is_other(self, classifiers: DocTypeClassifierSet) -> None:
        result = classifiers.classify(info("holiday.txt", "we went to the beach"))
        assert result.doc_type == DOC_TYPE_OTHER

    def test_resume_beats_jd_on_resume_content(self, classifiers: DocTypeClassifierSet) -> None:
        scores = classifiers.classify(info("x.pdf", RESUME)).scores
        assert scores["resume"] > scores["jd"]

    def test_headings_match_with_markdown_decoration(
        self, classifiers: DocTypeClassifierSet
    ) -> None:
        text = "## Experience\nx\n## Education\ny\n**Skills**\nz\n"
        assert classifiers.classify(info("cv.md", text)).doc_type == "resume"

    def test_rules_are_configurable(self) -> None:
        rules = DocTypeSettings().rules | {"resume": base.DocTypeRule(filename=("vitae",))}
        custom = DocTypeClassifierSet(DocTypeSettings(rules=rules, threshold=0.4), ScopeSettings())
        assert custom.classify(info("my_vitae.pdf", "x")).doc_type == "resume"
        assert custom.classify(info("resume.pdf", "x")).doc_type == DOC_TYPE_OTHER

    def test_new_type_is_one_decorated_class(self) -> None:
        @register_doctype("recipe")
        class Recipe(DocTypeClassifier):
            name = "recipe"

            def score(self, info: DocInfo) -> float:
                return 0.9 if "ingredients" in info.text.lower() else 0.0

        try:
            custom = DocTypeClassifierSet(DocTypeSettings(), ScopeSettings())
            assert custom.classify(info("pasta.txt", "Ingredients: flour")).doc_type == "recipe"
        finally:
            base.DOCTYPES.remove("recipe")


class TestPrototypes:
    def test_prototype_similarity_adds_evidence(self) -> None:
        resume_vec = np.array([1.0, 0.0, 0.0])
        protos = {"resume": resume_vec, "jd": np.array([0.0, 1.0, 0.0])}
        with_protos = DocTypeClassifierSet(DocTypeSettings(), ScopeSettings(), protos)
        weak = info("scan0001.pdf", "Experience\nSkills\n", vector=np.array([0.95, 0.05, 0.0]))
        plain = DocTypeClassifierSet(DocTypeSettings(), ScopeSettings())
        assert with_protos.classify(weak).scores["resume"] > plain.classify(weak).scores["resume"]

    def test_orthogonal_or_missing_vectors_add_nothing(self) -> None:
        protos = {"resume": np.array([1.0, 0.0])}
        classifier = DocTypeClassifierSet(DocTypeSettings(), ScopeSettings(), protos)
        orth = classifier.classify(info("a.pdf", "x", vector=np.array([0.0, 1.0])))
        none = classifier.classify(info("a.pdf", "x"))
        zero = classifier.classify(info("a.pdf", "x", vector=np.array([0.0, 0.0])))
        assert orth.scores["resume"] == none.scores["resume"] == zero.scores["resume"]

    def test_build_prototypes_embeds_each_type_once(self) -> None:
        calls: list[list[str]] = []

        def embed(texts: list[str]) -> np.ndarray:
            calls.append(texts)
            return np.eye(len(texts))

        protos = build_prototypes(embed)
        assert len(calls) == 1
        assert set(protos) == set(base.PROTOTYPE_TEXTS)
        assert build_prototypes(embed, {"a": "x"})["a"].tolist() == [1.0]


class TestVersions:
    base_text = " ".join(f"word{i}" for i in range(200))

    def cand(
        self, path: str, text: str, mod: int = 1, doc_type: str = "resume"
    ) -> VersionCandidate:
        return VersionCandidate(path, doc_type, text, mod)

    def test_near_duplicates_group_and_newest_wins(self) -> None:
        edited = self.base_text.replace("word10 ", "changed ", 1)
        items = [
            self.cand("D:/r/Resume_v1.pdf", self.base_text, mod=1),
            self.cand("D:/r/Resume_final.docx", edited, mod=3),
            self.cand("D:/r/resume (2).pdf", self.base_text, mod=2),
            self.cand("D:/r/Other.pdf", " ".join(f"zzz{i}" for i in range(200)), mod=9),
        ]
        groups = group_versions(items, frozenset({"resume"}))
        assert groups["D:/r/Resume_v1.pdf"] == groups["D:/r/Resume_final.docx"]
        assert groups["D:/r/resume (2).pdf"] == groups["D:/r/Resume_v1.pdf"]
        assert "D:/r/Other.pdf" not in groups
        shown = newest_per_group(items, groups)
        assert sorted(shown) == ["D:/r/Other.pdf", "D:/r/Resume_final.docx"]

    def test_group_id_is_stable_across_input_order(self) -> None:
        items = [self.cand("D:/a.pdf", self.base_text), self.cand("D:/b.pdf", self.base_text)]
        first = group_versions(items, frozenset({"resume"}))
        second = group_versions(list(reversed(items)), frozenset({"resume"}))
        assert first == second

    def test_dissimilar_documents_stay_apart(self) -> None:
        items = [
            self.cand("a", self.base_text),
            self.cand("b", self.base_text[: len(self.base_text) // 2]),
        ]
        assert group_versions(items, frozenset({"resume"})) == {}

    def test_only_versioned_types_and_same_type_are_grouped(self) -> None:
        items = [
            self.cand("a", self.base_text, doc_type="notes"),
            self.cand("b", self.base_text, doc_type="notes"),
            self.cand("c", self.base_text, doc_type="resume"),
            self.cand("d", self.base_text, doc_type="jd"),
        ]
        assert group_versions(items, frozenset({"resume", "jd"})) == {}

    def test_shingle_edge_cases(self) -> None:
        assert shingles("") == frozenset()
        assert len(shingles("only three words")) == 1
        assert jaccard(frozenset(), frozenset()) == 0.0
        assert jaccard(shingles(self.base_text), shingles(self.base_text)) == 1.0
