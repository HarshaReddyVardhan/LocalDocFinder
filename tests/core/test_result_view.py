from localdoc_finder.core.result_view import SortOrder, ext_counts, refine
from localdoc_finder.core.skills.search import SearchResult


def hit(path: str, *, ext: str = "", indexed_at: float = 0.0, relevance: int = 50) -> SearchResult:
    return SearchResult(
        path, "p", "doc", "", "", 0, 0, 0, "", 0.5, ext=ext, indexed_at=indexed_at,
        relevance=relevance,
    )  # fmt: skip


RESULTS = [
    hit(r"D:\x\zeta.pdf", ext=".pdf", indexed_at=30),
    hit(r"D:\x\Alpha.txt", ext=".txt", indexed_at=10),
    hit(r"D:\x\mid.PDF", indexed_at=20),  # no ext column: taken from the name, any case
    hit(r"D:\x\README", indexed_at=40),
]


def names(results: list[SearchResult]) -> list[str]:
    return [r.path.rsplit("\\", 1)[1] for r in results]


def test_relevance_keeps_the_search_order() -> None:
    assert names(refine(RESULTS)) == ["zeta.pdf", "Alpha.txt", "mid.PDF", "README"]


def test_sort_by_date_indexed_newest_first() -> None:
    ordered = refine(RESULTS, order=SortOrder.DATE_INDEXED)
    assert names(ordered) == ["README", "zeta.pdf", "mid.PDF", "Alpha.txt"]


def test_sort_by_name_ignores_case() -> None:
    assert names(refine(RESULTS, order=SortOrder.NAME)) == [
        "Alpha.txt", "mid.PDF", "README", "zeta.pdf",
    ]  # fmt: skip


def test_filter_by_extension_then_sort() -> None:
    pdfs = refine(RESULTS, ".pdf", SortOrder.NAME)
    assert names(pdfs) == ["mid.PDF", "zeta.pdf"]
    assert refine(RESULTS, "") == [RESULTS[3]]  # files without an extension
    assert refine(RESULTS, ".md") == []


def test_ties_keep_the_search_ranking() -> None:
    same = [hit(r"D:\x\b.txt"), hit(r"D:\x\a.txt")]
    assert refine(same, order=SortOrder.DATE_INDEXED) == same


def test_ext_counts_most_common_first() -> None:
    assert ext_counts(RESULTS) == [(".pdf", 2), ("", 1), (".txt", 1)]
