from localdoc_finder.core.relevance import keywords, reduce_to_relevant


def test_keywords_keep_technical_tokens() -> None:
    assert keywords("Knows C++ and Node.js", "AWS") >= {"c++", "node.js", "aws", "knows"}


def test_short_text_is_untouched() -> None:
    assert reduce_to_relevant("one\n\ntwo", {"two"}, 100) == ("one\n\ntwo", False)


def test_the_most_relevant_paragraphs_are_kept_in_their_order() -> None:
    paragraphs = [f"filler paragraph {i} " * 10 for i in range(20)]
    paragraphs[3] = "Kubernetes clusters on AWS " * 5
    paragraphs[17] = "Python services on Kubernetes " * 5
    text = "\n\n".join(paragraphs)
    kept, cut = reduce_to_relevant(text, {"kubernetes", "python", "aws"}, 120)
    assert cut
    assert kept.index("Kubernetes clusters") < kept.index("Python services")
    assert "filler paragraph 10" not in kept


def test_without_wanted_words_the_start_is_kept() -> None:
    text = "\n\n".join(f"part {i} " * 30 for i in range(10))
    kept, cut = reduce_to_relevant(text, set(), 100)
    assert cut and kept.startswith("part 0")


def test_one_huge_paragraph_is_cut_to_the_budget() -> None:
    kept, cut = reduce_to_relevant("word " * 5000, {"word"}, 50)
    assert cut and len(kept) <= 50 * 4
