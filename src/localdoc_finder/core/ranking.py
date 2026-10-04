"""Turn per-chunk evidence into ranked files with a relevance score and a weak tier.

Pure functions only; ``skills/search.py`` gathers the evidence. A file's score adds up:

- its best chunk's fused score (similarity above the model's floor, plus BM25),
- a small bonus for a second matching chunk (two passages beat one),
- how much of the query its file name covers, word for word, and a bonus when the query *is* the
  file name.

A result is **weak** when nothing vouches for it or when it trails the best result by more than
``relative_gap``. Vouching takes a similarity at or above the model's floor, or a keyword or
file-name match covering at least ``min_term_coverage`` of the query's words: one incidental
word ("chip" in a UI file for "chocolate chip cookie recipe") is not evidence. Weak results are
still returned, after the strong ones, so the list can show everything while ranking it honestly.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from localdoc_finder.core.settings import SearchSettings
from localdoc_finder.core.store.search_columns import name_words

_MIN_TERM_CHARS = 3
_PLURAL_MIN_CHARS = 4  # "logs" ~ "log" is too loose; "payments" ~ "payment" is not
# Function words, prepositions included: they say nothing about what a query is looking for.
STOP_WORDS = frozenset({
    "a", "about", "above", "across", "after", "against", "all", "along", "among", "an", "and",
    "any", "are", "around", "as", "at", "be", "been", "before", "behind", "below", "between",
    "beyond", "but", "by", "can", "could", "did", "do", "does", "doing", "during", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "my",
    "near", "no", "not", "of", "off", "on", "onto", "or", "our", "out", "over", "please", "show",
    "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "toward", "under", "until", "up", "upon",
    "us", "via", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "within", "without", "would", "you", "your",
})  # fmt: skip


def query_terms(text: str) -> list[str]:
    """The words a query is about: lower-case, three letters or more, stop words dropped."""
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) >= _MIN_TERM_CHARS and w not in STOP_WORDS]


_SUFFIXES = (("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""))
_MIN_STEM_CHARS = 3


def _stem(word: str) -> str:
    """A light English stem, enough to see "retried" in "retry" and "payments" in "payment"."""
    for suffix, replacement in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM_CHARS:
            return word[: -len(suffix)] + replacement
    return word


def term_coverage(terms: Sequence[str], text: str) -> float:
    """Share of ``terms`` (from ``query_terms``) that occur in ``text`` as words."""
    if not terms:
        return 0.0
    present = {_stem(w) for w in re.findall(r"\w+", text.lower())}
    return sum(_stem(term) in present for term in terms) / len(terms)


def _same_word(first: str, second: str) -> bool:
    if first == second:
        return True
    shortest = min(len(first), len(second))
    return shortest >= _PLURAL_MIN_CHARS and first.removesuffix("s") == second.removesuffix("s")


@dataclass(frozen=True)
class NameMatch:
    share: float = 0.0  # query words found as whole words of the file name
    exact: bool = False  # the query is the file name, or its name without the extension

    @property
    def hit(self) -> bool:
        return self.share > 0


def name_match(path: str, text: str) -> NameMatch:
    """How well the file name answers the query, by whole words.

    ``log`` does not match ``catalog.py``; ``retry payment`` matches ``retryPayments.py`` fully.
    The extension never counts on its own: ``txt`` is in every ``.txt`` file's name.
    """
    name = Path(path).name.lower()
    query = " ".join(text.lower().split())
    if query and query in (name, Path(name).stem):
        return NameMatch(1.0, exact=True)
    extension = Path(name).suffix.lstrip(".")
    terms = [t for t in query_terms(text) if t != extension]
    if not terms:
        return NameMatch()
    tokens = name_words(path).split()
    found = sum(any(_same_word(term, token) for token in tokens) for term in terms)
    return NameMatch(found / len(terms))


@dataclass
class FileEvidence:
    """Everything known about one file for one query."""

    path: str
    best: float = 0.0  # fused score of its best chunk
    second: float = 0.0  # fused score of its second-best chunk
    similarity: float | None = None  # best cosine similarity of any of its chunks
    coverage: float = 0.0  # best share of query words in a chunk the keyword leg matched
    name: NameMatch = NameMatch()
    in_current_project: bool = False

    def add_chunk(self, fused: float, similarity: float | None, coverage: float | None) -> None:
        """One more chunk of the file; ``coverage`` is ``None`` when BM25 did not match it."""
        if fused > self.best:
            self.best, self.second = fused, self.best
        elif fused > self.second:
            self.second = fused
        if similarity is not None and (self.similarity is None or similarity > self.similarity):
            self.similarity = similarity
        if coverage is not None:
            self.coverage = max(self.coverage, coverage)


@dataclass(frozen=True)
class Ranked:
    evidence: FileEvidence
    score: float
    relevance: int  # 0-100, for display
    weak: bool


def file_score(evidence: FileEvidence, cfg: SearchSettings) -> float:
    score = evidence.best + cfg.multi_hit_bonus * evidence.second
    score += cfg.filename_weight * evidence.name.share
    if evidence.name.exact:
        score += cfg.filename_exact_bonus
    if evidence.in_current_project:
        score *= cfg.current_project_boost
    return score


def vouched_for(evidence: FileEvidence, floor: float, min_coverage: float) -> bool:
    """Some leg found real evidence: similarity at or above the floor, or keywords or file-name
    words covering enough of the query."""
    if evidence.similarity is not None and evidence.similarity >= floor:
        return True
    return evidence.coverage >= min_coverage or evidence.name.share >= min_coverage


def rank_files(evidence: Sequence[FileEvidence], cfg: SearchSettings, floor: float) -> list[Ranked]:
    """Strong results by score, then weak ones by score."""
    scored = [
        (e, file_score(e, cfg), vouched_for(e, floor, cfg.min_term_coverage)) for e in evidence
    ]
    top = max((score for _, score, vouched in scored if vouched), default=0.0)
    cutoff = (1.0 - cfg.relative_gap) * top
    ordered = sorted(
        ((e, score, not vouched or score < cutoff) for e, score, vouched in scored),
        key=lambda item: (item[2], -item[1]),
    )
    ranked: list[Ranked] = []
    ceiling = 100
    for e, score, weak in ordered:
        # Never shown as more relevant than a result above it: an unvouched file can outscore
        # the last strong one and still belongs below it.
        ceiling = min(ceiling, round(100 * min(score, 1.0)))
        ranked.append(Ranked(e, score, ceiling, weak))
    return ranked
