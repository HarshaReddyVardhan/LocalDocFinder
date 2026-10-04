"""Cut a long document to the parts that matter for a query, keeping their original order.

Used by Match (a resume against a job description) and Chat (a pinned document against the
user's message): sending only the start of a long document loses whatever is further down.
"""

import re

from localdoc_finder.core.tokens import estimate_tokens, fit_to_budget

_WORD = re.compile(r"[a-z0-9+#.]{2,}")
_SECTION_SPLIT = re.compile(r"\n\s*\n")


def keywords(*texts: str) -> set[str]:
    """Lower-case words (and tokens such as ``c++`` or ``node.js``) of ``texts``."""
    words: set[str] = set()
    for text in texts:
        words.update(_WORD.findall(text.lower()))
    return words


def reduce_to_relevant(text: str, wanted: set[str], budget_tokens: int) -> tuple[str, bool]:
    """Fit ``text`` to ``budget_tokens`` by keeping the paragraphs sharing most ``wanted`` words.

    Paragraphs keep their order. Returns the text and whether anything was dropped. With no
    ``wanted`` words this keeps the first paragraphs that fit.
    """
    if estimate_tokens(text) <= budget_tokens:
        return text, False
    sections = [s for s in _SECTION_SPLIT.split(text) if s.strip()]
    ranked = sorted(
        range(len(sections)),
        key=lambda i: len(wanted & set(_WORD.findall(sections[i].lower()))),
        reverse=True,
    )
    kept: set[int] = set()
    used = 0
    for index in ranked:
        cost = estimate_tokens(sections[index])
        if used + cost > budget_tokens:
            continue
        kept.add(index)
        used += cost
    if not kept:  # one huge section: cut it
        return fit_to_budget(sections[0] if sections else text, budget_tokens)[0], True
    return "\n\n".join(sections[i] for i in sorted(kept)), True
