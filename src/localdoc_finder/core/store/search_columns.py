"""Chunk columns derived from a chunk's text and path, for ranking and keyword search.

``content_chars``: letters and digits in the chunk body (the header line excluded). A chunk with
almost none, such as a scanned page OCR could not read, says nothing about its file.
``name``: the lower-case file name, for the filename leg of search.
``search_text``: the file-name words plus the body, what BM25 indexes. Directory names are left
out on purpose: a folder called ``billing`` must not make every file in it match "billing".

Each column has a Python form (the indexer) and a SQL form (the schema migration that backfills
an existing index). They live together so the two cannot drift apart.
"""

import re

_SEPARATORS = re.compile(r"[\\/]")
_EXTENSION = re.compile(r"\.[^.]*$")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_WORD_BREAKS = re.compile(r"[_\-.]+")


def chunk_body(text: str) -> str:
    """The stored text without its ``project > path > symbol`` header line."""
    return text.split("\n", 1)[1] if "\n" in text else text


def content_chars(body: str) -> int:
    return sum(c.isalnum() for c in body)


def file_name(path: str) -> str:
    return _SEPARATORS.split(path)[-1].lower()


def name_words(path: str) -> str:
    """``D:\\x\\parseQuery_v2.py`` -> ``parse query v2``: identifiers split into words."""
    stem = _EXTENSION.sub("", _SEPARATORS.split(path)[-1])
    return _WORD_BREAKS.sub(" ", _CAMEL.sub(r"\1 \2", stem)).lower().strip()


def search_text(path: str, body: str) -> str:
    return f"{name_words(path)}\n{body}"


# The same derivations in DataFusion SQL, over the stored ``text`` and ``path`` columns.
_SQL_BODY = "substr(text, strpos(text, chr(10)) + 1)"
_SQL_BASE = r"regexp_replace(path, '^.*[\\/]', '')"
_SQL_WORDS = (
    "trim(lower(regexp_replace(regexp_replace(regexp_replace("
    rf"{_SQL_BASE}, '\.[^.]*$', ''), '([a-z0-9])([A-Z])', '${{1}} ${{2}}', 'g'), "
    r"'[_\-.]+', ' ', 'g')))"
)
SQL_COLUMNS: dict[str, str] = {
    "content_chars": (
        rf"CAST(length(regexp_replace({_SQL_BODY}, '[^\p{{L}}\p{{N}}]', '', 'g')) AS INT)"
    ),
    "name": f"lower({_SQL_BASE})",
    "search_text": f"concat({_SQL_WORDS}, chr(10), {_SQL_BODY})",
}
