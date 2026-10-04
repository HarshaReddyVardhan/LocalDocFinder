from pathlib import Path

import pytest

from localdoc_finder.core.store.search_columns import (
    SQL_COLUMNS,
    chunk_body,
    content_chars,
    file_name,
    name_words,
    search_text,
)

CASES = [
    (
        "D:\\work\\billing\\retryPayments.py",
        "p > D:\\work\\billing\\retryPayments.py\ndef retry():",
    ),
    (
        "D:/scans/Lease_Agreement-scan.v2.PDF",
        "p > D:/scans/Lease_Agreement-scan.v2.PDF > p.1\nRent: $1,450 café",
    ),
    ("C:\\x\\scan_0042.pdf", "p > C:\\x\\scan_0042.pdf > p.2\n  .. -- ..  "),
    ("README", "p > README\nline one\nline two"),
]


def test_the_header_line_is_not_part_of_the_body() -> None:
    assert chunk_body("p > a.py\nbody\nmore") == "body\nmore"
    assert chunk_body("no header") == "no header"


def test_content_counts_letters_and_digits_only() -> None:
    assert content_chars("Rent: $1,450 café") == len("Rent1450café")
    assert content_chars("  .. -- ..  ") == 0


@pytest.mark.parametrize(
    ("path", "words"),
    [
        ("D:\\work\\billing\\retryPayments.py", "retry payments"),
        ("D:/scans/Lease_Agreement-scan.v2.PDF", "lease agreement scan v2"),
        ("C:\\x\\HTTPServer.go", "httpserver"),
        ("README", "readme"),
    ],
)
def test_file_names_become_words_without_directories(path: str, words: str) -> None:
    assert name_words(path) == words
    assert "\\" not in file_name(path) and "/" not in file_name(path)


def test_search_text_is_the_name_words_then_the_body() -> None:
    assert search_text("D:\\billing\\pay_run.py", "charge") == "pay run\ncharge"


def test_the_sql_backfill_matches_the_indexer(tmp_path: Path) -> None:
    """The migration's SQL must derive exactly what the indexer writes for new rows."""
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(tmp_path))
    paths = [p for p, _ in CASES]
    texts = [t for _, t in CASES]
    table = db.create_table("t", data=pa.table({"path": paths, "text": texts}))
    table.add_columns(SQL_COLUMNS)
    for row in table.to_arrow().to_pylist():
        body = chunk_body(row["text"])
        assert row["content_chars"] == content_chars(body), row["path"]
        assert row["name"] == file_name(row["path"]), row["path"]
        assert row["search_text"] == search_text(row["path"], body), row["path"]
