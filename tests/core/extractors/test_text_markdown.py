import json
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.core.extractors.conftest import Writer

from vector_embed.core.extractors import chunking
from vector_embed.core.extractors.base import Chunk, ExtractError, ExtractorSet
from vector_embed.core.extractors.markdown import heading_sections
from vector_embed.core.settings import ChunkingSettings

Builder = Callable[[ChunkingSettings], ExtractorSet]


class TestChunkingHelpers:
    def test_an_overlap_larger_than_the_window_cannot_explode_the_chunk_count(self) -> None:
        lines = [f"line {i:04d}" for i in range(1000)]  # 9 chars + newline each
        parts = chunking.split_by_lines(lines, 1, max_chars=50, overlap=20)
        window = 50 // 10  # lines per window
        assert len(parts) <= len(lines) // (window // 2) + 1  # at least half a window per step
        covered = {n for _, start, end in parts for n in range(start, end + 1)}
        assert covered == set(range(1, 1001))  # still every line

    def test_split_respects_max_chars_and_overlap(self) -> None:
        lines = [f"line{i:02d}" for i in range(10)]  # 7 chars with newline each
        parts = chunking.split_by_lines(lines, 1, max_chars=22, overlap=1)
        assert parts[0] == ("line00\nline01\nline02", 1, 3)
        assert parts[1][1] == 3  # overlap of one line
        assert all(len(t) <= 22 for t, _, _ in parts)
        assert parts[-1][2] == 10

    def test_enormous_single_line_is_hard_cut(self) -> None:
        parts = chunking.split_by_lines(["x" * 25], 1, max_chars=10, overlap=0)
        assert [len(t) for t, _, _ in parts] == [10, 10, 5]

    def test_first_line_offset(self) -> None:
        assert chunking.split_by_lines(["a", "b"], 10, 100, 0) == [("a\nb", 10, 11)]

    def test_merge_small_joins_neighbours_of_same_kind(self) -> None:
        merged = chunking.merge_small(
            [Chunk("a", "code", "f", 1, 1), Chunk("b", "code", "g", 2, 2), Chunk("c", "doc", "h")],
            ChunkingSettings(),
        )
        assert [c.text for c in merged] == ["a\nb", "c"]
        assert merged[0].symbol == "f, g"
        assert merged[0].end_line == 2

    def test_merge_small_keeps_big_chunks_apart(self) -> None:
        cfg = ChunkingSettings(min_chunk_chars=5, target_chunk_chars=50)
        assert len(chunking.merge_small([Chunk("x" * 10), Chunk("y" * 10)], cfg)) == 2

    def test_read_text_handles_encodings_and_binaries(self, tmp_path: Path) -> None:
        bom = tmp_path / "bom.txt"
        bom.write_bytes(b"\xef\xbb\xbfhello")
        utf16 = tmp_path / "u16.txt"
        utf16.write_bytes("héllo".encode("utf-16"))
        binary = tmp_path / "b.bin"
        binary.write_bytes(b"ab\x00cd")
        assert chunking.read_text(str(bom)) == "hello"
        assert chunking.read_text(str(utf16)) == "héllo"
        with pytest.raises(ExtractError, match="binary"):
            chunking.read_text(str(binary))


class TestTextExtractor:
    def test_prose_is_doc_and_config_is_code(self, extractors: ExtractorSet, write: Writer) -> None:
        assert extractors.extract(write("notes.txt", "hello world"))[0].kind == "doc"
        assert extractors.extract(write("conf.yaml", "a: 1\n"))[0].kind == "code"
        assert extractors.extract(write("Dockerfile", "FROM x\n"))[0].kind == "code"

    def test_long_text_is_windowed_with_line_numbers(
        self, extractors: ExtractorSet, write: Writer
    ) -> None:
        body = "\n".join(f"line {i} " + "w" * 60 for i in range(200))
        chunks = extractors.extract(write("big.txt", body))
        assert len(chunks) > 3
        assert chunks[0].start_line == 1
        assert chunks[-1].end_line == 200

    def test_rtf_is_stripped(self, extractors: ExtractorSet, write: Writer) -> None:
        chunks = extractors.extract(write("a.rtf", r"{\rtf1\ansi Hello \b world\b0 }"))
        assert "Hello" in chunks[0].text
        assert "\\rtf" not in chunks[0].text

    def test_binary_generated_and_big_data_are_rejected(
        self, extractors: ExtractorSet, write: Writer
    ) -> None:
        with pytest.raises(ExtractError, match="binary"):
            extractors.extract(write("x.txt", b"a\x00b"))
        with pytest.raises(ExtractError, match="generated"):
            extractors.extract(write("bundle.txt", "x" * 5000))
        with pytest.raises(ExtractError, match="too large"):
            extractors.extract(write("dump.json", '{"a": "' + ("y " * 400_000) + '"}'))

    def test_blank_file_yields_nothing(self, extractors: ExtractorSet, write: Writer) -> None:
        assert extractors.extract(write("blank.txt", "\n\n  \n")) == []

    def test_unknown_extension_has_no_extractor(
        self, extractors: ExtractorSet, write: Writer
    ) -> None:
        with pytest.raises(ExtractError, match="no extractor"):
            extractors.extract(write("a.xyz", "x"))

    def test_chunk_cap(self, with_chunking: Builder, write: Writer) -> None:
        limited = with_chunking(ChunkingSettings(max_chunks_per_file=2))
        body = "\n".join("w" * 200 for _ in range(200))
        assert len(limited.extract(write("cap.txt", body))) == 2

    def test_oversized_chunks_are_truncated(self, with_chunking: Builder, write: Writer) -> None:
        limited = with_chunking(ChunkingSettings(max_chunk_chars=100))
        chunks = limited.extract(write("t.txt", "a" * 250 + "\nb"))
        assert all(len(c.text) <= 100 for c in chunks)

    def test_transcripts_are_text_only_when_enabled(
        self, extractors: ExtractorSet, write: Writer
    ) -> None:
        with pytest.raises(ExtractError, match="no extractor"):
            extractors.extract(write("s.jsonl", '{"a": 1}'))


class TestNotebookAndSvg:
    def test_notebook_keeps_cells_but_not_outputs(
        self, extractors: ExtractorSet, write: Writer
    ) -> None:
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "words"]},
                {"cell_type": "code", "source": "x = 1", "outputs": [{"data": "BASE64" * 100}]},
                {"cell_type": "code", "source": "   "},
            ]
        }
        chunks = extractors.extract(write("a.ipynb", json.dumps(notebook)))
        assert [(c.kind, c.symbol) for c in chunks] == [("doc", "cell 1"), ("code", "cell 2")]
        assert all("BASE64" not in c.text for c in chunks)

    def test_invalid_notebook(self, extractors: ExtractorSet, write: Writer) -> None:
        with pytest.raises(ExtractError, match="invalid notebook"):
            extractors.extract(write("bad.ipynb", "{nope"))

    def test_svg_keeps_text_only(self, extractors: ExtractorSet, write: Writer) -> None:
        svg = '<svg><path d="M0 0L9 9"/><title>Login flow</title><text>Start</text></svg>'
        assert extractors.extract(write("d.svg", svg))[0].text == "Login flow Start"
        with pytest.raises(ExtractError, match="without text"):
            extractors.extract(write("e.svg", '<svg><path d="M0 0"/></svg>'))


class TestMarkdown:
    def test_sections_carry_heading_path(self) -> None:
        lines = ["intro", "# A", "a body", "## B", "b body", "# C", "c body"]
        assert heading_sections(lines) == [
            ("", 0, 1),
            ("A", 1, 3),
            ("A > B", 3, 5),
            ("C", 5, 7),
        ]

    def test_fenced_hashes_are_not_headings(self) -> None:
        assert heading_sections(["# Real", "```", "# not a heading", "```"]) == [("Real", 0, 4)]

    def test_small_sections_merge_and_big_ones_split(
        self, with_chunking: Builder, write: Writer
    ) -> None:
        cfg = ChunkingSettings(max_chunk_chars=300, target_chunk_chars=150, min_chunk_chars=80)
        body = "\n".join(f"sentence {i} " * 8 for i in range(30))
        md = "# A\nshort\n# B\nshort too\n# Big\n" + body
        chunks = with_chunking(cfg).extract(write("p.md", md))
        assert any("A" in c.symbol and "B" in c.symbol for c in chunks)
        assert sum(1 for c in chunks if c.symbol == "Big") > 1
        assert all(c.kind == "doc" for c in chunks)

    def test_plan_files_are_markdown(self, extractors: ExtractorSet, write: Writer) -> None:
        plan = "# Plan\n\nstep one\n\n## Verify\n\nrun tests\n"
        chunks = extractors.extract(write("plan.md", plan))
        assert chunks
        assert chunks[0].start_line == 1
