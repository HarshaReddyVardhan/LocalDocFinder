import textwrap
from collections.abc import Callable

import pytest
from tests.core.extractors.conftest import Writer

from localdoc_finder.core.extractors import code
from localdoc_finder.core.extractors.base import ExtractorSet
from localdoc_finder.core.settings import ChunkingSettings

Builder = Callable[[ChunkingSettings], ExtractorSet]

PY = textwrap.dedent(
    '''
    import os
    from pathlib import Path

    def top(a):
        """Top-level function."""
        return a + 1


    class Greeter:
        """Says hello."""

        def hello(self, name):
            return f"hi {name}"

        def bye(self):
            return "bye"
    '''
).lstrip()


def test_python_functions_and_methods_get_qualified_symbols(
    no_merge: ExtractorSet, write: Writer
) -> None:
    chunks = no_merge.extract(write("m.py", PY))
    symbols = [c.symbol for c in chunks]
    assert "<outline>" in symbols
    assert "top" in symbols
    assert any(s.startswith("Greeter") for s in symbols)
    outline = chunks[0]
    assert outline.kind == "outline"
    assert "Imports:" in outline.text
    assert "import os" in outline.text
    assert "class Greeter" in outline.text


def test_chunks_have_line_ranges(no_merge: ExtractorSet, write: Writer) -> None:
    chunks = no_merge.extract(write("m.py", PY))
    top = next(c for c in chunks if c.symbol == "top")
    assert top.start_line == 4
    assert "return a + 1" in top.text


def test_large_class_is_split_into_methods(with_chunking: Builder, write: Writer) -> None:
    methods = "\n".join(
        f"    def method_{i}(self):\n        return {i}  # " + "x" * 80 for i in range(12)
    )
    source = f"class Big:\n    '''doc'''\n\n{methods}\n"
    cfg = ChunkingSettings(max_chunk_chars=400, target_chunk_chars=200, min_chunk_chars=50)
    symbols = [c.symbol for c in with_chunking(cfg).extract(write("big.py", source))]
    assert any("Big.method_3" in s for s in symbols)
    assert any("(header)" in s for s in symbols)


def test_oversized_function_is_split_into_parts(with_chunking: Builder, write: Writer) -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(80))
    cfg = ChunkingSettings(max_chunk_chars=300, target_chunk_chars=150, min_chunk_chars=20)
    chunks = with_chunking(cfg).extract(write("long.py", f"def long_fn():\n{body}\n"))
    assert "long_fn (part 2)" in [c.symbol for c in chunks]


def test_javascript_arrow_functions_and_exports(no_merge: ExtractorSet, write: Writer) -> None:
    js = "export function a() { return 1 }\nconst b = () => { return 2 }\n"
    symbols = [c.symbol for c in no_merge.extract(write("x.js", js))]
    assert "a" in symbols
    assert "b" in symbols


def test_rust_and_go_are_parsed(extractors: ExtractorSet, write: Writer) -> None:
    rs = "struct S;\nimpl S {\n    fn run(&self) {}\n}\nfn main() {}\n"
    assert "main" in extractors.extract(write("a.rs", rs))[0].text
    go = "package main\n\ntype T struct{}\n\nfunc (t T) Run() {}\n\nfunc main() {}\n"
    assert "main" in extractors.extract(write("a.go", go))[0].text


def test_loose_code_is_kept_as_module_chunk(no_merge: ExtractorSet, write: Writer) -> None:
    chunks = no_merge.extract(write("s.py", "X = 1\nY = 2\n\ndef f():\n    pass\n"))
    assert any(c.symbol == "<module>" for c in chunks)


def test_missing_parser_falls_back_to_line_chunks(
    extractors: ExtractorSet, write: Writer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(code._parsers, "python", None)
    chunks = extractors.extract(write("f.py", "x = 1\ny = 2\n"))
    assert chunks[0].kind == "code"
    assert chunks[0].symbol == ""


def test_parser_crash_falls_back(
    extractors: ExtractorSet, write: Writer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        def parse(self, _src: bytes) -> None:
            raise RuntimeError("grammar bug")

    monkeypatch.setitem(code._parsers, "python", Boom())
    assert extractors.extract(write("g.py", "x = 1\n"))[0].symbol == ""


def test_unknown_language_extension_uses_line_chunks(
    extractors: ExtractorSet, write: Writer
) -> None:
    assert extractors.extract(write("a.dart", "void main() {}\n"))


def test_comment_only_file_still_has_chunks(extractors: ExtractorSet, write: Writer) -> None:
    assert extractors.extract(write("only_comment.py", "# just a comment\n"))
