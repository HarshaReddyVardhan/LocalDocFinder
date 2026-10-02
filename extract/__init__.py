"""Turn a file into chunks. `extract(path)` dispatches on extension."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import indexer_config as cfg


@dataclass
class Chunk:
    text: str
    kind: str = "doc"        # code | outline | doc | image
    symbol: str = ""         # e.g. "Class.method", a heading path, "slide 3"
    start_line: int = 0      # 1-based, 0 = n/a
    end_line: int = 0
    page: int = 0            # PDF page / slide number, 0 = n/a


class ExtractError(Exception):
    """File could not be turned into text (corrupt, encrypted, ...). Not retried."""


def read_text(path: str) -> str:
    """Read a text file; reject binaries; tolerate odd encodings."""
    with open(path, "rb") as f:
        raw = f.read()
    if b"\x00" in raw[:8192]:
        raise ExtractError("binary file")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def looks_generated(text: str) -> bool:
    """Minified / generated files: very long lines or a generated-code header."""
    if not text:
        return False
    head = "\n".join(text.splitlines()[:10]).lower()
    if any(m in head for m in cfg.GENERATED_MARKERS):
        return True
    lines = text.count("\n") + 1
    return len(text) / lines > cfg.MAX_AVG_LINE_LENGTH


def split_by_lines(lines: List[str], first_line: int, max_chars: int = None,
                   overlap: int = None) -> List[tuple]:
    """Split lines into windows <= max_chars with `overlap` lines shared.

    Returns (text, start_line, end_line) with 1-based line numbers; first_line is the
    1-based number of lines[0].
    """
    max_chars = max_chars or cfg.MAX_CHUNK_CHARS
    overlap = cfg.LINE_OVERLAP if overlap is None else overlap
    out = []
    i, n = 0, len(lines)
    while i < n:
        size, j = 0, i
        while j < n and (size + len(lines[j]) + 1 <= max_chars or j == i):
            size += len(lines[j]) + 1
            j += 1
        # A single enormous line is hard-cut so it can never exceed the context window.
        text = "\n".join(lines[i:j])
        if len(text) > max_chars:
            for k in range(0, len(text), max_chars):
                out.append((text[k:k + max_chars], first_line + i, first_line + j - 1))
        else:
            out.append((text, first_line + i, first_line + j - 1))
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return out


def merge_small(chunks: List[Chunk]) -> List[Chunk]:
    """Greedily merge adjacent small chunks of the same kind up to TARGET_CHUNK_CHARS."""
    out: List[Chunk] = []
    for c in chunks:
        prev = out[-1] if out else None
        if (prev and prev.kind == c.kind and prev.page == c.page
                and len(prev.text) < cfg.MIN_CHUNK_CHARS
                and len(prev.text) + len(c.text) + 1 <= cfg.TARGET_CHUNK_CHARS):
            names = [s for s in (prev.symbol, c.symbol) if s]
            prev.symbol = ", ".join(dict.fromkeys(", ".join(names).split(", ")))[:120]
            prev.text = prev.text + "\n" + c.text
            prev.end_line = c.end_line or prev.end_line
        else:
            out.append(Chunk(**c.__dict__))
    return out


def finalize(chunks: List[Chunk], limit: int = None) -> List[Chunk]:
    """Drop blanks, hard-cap text size and chunk count."""
    out = []
    for c in chunks:
        c.text = c.text.strip()
        if not c.text:
            continue
        if len(c.text) > cfg.MAX_CHUNK_CHARS:
            c.text = c.text[: cfg.MAX_CHUNK_CHARS]
        out.append(c)
    return out[: (limit or cfg.MAX_CHUNKS_PER_FILE)]


def extract(path: str) -> List[Chunk]:
    """Dispatch on extension. Raises ExtractError for unreadable files."""
    p = Path(path)
    ext = p.suffix.lower()
    name = p.name.lower()

    if ext == ".pdf":
        from . import pdf
        return finalize(pdf.extract_pdf(path), cfg.MAX_CHUNKS_PER_DOC)
    if ext in (".docx", ".pptx"):
        from . import office
        return finalize(office.extract_office(path), cfg.MAX_CHUNKS_PER_DOC)
    if ext in cfg.IMAGE_EXTS:
        from . import image
        return finalize(image.extract_image(path))
    if ext in (".md", ".markdown", ".mdc"):
        from . import markdown
        return finalize(markdown.extract_markdown(read_text(path)))
    if ext == ".ipynb":
        from . import text
        return finalize(text.extract_notebook(read_text(path)))
    if ext == ".svg":
        from . import text
        return finalize(text.extract_svg(read_text(path)))

    if ext in cfg.DATA_EXTS and os.path.getsize(path) > cfg.MAX_DATA_FILE_KB * 1024:
        raise ExtractError("data file too large")
    content = read_text(path)
    if ext == ".rtf":
        from . import text
        content = text.strip_rtf(content)
    if looks_generated(content):
        raise ExtractError("generated or minified file")

    if ext in cfg.CODE_EXTS:
        from . import code
        return finalize(code.extract_code(content, path, ext))
    from . import text
    return finalize(text.extract_text(content, name))
