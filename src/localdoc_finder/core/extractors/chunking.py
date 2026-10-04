"""Text-splitting helpers shared by the extractors."""

from pathlib import Path

from localdoc_finder.core.extractors.base import Chunk, ExtractContext, ExtractError
from localdoc_finder.core.settings import ChunkingSettings

_BINARY_SNIFF_BYTES = 8192


def read_text(path: str) -> str:
    """Read a text file: reject binaries, tolerate odd encodings."""
    with open(path, "rb") as handle:  # noqa: PTH123  # plain bytes read; Path adds nothing
        raw = handle.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 is full of NULs, so check it first
        return raw.decode("utf-16", errors="replace")
    if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
        raise ExtractError("binary file")
    return raw.decode("utf-8-sig", errors="replace")


def split_by_lines(
    lines: list[str], first_line: int, max_chars: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Split ``lines`` into windows of at most ``max_chars`` sharing ``overlap`` lines.

    Returns ``(text, start_line, end_line)`` with 1-based numbers; ``first_line`` is the
    1-based number of ``lines[0]``.
    """
    out: list[tuple[str, int, int]] = []
    i, total = 0, len(lines)
    while i < total:
        size, j = 0, i
        while j < total and (size + len(lines[j]) + 1 <= max_chars or j == i):
            size += len(lines[j]) + 1
            j += 1
        text = "\n".join(lines[i:j])
        if len(text) > max_chars:  # one enormous line: hard-cut so it fits the context window
            out.extend(
                (text[k : k + max_chars], first_line + i, first_line + j - 1)
                for k in range(0, len(text), max_chars)
            )
        else:
            out.append((text, first_line + i, first_line + j - 1))
        if j >= total:
            break
        # An overlap larger than the window would step one line at a time and emit a chunk per
        # line (quadratic text); never share more than half of a window.
        i = max(j - min(overlap, (j - i) // 2), i + 1)
    return out


def split_text(
    cfg: ChunkingSettings, lines: list[str], first_line: int = 1, max_chars: int | None = None
) -> list[tuple[str, int, int]]:
    """``split_by_lines`` with the configured size and overlap."""
    return split_by_lines(lines, first_line, max_chars or cfg.max_chunk_chars, cfg.line_overlap)


def merge_small(chunks: list[Chunk], cfg: ChunkingSettings) -> list[Chunk]:
    """Greedily merge adjacent small chunks of one kind/page up to ``target_chunk_chars``."""
    out: list[Chunk] = []
    for chunk in chunks:
        prev = out[-1] if out else None
        if (
            prev
            and prev.kind == chunk.kind
            and prev.page == chunk.page
            and len(prev.text) < cfg.min_chunk_chars
            and len(prev.text) + len(chunk.text) + 1 <= cfg.target_chunk_chars
        ):
            names = [s for s in (prev.symbol, chunk.symbol) if s]
            prev.symbol = ", ".join(dict.fromkeys(", ".join(names).split(", ")))[:120]
            prev.text = prev.text + "\n" + chunk.text
            prev.end_line = chunk.end_line or prev.end_line
        else:
            out.append(Chunk(**chunk.__dict__))
    return out


def read_source(ctx: ExtractContext, path: Path) -> str:
    """Text of a source file; rejects oversized data dumps and generated/minified files."""
    if (
        path.suffix.lower() in ctx.scope_settings.data_exts
        and path.stat().st_size > ctx.chunking.max_data_file_kb * 1024
    ):
        raise ExtractError("data file too large")
    content = read_text(str(path))
    if ctx.scope.looks_generated(content):
        raise ExtractError("generated or minified file")
    return content
