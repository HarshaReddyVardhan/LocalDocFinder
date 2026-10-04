"""Markdown: chunk by heading and keep the heading path as the symbol."""

import re
from collections.abc import Iterable
from pathlib import Path

from localdoc_finder.core.extractors.base import KIND_DOC, Chunk, Extractor, register_extractor
from localdoc_finder.core.extractors.chunking import merge_small, read_text, split_by_lines

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_MARKDOWN_EXTS = {".md", ".markdown", ".mdc"}


def heading_sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """``(heading path, start index, end index exclusive)`` per section; fences are opaque."""
    sections: list[tuple[str, int, int]] = []
    stack: list[tuple[int, str]] = []
    start, path = 0, ""
    in_fence = False
    for index, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if not match:
            continue
        if index > start:
            sections.append((path, start, index))
        level, title = len(match.group(1)), match.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = " > ".join(t for _, t in stack)
        start = index
    if start < len(lines):
        sections.append((path, start, len(lines)))
    return sections


@register_extractor("markdown")
class MarkdownExtractor(Extractor):
    name = "markdown"
    priority = 50

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _MARKDOWN_EXTS

    def extract(self, path: Path) -> Iterable[Chunk]:
        cfg = self.ctx.chunking
        lines = read_text(str(path)).splitlines()
        chunks: list[Chunk] = []
        for heading, begin, end in heading_sections(lines):
            body = lines[begin:end]
            if not "".join(body).strip():
                continue
            text = "\n".join(body)
            if len(text) <= cfg.max_chunk_chars:
                chunks.append(Chunk(text, KIND_DOC, heading, begin + 1, end))
                continue
            chunks.extend(
                Chunk(t, KIND_DOC, heading, s, e)
                for t, s, e in split_by_lines(
                    body, begin + 1, cfg.target_chunk_chars * 2, cfg.line_overlap
                )
            )
        return merge_small(chunks, cfg)
