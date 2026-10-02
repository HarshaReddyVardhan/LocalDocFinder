"""Markdown: chunk by heading, keep the heading path as the symbol."""
import re
from typing import List

import indexer_config as cfg
from . import Chunk, merge_small, split_by_lines

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def extract_markdown(content: str) -> List[Chunk]:
    lines = content.splitlines()
    sections = []  # (path, start_idx, end_idx_exclusive)
    stack: List[tuple] = []  # (level, title)
    cur_start, cur_path = 0, ""
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else _HEADING.match(line)
        if not m:
            continue
        if i > cur_start:
            sections.append((cur_path, cur_start, i))
        level, title = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        cur_path = " > ".join(t for _, t in stack)
        cur_start = i
    if cur_start < len(lines):
        sections.append((cur_path, cur_start, len(lines)))

    chunks: List[Chunk] = []
    for path, a, b in sections:
        body = lines[a:b]
        if not "".join(body).strip():
            continue
        text = "\n".join(body)
        if len(text) <= cfg.MAX_CHUNK_CHARS:
            chunks.append(Chunk(text, "doc", path, a + 1, b))
        else:
            for t, s, e in split_by_lines(body, a + 1, max_chars=cfg.TARGET_CHUNK_CHARS * 2):
                chunks.append(Chunk(t, "doc", path, s, e))
    return merge_small(chunks)
