"""Plain text, config/data files, notebooks, SVG, RTF."""
import json
import os
import re
from typing import List

from . import Chunk, ExtractError, split_by_lines

_PROSE_EXTS = {".txt", ".rtf", ".tex", ""}
_TEXT_CHUNK_CHARS = 2500


def extract_text(content: str, name: str) -> List[Chunk]:
    ext = os.path.splitext(name)[1].lower()
    kind = "doc" if ext in _PROSE_EXTS and name not in ("dockerfile", "makefile") else "code"
    lines = content.splitlines()
    return [Chunk(t, kind, "", s, e)
            for t, s, e in split_by_lines(lines, 1, max_chars=_TEXT_CHUNK_CHARS, overlap=3)]


def strip_rtf(rtf: str) -> str:
    rtf = re.sub(r"\\'[0-9a-fA-F]{2}", "", rtf)
    rtf = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", rtf)
    rtf = re.sub(r"[{}]", "", rtf)
    return rtf


def extract_notebook(raw: str) -> List[Chunk]:
    """Jupyter: markdown + code cells only (outputs hold huge base64 blobs, so they are skipped)."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ExtractError(f"invalid notebook: {e}")
    chunks: List[Chunk] = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            continue
        kind = "doc" if cell.get("cell_type") == "markdown" else "code"
        for t, s, e in split_by_lines(src.splitlines(), 1):
            chunks.append(Chunk(t, kind, f"cell {i}", s, e))
    return chunks


def extract_svg(raw: str) -> List[Chunk]:
    """SVG is mostly path data; keep only the human-readable parts."""
    parts = re.findall(r"<(?:text|title|desc|tspan)[^>]*>([^<]+)</", raw)
    text = " ".join(p.strip() for p in parts if p.strip())
    if len(text) < 3:
        raise ExtractError("svg without text")
    return [Chunk(text[:4000], "doc", "svg text")]
