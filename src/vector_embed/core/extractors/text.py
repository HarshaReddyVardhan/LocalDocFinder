"""Plain text, config/data files, Jupyter notebooks and SVG."""

import json
import re
from collections.abc import Iterable
from pathlib import Path

from vector_embed.core.extractors.base import (
    KIND_CODE,
    KIND_DOC,
    Chunk,
    ExtractError,
    Extractor,
    register_extractor,
)
from vector_embed.core.extractors.chunking import read_source, read_text, split_by_lines

_PROSE_EXTS = {".txt", ".tex", ""}
_CODE_FILENAMES = {"dockerfile", "makefile"}
_TEXT_CHUNK_CHARS = 2500
_TEXT_OVERLAP = 3
_SVG_MAX_CHARS = 4000
_SVG_MIN_CHARS = 3


@register_extractor("text")
class TextExtractor(Extractor):
    """Fallback for every other allowed text-like file (config, prose, scripts without a parser)."""

    name = "text"
    priority = 900

    def supports(self, path: Path) -> bool:
        cfg = self.ctx.scope_settings
        name = path.name.lower()
        ext = path.suffix.lower()
        return (
            ext in cfg.text_exts
            or name in cfg.text_filenames
            or (ext == ".jsonl" and cfg.index_ai_transcripts)
        )

    def extract(self, path: Path) -> Iterable[Chunk]:
        content = read_source(self.ctx, path)
        is_prose = path.suffix.lower() in _PROSE_EXTS and path.name.lower() not in _CODE_FILENAMES
        kind = KIND_DOC if is_prose else KIND_CODE
        return [
            Chunk(text, kind, "", start, end)
            for text, start, end in split_by_lines(
                content.splitlines(), 1, _TEXT_CHUNK_CHARS, _TEXT_OVERLAP
            )
        ]


@register_extractor("notebook")
class NotebookExtractor(Extractor):
    """Jupyter: markdown + code cells only (outputs hold huge base64 blobs)."""

    name = "notebook"
    priority = 50

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".ipynb"

    def extract(self, path: Path) -> Iterable[Chunk]:
        try:
            notebook = json.loads(read_text(str(path)))
        except json.JSONDecodeError as exc:
            raise ExtractError(f"invalid notebook: {exc}") from exc
        chunks: list[Chunk] = []
        cfg = self.ctx.chunking
        for index, cell in enumerate(notebook.get("cells", []), 1):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if not source.strip():
                continue
            kind = KIND_DOC if cell.get("cell_type") == "markdown" else KIND_CODE
            chunks.extend(
                Chunk(text, kind, f"cell {index}", start, end)
                for text, start, end in split_by_lines(
                    source.splitlines(), 1, cfg.max_chunk_chars, cfg.line_overlap
                )
            )
        return chunks


@register_extractor("svg")
class SvgExtractor(Extractor):
    """SVG is mostly path data; keep only the human-readable text."""

    name = "svg"
    priority = 50

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".svg"

    def extract(self, path: Path) -> Iterable[Chunk]:
        raw = read_text(str(path))
        parts = re.findall(r"<(?:text|title|desc|tspan)[^>]*>([^<]+)</", raw)
        text = " ".join(p.strip() for p in parts if p.strip())
        if len(text) < _SVG_MIN_CHARS:
            raise ExtractError("svg without text")
        return [Chunk(text[:_SVG_MAX_CHARS], KIND_DOC, "svg text")]
