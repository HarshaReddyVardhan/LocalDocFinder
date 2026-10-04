"""PDF: text per page (PyMuPDF), OCR for scanned pages and for embedded figures.

Lines repeated on most pages (running headers and footers, a scanner app's watermark) are
dropped before chunking: they say nothing about any one page, and they made every page of a
junk scan look alike. A scanned page whose OCR found little or no real text is kept as a
low-content chunk, findable by keyword but never by meaning.
"""

import io
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import xxhash
from PIL import Image

from localdoc_finder.core.extractors.base import (
    KIND_DOC,
    KIND_IMAGE,
    Chunk,
    ExtractContext,
    ExtractError,
    Extractor,
    Untyped,
    register_extractor,
)
from localdoc_finder.core.extractors.chunking import split_by_lines
from localdoc_finder.core.extractors.image import describe
from localdoc_finder.core.extractors.ocr import is_low_content

logger = logging.getLogger(__name__)

pymupdf.TOOLS.mupdf_display_errors(False)

_SCANNED_PAGE_MIN_CHARS = 30
_SCAN_DPI = 200  # small print needs it; a letter page stays under the OCR limit
REPEATED_LINE_SHARE = 0.5  # a line on at least this share of the pages (and on two) is furniture
_DIGITS = re.compile(r"\d+")


@dataclass
class _Page:
    number: int
    lines: list[str]
    scanned: bool = False
    read: bool = True  # False for a scanned page past the OCR allowance
    figures: list[Chunk] = field(default_factory=list)


def _line_key(line: str) -> str:
    """Lines compared across pages: case, spacing and numbers ignored, so "Page 3 of 9" on page 3
    and "Page 4 of 9" on page 4 are the same footer."""
    return _DIGITS.sub("#", " ".join(line.lower().split()))


def repeated_lines(pages: Sequence[Sequence[str]], share: float = REPEATED_LINE_SHARE) -> set[str]:
    """Keys (``_line_key``) of the lines found on at least ``share`` of the pages and on two."""
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({_line_key(line) for line in lines if line.strip()})
    needed = max(2.0, share * len(pages))
    return {key for key, count in counts.items() if count >= needed}


def _metadata_lines(doc: Untyped) -> list[str]:
    """The document's own title and subject, which often name it better than its first page."""
    meta = doc.metadata or {}
    return [
        f"{label}: {value.strip()}"
        for label, value in (("Title", meta.get("title")), ("Subject", meta.get("subject")))
        if isinstance(value, str) and value.strip()
    ]


def _page_chunks(
    ctx: ExtractContext, text: str, page_no: int, *, low_content: bool = False
) -> list[Chunk]:
    text = text.strip()
    if not text:
        return []
    cfg = ctx.chunking
    symbol = f"p.{page_no}"
    if len(text) <= cfg.max_chunk_chars:
        return [Chunk(text, KIND_DOC, symbol, page=page_no, low_content=low_content)]
    return [
        Chunk(part, KIND_DOC, symbol, page=page_no, low_content=low_content)
        for part, _, _ in split_by_lines(
            text.splitlines(), 1, cfg.target_chunk_chars * 2, cfg.line_overlap
        )
    ]


def _has_picture(page: Untyped) -> bool:
    """Any picture on the page, inline ones included (scanners often write those)."""
    return bool(page.get_image_info())


def _ocr_scanned_page(ctx: ExtractContext, page: Untyped) -> list[str]:
    pixmap = page.get_pixmap(dpi=_SCAN_DPI)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return ctx.ocr.ocr_lines(image)


def _figure_chunks(
    ctx: ExtractContext,
    doc: Untyped,
    page: Untyped,
    page_no: int,
    *,
    seen: set[str],
    budget: int,
) -> tuple[list[Chunk], int]:
    """OCR the page's embedded figures; returns the chunks and the remaining OCR budget."""
    chunks: list[Chunk] = []
    min_pixels = ctx.images.min_pixels
    for info in page.get_images(full=True):
        if budget <= 0:
            break
        xref = str(info[0])
        if xref in seen:
            continue
        seen.add(xref)
        try:
            data = doc.extract_image(info[0])
        except Exception:  # corrupt object in the PDF: skip just this image
            logger.debug("pdf: cannot extract image %s", xref, exc_info=True)
            continue
        if not data or min(data.get("width", 0), data.get("height", 0)) < min_pixels:
            continue
        key = xxhash.xxh3_64_hexdigest(data["image"])
        if key in seen:
            continue
        seen.add(key)
        budget -= 1
        try:
            with Image.open(io.BytesIO(data["image"])) as image:
                image.load()
                body = describe(ctx, image)
        except (OSError, ValueError, Image.DecompressionBombError):
            logger.debug("pdf: skipping an unreadable or oversized figure on p.%d", page_no)
            continue
        if body:
            chunks.append(Chunk(body, KIND_IMAGE, f"figure on p.{page_no}", page=page_no))
    return chunks, budget


@register_extractor("pdf")
class PdfExtractor(Extractor):
    name = "pdf"
    priority = 30
    is_document = True

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(self, path: Path) -> Iterable[Chunk]:
        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # pymupdf raises several unrelated types for bad files
            raise ExtractError(f"cannot open pdf: {exc}") from exc
        try:
            if doc.needs_pass or doc.is_encrypted:
                raise ExtractError("encrypted pdf")
            chunks = self._chunks(doc)
            if not chunks:  # nothing readable: the file is still found by its name
                summary = f"PDF file: {path.name} ({doc.page_count} pages)"
                chunks = [Chunk(summary, KIND_DOC, low_content=True)]
            return chunks
        finally:
            doc.close()

    def _chunks(self, doc: Untyped) -> list[Chunk]:
        pages = self._read_pages(doc)
        furniture = repeated_lines([page.lines for page in pages])
        metadata = _metadata_lines(doc)
        chunks: list[Chunk] = []
        for page in pages:
            body = "\n".join(line for line in page.lines if _line_key(line) not in furniture)
            # Judged on the page alone: a scanner app's title must not make a blank scan count.
            low = page.scanned and (not page.read or is_low_content(body))
            if page.number == 1 and metadata:
                body = "\n".join([*metadata, body])
            chunks += _page_chunks(self.ctx, body, page.number, low_content=low)
            chunks += page.figures
        return chunks

    def _read_pages(self, doc: Untyped) -> list[_Page]:
        ctx = self.ctx
        pages: list[_Page] = []
        # Separate allowances: a long scanned document must not leave its figures un-read.
        page_budget = ctx.images.max_scanned_pages  # whole-page OCR
        budget = ctx.images.max_per_doc  # embedded figures
        seen: set[str] = set()
        for page_no, page in enumerate(doc, 1):
            text = page.get_text("text", sort=True)
            if len(text.strip()) < _SCANNED_PAGE_MIN_CHARS and _has_picture(page):
                read = page_budget > 0
                page_budget -= int(read)
                lines = _ocr_scanned_page(ctx, page) if read else text.splitlines()
                pages.append(_Page(page_no, lines, scanned=True, read=read))
                continue
            current = _Page(page_no, text.splitlines())
            if budget > 0:
                current.figures, budget = _figure_chunks(
                    ctx, doc, page, page_no, seen=seen, budget=budget
                )
            pages.append(current)
        return pages
