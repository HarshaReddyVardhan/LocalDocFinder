"""PDF: text per page (PyMuPDF), OCR for scanned pages and for embedded figures."""

import io
import logging
from collections.abc import Iterable
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

logger = logging.getLogger(__name__)

pymupdf.TOOLS.mupdf_display_errors(False)

_SCANNED_PAGE_MIN_CHARS = 30
_SCAN_DPI = 200  # small print needs it; a letter page stays under the OCR limit


def _page_chunks(ctx: ExtractContext, text: str, page_no: int) -> list[Chunk]:
    text = text.strip()
    if not text:
        return []
    cfg = ctx.chunking
    symbol = f"p.{page_no}"
    if len(text) <= cfg.max_chunk_chars:
        return [Chunk(text, KIND_DOC, symbol, page=page_no)]
    return [
        Chunk(part, KIND_DOC, symbol, page=page_no)
        for part, _, _ in split_by_lines(
            text.splitlines(), 1, cfg.target_chunk_chars * 2, cfg.line_overlap
        )
    ]


def _has_picture(page: Untyped) -> bool:
    """Any picture on the page, inline ones included (scanners often write those)."""
    return bool(page.get_image_info())


def _ocr_scanned_page(ctx: ExtractContext, page: Untyped) -> str:
    pixmap = page.get_pixmap(dpi=_SCAN_DPI)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return ctx.ocr.ocr_image(image)


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
            return self._pages(doc)
        finally:
            doc.close()

    def _pages(self, doc: Untyped) -> list[Chunk]:
        ctx = self.ctx
        chunks: list[Chunk] = []
        # Separate allowances: a long scanned document must not leave its figures un-read.
        page_budget = ctx.images.max_scanned_pages  # whole-page OCR
        budget = ctx.images.max_per_doc  # embedded figures
        seen: set[str] = set()
        for page_no, page in enumerate(doc, 1):
            text = page.get_text("text", sort=True)
            scanned = len(text.strip()) < _SCANNED_PAGE_MIN_CHARS and _has_picture(page)
            if scanned:
                if page_budget > 0:
                    page_budget -= 1
                    text = _ocr_scanned_page(ctx, page)
                chunks += _page_chunks(ctx, text, page_no)
                continue
            chunks += _page_chunks(ctx, text, page_no)
            if budget > 0:
                figures, budget = _figure_chunks(ctx, doc, page, page_no, seen=seen, budget=budget)
                chunks += figures
        return chunks
