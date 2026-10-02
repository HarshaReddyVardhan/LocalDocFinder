"""PDF: text per page (PyMuPDF), OCR for scanned pages, OCR of embedded images."""
import io
from typing import List

import pymupdf
import xxhash
from PIL import Image

import indexer_config as cfg
from . import Chunk, ExtractError, split_by_lines
from . import ocr

pymupdf.TOOLS.mupdf_display_errors(False)

_SCANNED_PAGE_MIN_CHARS = 30
_SCAN_DPI = 150


def _page_chunks(text: str, pno: int) -> List[Chunk]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= cfg.MAX_CHUNK_CHARS:
        return [Chunk(text, "doc", f"p.{pno}", page=pno)]
    return [Chunk(t, "doc", f"p.{pno}", page=pno)
            for t, _, _ in split_by_lines(text.splitlines(), 1, max_chars=cfg.TARGET_CHUNK_CHARS * 2)]


def extract_pdf(path: str) -> List[Chunk]:
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise ExtractError(f"cannot open pdf: {e}")
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ExtractError("encrypted pdf")
        chunks: List[Chunk] = []
        budget = cfg.MAX_IMAGES_PER_DOC  # OCR operations (scanned pages + embedded images)
        seen = set()
        for pno, page in enumerate(doc, 1):
            text = page.get_text("text", sort=True)
            scanned = len(text.strip()) < _SCANNED_PAGE_MIN_CHARS and bool(page.get_images())
            if scanned:
                if budget > 0:
                    budget -= 1
                    pix = page.get_pixmap(dpi=_SCAN_DPI)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    text = ocr.ocr_image(img)
                chunks += _page_chunks(text, pno)
                continue
            chunks += _page_chunks(text, pno)
            if budget <= 0:
                continue
            for info in page.get_images(full=True):
                if budget <= 0:
                    break
                xref = info[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    data = doc.extract_image(xref)
                except Exception:
                    continue
                if not data or min(data.get("width", 0), data.get("height", 0)) < cfg.MIN_IMAGE_PIXELS:
                    continue
                key = xxhash.xxh3_64_hexdigest(data["image"])
                if key in seen:
                    continue
                seen.add(key)
                budget -= 1
                try:
                    with Image.open(io.BytesIO(data["image"])) as im:
                        im.load()
                        from .image import describe
                        body = describe(im, "figure")
                except Exception:
                    continue
                if body:
                    chunks.append(Chunk(body, "image", f"figure on p.{pno}", page=pno))
        return chunks
    finally:
        doc.close()
