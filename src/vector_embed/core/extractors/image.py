"""Images become text (OCR + optional caption) so everything shares one embedding space."""

import base64
import io
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

import imagehash
import xxhash
from PIL import Image

from vector_embed.core.extractors.base import (
    KIND_DOC,
    KIND_IMAGE,
    Chunk,
    ExtractContext,
    ExtractError,
    Extractor,
    Untyped,
    register_extractor,
)

logger = logging.getLogger(__name__)

_MIN_OCR_TEXT = 4
_CAPTION_SIZE = (896, 896)
_CAPTION_PROMPT = (
    "Describe this image in one or two sentences, mentioning any visible text, "
    "diagrams, UI, objects and scene."
)
_WORKING_SIZE = 4000  # JPEG draft target: the OCR limit, above the thumbnail and caption sizes
_LOOK_CACHE_SIZE = 256  # images remembered by appearance within one extractor
_JPEG_QUALITY = 85
_CAPTION_KEEP_ALIVE = "5m"  # between images of one pass; released explicitly after it
_THUMB_QUALITY = 80
_MULTI_PAGE_EXTS = frozenset({".tif", ".tiff"})  # scanners save documents as these


class OllamaCaptioner:
    """One-sentence captions from a small vision model (opt-in, run during idle indexing).

    The vision model stays loaded between the images of one extraction pass, and ``release``
    unloads it before embedding starts: it must never share the GPU with the embedder.
    """

    def __init__(
        self,
        client: Untyped,
        model: str,
        *,
        free_gpu: Callable[[], None] = lambda: None,
        keep_alive: str = _CAPTION_KEEP_ALIVE,
    ) -> None:
        self._client = client
        self._model = model
        self._free_gpu = free_gpu
        self._keep_alive = keep_alive
        self._resident = False

    def __call__(self, image: Image.Image) -> str:
        small = image.convert("RGB")
        small.thumbnail(_CAPTION_SIZE)
        buffer = io.BytesIO()
        small.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
        try:
            if not self._resident:
                self._free_gpu()  # the embedder leaves before the vision model arrives
                self._resident = True
            reply = self._client.chat(
                model=self._model,
                keep_alive=self._keep_alive,
                messages=[
                    {
                        "role": "user",
                        "content": _CAPTION_PROMPT,
                        "images": [base64.b64encode(buffer.getvalue()).decode()],
                    }
                ],
            )
            return " ".join(str(reply["message"]["content"]).split())
        except Exception:  # captioning is best-effort enrichment
            logger.debug("image: caption failed", exc_info=True)
            return ""

    def release(self) -> None:
        """Unload the vision model now (``keep_alive=0``); a no-op when nothing was captioned."""
        if not self._resident:
            return
        self._resident = False
        try:
            self._client.generate(model=self._model, prompt="", keep_alive=0)
        except Exception:  # best effort: the server may be gone already
            logger.debug("image: caption model unload failed", exc_info=True)


def phash(image: Image.Image) -> str:
    """Perceptual hash, used to deduplicate near-identical images."""
    return str(imagehash.phash(image))


def make_thumbnail(ctx: ExtractContext, image: Image.Image, file_hash: str) -> Path | None:
    """Save a UI thumbnail under ``ctx.thumbs_dir``; ``None`` when disabled or it fails."""
    if ctx.thumbs_dir is None:
        return None
    target = ctx.thumbs_dir / f"{file_hash}.jpg"
    try:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            thumb = image.convert("RGB")
            size = ctx.images.thumbnail_size
            thumb.thumbnail((size, size))
            thumb.save(target, format="JPEG", quality=_THUMB_QUALITY)
    except OSError:
        logger.debug("image: thumbnail failed for %s", file_hash, exc_info=True)
        return None
    return target


def thumbnail_key(path: str | Path, mtime_ns: int, size: int) -> str:
    """Name of an image's thumbnail: its path and version, no file read needed.

    Built from what the manifest already records (mtime and size), so the indexer, the results
    list and the cleanup after a delete all compute the same key without opening the image.
    """
    return xxhash.xxh3_64_hexdigest(f"{os.path.normcase(path)}|{mtime_ns}|{size}".encode())


def thumbnail_path(thumbs_dir: Path, image_path: Path) -> Path:
    """Where ``make_thumbnail`` stores the thumbnail of ``image_path`` (in its current state)."""
    info = image_path.stat()  # raises OSError for a missing file: the caller has no thumbnail
    return thumbs_dir / f"{thumbnail_key(image_path, info.st_mtime_ns, info.st_size)}.jpg"


def remove_thumbnail(thumbs_dir: Path | None, path: str, mtime_ns: int, size: int) -> None:
    """Delete the thumbnail of one version of an image (it was deleted or replaced)."""
    if thumbs_dir is not None:
        (thumbs_dir / f"{thumbnail_key(path, mtime_ns, size)}.jpg").unlink(missing_ok=True)


def describe(ctx: ExtractContext, image: Image.Image, with_caption: bool = True) -> str:
    """OCR text plus an optional caption; ``""`` when the image carries neither."""
    parts: list[str] = []
    text = ctx.ocr.ocr_image(image)
    if len(text) >= _MIN_OCR_TEXT:
        parts.append(f"Text in image: {text}")
    if with_caption and ctx.images.enable_captions and ctx.captioner is not None:
        caption = ctx.captioner(image)
        if caption:
            parts.append(f"Caption: {caption}")
    return "\n".join(parts)


def image_key(data: bytes) -> str:
    """Content hash used to skip an image that already appeared in the same document."""
    return xxhash.xxh3_64_hexdigest(data)


def image_chunk(
    ctx: ExtractContext, data: bytes, page: int, seen: set[str], label: str
) -> Chunk | None:
    """A chunk for an embedded image (PDF/Office); skips duplicates, tiny or unreadable ones."""
    key = image_key(data)
    if key in seen:
        return None
    seen.add(key)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if min(image.size) < ctx.images.min_pixels:
                return None
            body = describe(ctx, image)
    except (OSError, ValueError, Image.DecompressionBombError):
        logger.debug("image: cannot decode embedded image %s", label, exc_info=True)
        return None
    return Chunk(body, KIND_IMAGE, label, page=page) if body else None


@register_extractor("image")
class ImageExtractor(Extractor):
    name = "image"
    priority = 40

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.ctx.scope_settings.image_exts

    def __init__(self, ctx: ExtractContext) -> None:
        super().__init__(ctx)
        # perceptual hash -> text, for the images of this run: a copy is not OCR'd again
        self._by_look: OrderedDict[str, str] = OrderedDict()

    def extract(self, path: Path) -> Iterable[Chunk]:
        info = path.stat()
        try:
            image = Image.open(path)  # reads the header only: the size is known before decoding
            width, height = image.size
            if width * height > self.ctx.images.max_decode_pixels:
                raise ExtractError(f"image too large to decode safely ({width}x{height})")
            # JPEGs can be decoded at a reduced size, much faster; nothing below needs more.
            scale = _WORKING_SIZE / max(width, height)
            if scale < 1:  # keeps the aspect ratio, so the decoder can halve both sides
                image.draft("RGB", (max(1, round(width * scale)), max(1, round(height * scale))))
            image.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ExtractError(f"unreadable image: {exc}") from exc
        min_pixels = self.ctx.images.min_pixels
        if width < min_pixels and height < min_pixels:
            raise ExtractError("image too small")
        key = thumbnail_key(path, info.st_mtime_ns, info.st_size)
        make_thumbnail(self.ctx, image, key)
        if path.suffix.lower() in _MULTI_PAGE_EXTS and getattr(image, "n_frames", 1) > 1:
            return self._pages(path, image)
        body = self._described(image)
        # Even with no text, the file name makes the image findable by name.
        text = f"Image file: {path.name} ({width}x{height})\n{body}".strip()
        return [Chunk(text, KIND_IMAGE, path.name)]

    def _pages(self, path: Path, image: Image.Image) -> list[Chunk]:
        """A multi-page scan: one chunk per page, like a scanned PDF."""
        pages = min(getattr(image, "n_frames", 1), self.ctx.images.max_scanned_pages)
        chunks = [Chunk(f"Scanned document: {path.name} ({pages} pages)", KIND_IMAGE, path.name)]
        for number in range(1, pages + 1):
            try:
                image.seek(number - 1)
                image.load()
            except (OSError, EOFError, ValueError):  # a damaged page ends the readable part
                logger.debug("image: cannot read page %d of %s", number, path.name)
                break
            text = describe(self.ctx, image, with_caption=False)
            if text:
                chunks.append(Chunk(text, KIND_DOC, f"page {number}", page=number))
        return chunks

    def _described(self, image: Image.Image) -> str:
        """OCR and caption, reusing the result for an image that looks identical to one seen."""
        look = phash(image)
        cached = self._by_look.get(look)
        if cached is not None:
            self._by_look.move_to_end(look)
            return cached
        body = describe(self.ctx, image)
        self._by_look[look] = body
        if len(self._by_look) > _LOOK_CACHE_SIZE:
            self._by_look.popitem(last=False)
        return body
