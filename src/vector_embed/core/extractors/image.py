"""Images become text (OCR + optional caption) so everything shares one embedding space."""

import base64
import io
import logging
from collections.abc import Iterable
from pathlib import Path

import imagehash
import xxhash
from PIL import Image

from vector_embed.core.extractors.base import (
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
_JPEG_QUALITY = 85
_THUMB_QUALITY = 80


class OllamaCaptioner:
    """One-sentence captions from a small vision model (opt-in, run during idle indexing)."""

    def __init__(self, client: Untyped, model: str) -> None:
        self._client = client
        self._model = model

    def __call__(self, image: Image.Image) -> str:
        small = image.convert("RGB")
        small.thumbnail(_CAPTION_SIZE)
        buffer = io.BytesIO()
        small.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
        try:
            reply = self._client.chat(
                model=self._model,
                keep_alive=0,  # free the VRAM right after
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


def thumbnail_path(thumbs_dir: Path, image_path: Path) -> Path:
    """Where ``make_thumbnail`` stores the thumbnail of ``image_path`` (keyed by file content)."""
    return thumbs_dir / f"{xxhash.xxh3_64_hexdigest(image_path.read_bytes())}.jpg"


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


def image_chunk(
    ctx: ExtractContext, data: bytes, page: int, seen: set[str], label: str
) -> Chunk | None:
    """A chunk for an embedded image (PDF/Office); skips duplicates, tiny or unreadable ones."""
    key = xxhash.xxh3_64_hexdigest(data)
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

    def extract(self, path: Path) -> Iterable[Chunk]:
        Image.MAX_IMAGE_PIXELS = self.ctx.images.max_decode_pixels
        try:
            image = Image.open(path)
            image.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ExtractError(f"unreadable image: {exc}") from exc
        width, height = image.size
        min_pixels = self.ctx.images.min_pixels
        if width < min_pixels and height < min_pixels:
            raise ExtractError("image too small")
        make_thumbnail(self.ctx, image, xxhash.xxh3_64_hexdigest(path.read_bytes()))
        body = describe(self.ctx, image)
        # Even with no text, the file name makes the image findable by name.
        text = f"Image file: {path.name} ({width}x{height})\n{body}".strip()
        return [Chunk(text, KIND_IMAGE, path.name)]
